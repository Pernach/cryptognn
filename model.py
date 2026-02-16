import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, LayerNorm, global_mean_pool
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EdgeTransformer(nn.Module):
    """
    Edge-centric transformer для контекстного представления взаимодействий между активами.
    
    Входные данные: временные окна edge features
    Выходные данные: уточненные весовые коэффициенты и attention patterns
    """
    def __init__(self, edge_dim: int, hidden_dim: int = 64, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Linear projection из edge features в query space
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)
        
        # Transformer encoder для edge context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.edge_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection в refined edge weights
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Weight normalization [0, 1]
        )
        
        self.layer_norm = LayerNorm(hidden_dim)
    
    def forward(self, edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        edge_features: (num_edges, time_window, edge_dim)
        returns: (refined_weights, attention_weights)
        """
        batch_size, seq_len, feat_dim = edge_features.shape
        
        # Проекция
        x = self.edge_projection(edge_features)  # (batch, seq_len, hidden_dim)
        
        # Трансформер энкодер
        x = self.edge_transformer(x)  # (batch, seq_len, hidden_dim)
        x = self.layer_norm(x)
        
        # Глобальная агрегация временного окна
        x_agg = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Refined weights
        refined_weights = self.output_projection(x_agg)  # (batch, 1)
        
        # Attention weights (для интерпретации)
        attention_logits = torch.matmul(x, x.transpose(-2, -1))  # (batch, seq_len, seq_len)
        attention_weights = F.softmax(attention_logits.mean(dim=0), dim=-1)  # (seq_len, seq_len)
        
        return refined_weights.squeeze(-1), attention_weights


class CryptoGraphTransformer(nn.Module):
    """
    Гибридная архитектура: Local MPNN + Global Attention для справедливой оценки крипто.
    
    Архитектура GPS Layer (Graph Positional Structure):
    - Локальный MPNN: GAT для неигнорирования структуры графа
    - Глобальное внимание: Transformer для дальних зависимостей
    - Edge Transformer: Контекстные представления взаимодействий
    """
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_edge_transformer: bool = True
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_edge_transformer = use_edge_transformer
        
        # Edge Transformer (опциональный)
        if use_edge_transformer:
            self.edge_transformer = EdgeTransformer(edge_dim, hidden_dim // 2, num_heads, 2)
            edge_processing_dim = 1  # Refined weights
        else:
            edge_processing_dim = edge_dim
        
        # Позиционное кодирование графа (structural features)
        self.positional_encoding = nn.Linear(2, hidden_dim // 4)  # degree, eigenvector
        
        # GPS layers: гибрид MPNN + Transformer
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])
        
        for i in range(num_layers):
            in_dim = node_dim + hidden_dim // 4 if i == 0 else hidden_dim
            
            # Локальный MPNN: GAT
            self.layers.append(nn.ModuleDict({
                'gat': GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,  # Усреднение heads
                    edge_dim=edge_processing_dim,
                    dropout=dropout
                ),
                'transformer': TransformerConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout
                ),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
            }))
            
            self.layer_norms.append(nn.ModuleList([
                LayerNorm(hidden_dim),
                LayerNorm(hidden_dim),
                LayerNorm(hidden_dim)
            ]))
        
        # Output head для справедливой стоимости
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)  # Log price or normalized value
        )
        
        # Uncertainty head (для доверительных интервалов)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Положительное значение
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pos_encoding: torch.Tensor,
        edge_temporal: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        x: node features (num_nodes, node_dim)
        edge_index: (2, num_edges)
        edge_attr: (num_edges, edge_dim)
        pos_encoding: (num_nodes, 2) - structural features
        edge_temporal: (num_edges, time_window, edge_dim) - для edge transformer
        batch: batch assignment
        
        returns: (fair_values, uncertainties, attention_dict)
        """
        
        # Позиционное кодирование
        pos_emb = self.positional_encoding(pos_encoding)  # (num_nodes, hidden_dim//4)
        x = torch.cat([x, pos_emb], dim=-1)  # Augment node features
        
        attention_weights_all = []
        refined_edge_weights = None
        
        # Edge Transformer для контекстных представлений
        if self.use_edge_transformer and edge_temporal is not None:
            refined_edge_weights, edge_attention = self.edge_transformer(edge_temporal)
            edge_attr_refined = refined_edge_weights.unsqueeze(-1)  # (num_edges, 1)
            attention_weights_all.append(edge_attention)
        else:
            edge_attr_refined = edge_attr
        
        # GPS layers
        for layer_idx, layer_dict in enumerate(self.layers):
            x_input = x
            
            # Local MPNN (GAT)
            x_gat = layer_dict['gat'](
                x_input,
                edge_index,
                edge_attr=edge_attr_refined if edge_attr_refined is not None else None
            )
            x_gat = self.layer_norms[layer_idx](x_gat + x_input)
            x_gat = self.dropouts[layer_idx](x_gat)
            
            # Global Attention (Transformer)
            x_trans = layer_dict['transformer'](x_gat, edge_index)
            x_trans = self.layer_norms[layer_idx](x_trans + x_gat)
            x_trans = self.dropouts[layer_idx](x_trans)
            
            # FFN
            x_ffn = layer_dict['mlp'](x_trans)
            x = self.layer_norms[layer_idx](x_ffn + x_trans)
            x = self.dropouts[layer_idx](x)
        
        # Output predictions
        fair_values = self.output_head(x)  # (num_nodes, 1)
        uncertainties = self.uncertainty_head(x)  # (num_nodes, 1)
        
        # Attention analysis
        attention_dict = {
            'edge_attention': edge_attention if refined_edge_weights is not None else None,
            'edge_weights': refined_edge_weights
        }
        
        return fair_values, uncertainties, attention_dict


class CryptoDataProcessor:
    """
    Обработка данных из GeckoTerminal, Yahoo Finance и Fred API.
    Формирование node и edge features для GNN.
    """
    
    def __init__(self, crypto_list: List[str], macro_features: pd.DataFrame):
        """
        crypto_list: ['bitcoin', 'ethereum', ...]
        macro_features: DataFrame с Fed rate, VIX, S&P500
        """
        self.crypto_list = crypto_list
        self.macro_features = macro_features
        self.price_history = {}
        self.on_chain_features = {}
        self.scaler = StandardScaler()
        
    def compute_node_features(
        self,
        price_df: pd.DataFrame,
        on_chain_df: pd.DataFrame,
        market_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Формирование node features: (num_nodes, node_dim)
        """
        num_assets = len(self.crypto_list)
        node_features = []
        
        for asset in self.crypto_list:
            features = []
            
            # Price features
            if asset in price_df.columns:
                returns = price_df[asset].pct_change()
                features.extend([
                    returns.iloc[-1],  # 1d return
                    returns.iloc[-7:].std() if len(returns) > 7 else 0,  # 7d volatility
                    returns.iloc[-30:].std() if len(returns) > 30 else 0,  # 30d volatility
                    returns.mean() / (returns.std() + 1e-6),  # Sharpe ratio approx
                    (returns > 0).sum() / len(returns)  # Win rate
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # On-chain features (нормализованные)
            if asset in on_chain_df.columns:
                on_chain_vals = on_chain_df[asset].dropna().tail(1).values
                features.extend(list(on_chain_vals) if len(on_chain_vals) > 0 else *3)
            else:
                features.extend([0, 0, 0])
            
            # Market features
            if asset in market_df.columns:
                market_vals = market_df[asset].dropna().tail(1).values
                features.extend(list(market_vals) if len(market_vals) > 0 else *2)
            else:
                features.extend([0, 0])
            
            # Macro features (latest)
            if len(self.macro_features) > 0:
                macro_vals = self.macro_features.iloc[-1].values
                features.extend(list(macro_vals))
            else:
                features.extend(*3)
            
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        node_features = self.scaler.fit_transform(node_features)
        
        return node_features
    
    def compute_edge_features_and_graph(
        self,
        price_df: pd.DataFrame,
        correlation_threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Построение графа и edge features через корреляции и причинность.
        
        returns:
            edge_index: (2, num_edges)
            edge_attr: (num_edges, edge_dim)
            edge_temporal: (num_edges, time_window, edge_dim)
        """
        num_assets = len(self.crypto_list)
        edges = []
        edge_features = []
        edge_temporal_list = []
        
        # Временное окно для контекста
        window_size = 14  # 2 недели
        
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                asset_i = self.crypto_list[i]
                asset_j = self.crypto_list[j]
                
                if asset_i not in price_df.columns or asset_j not in price_df.columns:
                    continue
                
                # Корреляция цен
                returns_i = price_df[asset_i].pct_change().dropna()
                returns_j = price_df[asset_j].pct_change().dropna()
                
                # Обрезать до общего размера
                min_len = min(len(returns_i), len(returns_j))
                if min_len < 10:
                    continue
                
                returns_i = returns_i.iloc[-min_len:]
                returns_j = returns_j.iloc[-min_len:]
                
                corr, p_value = pearsonr(returns_i, returns_j)
                
                # Фильтр по значимости
                if abs(corr) < correlation_threshold or p_value > 0.05:
                    continue
                
                # Добавить обе направления (неориентированный граф)
                edges.append([i, j])
                edges.append([j, i])
                
                # Edge features (статические)
                spearman_corr, _ = spearmanr(returns_i, returns_j)
                
                # Скользящая волатильность взаимодействия
                price_i = price_df[asset_i].iloc[-min_len:].values
                price_j = price_df[asset_j].iloc[-min_len:].values
                
                interaction_vol = np.std(np.abs(np.diff(price_i) - np.diff(price_j)))
                
                edge_feat = np.array([
                    corr,
                    spearman_corr,
                    abs(corr),  # Absolute correlation strength
                    1 - p_value,  # Significance
                    interaction_vol
                ], dtype=np.float32)
                
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)  # For reverse direction
                
                # Temporal features (скользящее окно)
                edge_temporal_window = []
                for t in range(max(0, min_len - window_size), min_len):
                    slice_end = t + 1
                    slice_start = max(0, t - window_size + 1)
                    
                    ret_i = returns_i.iloc[slice_start:slice_end].values
                    ret_j = returns_j.iloc[slice_start:slice_end].values
                    
                    if len(ret_i) > 1:
                        step_corr, _ = pearsonr(ret_i, ret_j) if len(ret_i) > 1 else (0, 1)
                    else:
                        step_corr = 0
                    
                    temporal_feat = np.array([
                        step_corr,
                        np.std(ret_i) if len(ret_i) > 0 else 0,
                        np.std(ret_j) if len(ret_j) > 0 else 0
                    ], dtype=np.float32)
                    
                    edge_temporal_window.append(temporal_feat)
                
                # Pad to fixed window size
                while len(edge_temporal_window) < window_size:
                    edge_temporal_window.insert(0, np.zeros(3, dtype=np.float32))
                edge_temporal_window = edge_temporal_window[-window_size:]
                
                edge_temporal_list.append(np.array(edge_temporal_window))
                edge_temporal_list.append(np.array(edge_temporal_window))
        
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, 5), dtype=np.float32)
        edge_temporal = np.array(edge_temporal_list, dtype=np.float32) if edge_temporal_list else np.zeros((0, 14, 3), dtype=np.float32)
        
        return edge_index, edge_attr, edge_temporal
    
    def compute_positional_encoding(self, edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
        """
        Structural positional encoding для графа.
        """
        pos_encoding = np.zeros((num_nodes, 2), dtype=np.float32)
        
        # Degree centrality
        degree = np.zeros(num_nodes)
        if edge_index.shape > 0:
            for edge in edge_index.T:
                degree[edge] += 1
        
        pos_encoding[:, 0] = degree / (np.max(degree) + 1e-6)
        
        # Eigenvector centrality approximation (Katz centrality)
        pos_encoding[:, 1] = np.random.randn(num_nodes) * 0.1  # Random initialization
        
        return pos_encoding


class CryptoFairValueModel:
    """
    Полная pipeline для обучения и инференса справедливой оценки.
    """
    
    def __init__(
        self,
        crypto_list: List[str],
        device: str = 'cpu',
        hidden_dim: int = 128,
        num_layers: int = 4
    ):
        self.crypto_list = crypto_list
        self.device = torch.device(device)
        
        # Инициализация модели
        node_dim = 64  # Computed empirically
        edge_dim = 5
        
        self.model = CryptoGraphTransformer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_heads=8,
            num_layers=num_layers,
            dropout=0.1,
            use_edge_transformer=True
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.data_processor = None
        self.scaler_price = StandardScaler()
        
    def prepare_data(
        self,
        price_df: pd.DataFrame,
        on_chain_df: pd.DataFrame,
        market_df: pd.DataFrame,
        macro_features: pd.DataFrame
    ) -> torch.geometric.data.Data:
        """
        Подготовка данных в формат PyG.
        """
        self.data_processor = CryptoDataProcessor(self.crypto_list, macro_features)
        
        # Node features
        node_features = self.data_processor.compute_node_features(
            price_df, on_chain_df, market_df
        )
        
        # Edge features
        edge_index, edge_attr, edge_temporal = \
            self.data_processor.compute_edge_features_and_graph(price_df, correlation_threshold=0.2)
        
        # Positional encoding
        pos_encoding = self.data_processor.compute_positional_encoding(
            edge_index, len(self.crypto_list)
        )
        
        # Target values (normalized log prices)
        target_prices = []
        for asset in self.crypto_list:
            if asset in price_df.columns:
                last_price = price_df[asset].iloc[-1]
                target_prices.append([np.log(last_price + 1e-6)])
            else:
                target_prices.append()
        
        target_prices = np.array(target_prices, dtype=np.float32)
        self.scaler_price.fit(target_prices)
        target_prices = self.scaler_price.transform(target_prices)
        
        # Создать PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32) if edge_attr.shape > 0 else None,
            pos_encoding=torch.tensor(pos_encoding, dtype=torch.float32),
            y=torch.tensor(target_prices, dtype=torch.float32),
            edge_temporal=torch.tensor(edge_temporal, dtype=torch.float32) if len(edge_temporal) > 0 else None
        )
        
        return data.to(self.device)
    
    def train_step(self, data: torch.geometric.data.Data) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        fair_values, uncertainties, _ = self.model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            pos_encoding=data.pos_encoding,
            edge_temporal=data.edge_temporal
        )
        
        # Loss: MSE + uncertainty regularization
        loss_mse = F.mse_loss(fair_values, data.y)
        loss_uncertainty = (-uncertainties.log()).mean()  # Encourage low uncertainty
        
        loss = loss_mse + 0.1 * loss_uncertainty
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: torch.geometric.data.Data) -> Dict:
        """Evaluate model performance."""
        self.model.eval()
        
        with torch.no_grad():
            fair_values, uncertainties, attention = self.model(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                pos_encoding=data.pos_encoding,
                edge_temporal=data.edge_temporal
            )
        
        # Denormalize
        fair_values_denorm = self.scaler_price.inverse_transform(fair_values.cpu().numpy())
        fair_prices = np.exp(fair_values_denorm)
        
        market_prices = self.scaler_price.inverse_transform(data.y.cpu().numpy())
        market_prices = np.exp(market_prices)
        
        # Metrics
        mse = np.mean((fair_prices - market_prices) ** 2)
        mae = np.mean(np.abs(fair_prices - market_prices) / market_prices)
        r2 = 1 - np.sum((market_prices - fair_prices) ** 2) / np.sum((market_prices - market_prices.mean()) ** 2)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'fair_prices': fair_prices,
            'market_prices': market_prices,
            'uncertainties': uncertainties.cpu().numpy()
        }
    
    def predict_fair_value(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Predict fair values for crypto assets."""
        results = {}
        for asset, fair_price in zip(self.crypto_list, fair_prices.flatten()):
            results[asset] = fair_price
        return results

