import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, Tuple
import time


class CryptoDataLoader:
    """
    Загрузка данных с GeckoTerminal API, Yahoo Finance и Fred API.
    """
    
    def __init__(self):
        self.base_url_gecko = "https://api.geckoterminal.com/api/v2"
        self.fred_base = "https://api.stlouisfed.org/fred/series/data"
        
    def load_crypto_prices(
        self,
        crypto_ids: List[str],
        days: int = 365,
        vs_currency: str = "usd"
    ) -> pd.DataFrame:
        """
        Load crypto prices via free CoinGecko endpoint (used by GeckoTerminal).
        """
        prices_data = {}
        
        for crypto_id in crypto_ids:
            url = f"{self.base_url_gecko}/coins/{crypto_id}/ohlcv/daily"
            params = {
                "vs_currency": vs_currency,
                "days": days
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' in data:
                        dates = []
                        closes = []
                        
                        for ohlcv in data['data']:
                            dates.append(pd.to_datetime(ohlcv, unit='ms'))
                            closes.append(float(ohlcv))  # Close price
                        
                        prices_data[crypto_id] = pd.Series(closes, index=dates)
                        print(f"✓ Loaded {crypto_id}: {len(closes)} days")
                else:
                    print(f"✗ Failed to load {crypto_id}: {response.status_code}")
            except Exception as e:
                print(f"✗ Error loading {crypto_id}: {e}")
            
            time.sleep(0.5)  # Rate limiting
        
        df = pd.DataFrame(prices_data)
        df = df.sort_index()
        return df.fillna(method='ffill').fillna(method='bfill')
    
    def load_macro_indicators(self, fred_api_key: str) -> pd.DataFrame:
        """
        Load Fed Rate, VIX, S&P500 via Fred API and Yahoo Finance.
        """
        macro_data = {}
        dates_index = None
        
        # 1. Federal Funds Rate
        try:
            url_fedfunds = self.fred_base
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': fred_api_key,
                'file_type': 'json'
            }
            response = requests.get(url_fedfunds, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                dates = []
                rates = []
                for obs in data['observations']:
                    try:
                        dates.append(pd.to_datetime(obs['date']))
                        rates.append(float(obs['value']))
                    except:
                        continue
                
                macro_data['fed_rate'] = pd.Series(rates, index=dates)
                print(f"✓ Loaded Fed Funds Rate: {len(rates)} obs")
        except Exception as e:
            print(f"✗ Error loading Fed Funds Rate: {e}")
        
        # 2. VIX Index
        try:
            vix = yf.download('^VIX', progress=False)
            macro_data['vix'] = vix['Close']
            print(f"✓ Loaded VIX: {len(vix)} days")
        except Exception as e:
            print(f"✗ Error loading VIX: {e}")
        
        # 3. S&P 500
        try:
            sp500 = yf.download('^GSPC', progress=False)
            macro_data['sp500_return'] = sp500['Close'].pct_change()
            print(f"✓ Loaded S&P500: {len(sp500)} days")
        except Exception as e:
            print(f"✗ Error loading S&P500: {e}")
        
        # Combine and resample to daily
        df_macro = pd.DataFrame(macro_data)
        df_macro = df_macro.resample('D').last().fillna(method='ffill')
        
        return df_macro
    
    def load_on_chain_data(self, crypto_ids: List[str]) -> pd.DataFrame:
        """
        Load on-chain data: active addresses, transaction count.
        Using GeckoTerminal DEX data or approximations.
        """
        on_chain_data = {}
        
        for crypto_id in crypto_ids:
            # Approximation: use volume/price as proxy for activity
            # In production, would use Glassnode, IntoTheBlock, or similar
            
            try:
                url = f"{self.base_url_gecko}/coins/{crypto_id}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' in data and 'attributes' in data['data']:
                        attrs = data['data']['attributes']
                        
                        on_chain_data[crypto_id] = {
                            'market_cap_rank': attrs.get('market_cap_rank', 0),
                            'volume_24h_usd': attrs.get('volume_usd', {}).get('usd', 0),
                            'liquidity': attrs.get('liquidity', 0)
                        }
                        print(f"✓ Loaded on-chain data for {crypto_id}")
            except Exception as e:
                print(f"✗ Error loading on-chain data for {crypto_id}: {e}")
            
            time.sleep(0.5)
        
        return pd.DataFrame(on_chain_data).T
