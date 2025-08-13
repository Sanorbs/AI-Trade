import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from .websocket_client import RealTimeDataClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("WebSocket client not available, using simulation mode only")

class RealTimeDataManager:
    def __init__(self, api_key, tickers):
        self.api_key = api_key
        self.tickers = tickers
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        self.client = None
        self.latest_prices = {}
        self.price_history = {ticker: [] for ticker in tickers}
        self.is_connected = False
        
    def connect(self):
        """Connect to real-time data stream"""
        if not WEBSOCKET_AVAILABLE:
            print("WebSocket not available, using simulation mode")
            self.is_connected = True
            return True
            
        try:
            self.client = RealTimeDataClient(self.ws_url, self.tickers)
            self.client.connect()
            self.is_connected = True
            print(f"Connected to real-time data for {self.tickers}")
            return True
        except Exception as e:
            print(f"Failed to connect to real-time data: {e}")
            return False
    
    def get_latest_prices(self):
        """Get latest prices for all tickers"""
        if not self.is_connected:
            return None
            
        if not WEBSOCKET_AVAILABLE:
            # Return simulated prices
            return self.latest_prices
            
        prices = {}
        for ticker in self.tickers:
            data = self.client.get_latest(ticker)
            if data and 'price' in data:
                prices[ticker] = data['price']
                self.price_history[ticker].append({
                    'timestamp': datetime.now(),
                    'price': data['price']
                })
        return prices
    
    def get_price_dataframe(self, lookback_hours=24):
        """Get price data as DataFrame for the last N hours"""
        if not self.price_history:
            return None
            
        data = []
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        for ticker in self.tickers:
            for entry in self.price_history[ticker]:
                if entry['timestamp'] > cutoff_time:
                    data.append({
                        'timestamp': entry['timestamp'],
                        'ticker': ticker,
                        'price': entry['price']
                    })
        
        if data:
            df = pd.DataFrame(data)
            return df.pivot(index='timestamp', columns='ticker', values='price')
        return None
    
    def simulate_real_time_data(self, historical_data, num_updates=100):
        """Simulate real-time data updates for testing"""
        print("Simulating real-time data updates...")
        
        # Use historical data as base
        base_data = historical_data.copy()
        
        for i in range(num_updates):
            # Simulate price changes
            for ticker in self.tickers:
                if ticker in base_data.columns:
                    # Add some random noise to simulate real-time updates
                    noise = np.random.normal(0, 0.01)  # 1% standard deviation
                    current_price = base_data[ticker].iloc[-1]
                    new_price = current_price * (1 + noise)
                    
                    self.latest_prices[ticker] = new_price
                    self.price_history[ticker].append({
                        'timestamp': datetime.now() + timedelta(minutes=i),
                        'price': new_price
                    })
            
            time.sleep(0.1)  # Simulate 100ms delay
            
            if i % 10 == 0:
                print(f"Real-time update {i+1}/{num_updates}: {self.latest_prices}")
        
        return self.latest_prices
    
    def close(self):
        """Close the real-time data connection"""
        if self.client:
            self.client.close()
        self.is_connected = False 