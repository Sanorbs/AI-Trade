#!/usr/bin/env python3
"""
Test script for real-time data functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_stream.realtime_data_manager import RealTimeDataManager
import pandas as pd
import numpy as np

def test_real_time_data():
    """Test real-time data functionality"""
    print("Testing Real-Time Data Functionality")
    print("=" * 40)
    
    # Test tickers
    tickers = ["AAPL", "TSLA", "BA"]
    
    # Create a mock historical data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    historical_data = pd.DataFrame({
        'timestamp': dates,
        'AAPL': [150.0, 151.0, 152.0, 149.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0],
        'TSLA': [250.0, 252.0, 248.0, 253.0, 255.0, 257.0, 259.0, 261.0, 263.0, 265.0],
        'BA': [200.0, 201.0, 199.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0]
    })
    
    print("Historical data sample:")
    print(historical_data.head())
    print()
    
    # Test real-time data manager (simulation mode)
    print("Testing Real-Time Data Manager (Simulation Mode)")
    realtime_manager = RealTimeDataManager("test_key", tickers)
    
    # Simulate real-time updates
    print("Simulating real-time price updates...")
    latest_prices = realtime_manager.simulate_real_time_data(historical_data, num_updates=20)
    
    print(f"\nFinal simulated prices: {latest_prices}")
    
    # Get price history
    price_df = realtime_manager.get_price_dataframe(lookback_hours=1)
    if price_df is not None:
        print("\nPrice history (last hour):")
        print(price_df.tail())
    
    print("\nReal-time data test completed!")

if __name__ == "__main__":
    test_real_time_data() 