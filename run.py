import pickle
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import yaml

import os
import numpy as np
import pandas as pd
from utils.utils import get_data, make_dir
from environments.multi_stock_env import MultiStockEnv, play_one_episode, get_scaler
from models.dqn import DQNAgent
from models.ddpg import DDPGAgent
from models.lstm_forecast import lstm_forecast
from models.sentiment_analysis import sentiment_analysis

def main():
    # Config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Use config for model selection and data pipeline
    stock_tickers = config.get('stock_tickers', ["AAPL", "BA", "TSLA"])
    forecast_enabled = config.get('forecast', False)
    sentiment_enabled = config.get('sentiment', False)
    agent_type = config.get('agent', 'DQN')
    n_stock = len(stock_tickers)
    n_forecast = 0
    n_sentiment = 0
    models_folder = 'saved_models'
    rewards_folder = 'saved_rewards'
    rl_folder = 'saved_models/rl'
    rl_rewards = 'saved_rewards/rl'
    lstm_folder = 'saved_models/lstm'
    news_folder = './data/news'
    forecast_window = 10
    num_episodes = 300
    batch_size = 16
    initial_investment = 10000

    # Parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--forecast', type=str, default=None, help='Enable stock forecasting. Select "one" or "multi"')
    parser.add_argument('-s', '--sentiment', type=bool, default=False, help='Enable sentiment analysis. Select "True" or "False"')
    parser.add_argument('-a', '--agent', type=str, default="DQN", help='Select "DQN" or "DDPG"')
    args = parser.parse_args()

    make_dir(models_folder)
    make_dir(rewards_folder)
    make_dir(rl_folder)
    make_dir(rl_rewards)
    make_dir(lstm_folder)

    # Get data
    realtime_manager = None
    if config.get('realtime_data', False):
        print("Connecting to real-time data stream...")
        try:
            from data_stream.realtime_data_manager import RealTimeDataManager
            # Get API key from config
            api_key = config.get('finnhub_api_key', 'YOUR_API_KEY_HERE')
            
            if api_key == "YOUR_API_KEY_HERE":
                print("⚠️  WARNING: Please set your Finnhub API key in config.yaml")
                print("   Using simulated real-time data for demonstration...")
                realtime_manager = RealTimeDataManager("demo_key", stock_tickers)
                # Simulate real-time data
                historical_data = get_data("./data", stock_tickers)
                realtime_manager.simulate_real_time_data(historical_data, num_updates=50)
            else:
                realtime_manager = RealTimeDataManager(api_key, stock_tickers)
                
            if realtime_manager.connect() or api_key == "YOUR_API_KEY_HERE":
                print("Real-time data connection established!")
                # Get initial real-time prices
                latest_prices = realtime_manager.get_latest_prices()
                if latest_prices:
                    print(f"Latest real-time prices: {latest_prices}")
                else:
                    print("No real-time prices available, using historical data as base")
            else:
                print("Real-time connection failed, using historical data")
                realtime_manager = None
        except Exception as e:
            print(f"Real-time data setup failed: {e}")
            print("Falling back to historical data...")
            realtime_manager = None
    
    # Load historical data as base
    data = get_data("./data", stock_tickers)
    print()
    
    # Generate state (features) based on arguments (forecast & sentiment)
    if args.forecast == None and args.sentiment == False:
        data = data.drop(columns="timestamp").iloc[forecast_window:].reset_index(drop=True).values
    
    elif args.forecast != None and args.sentiment == False:
        concat_data = data.iloc[forecast_window:].reset_index(drop=True)
        for ticker in stock_tickers:
            print(f"Performing {ticker} {args.forecast.title()}step Forecast")
            predictions = {}
            predictions[f'{ticker}_Forecast'] = lstm_forecast(models_folder, ticker, data, forecast_window, args.forecast.lower())
    
            # predictions[f'{ticker}_Forecast'] = pd.DataFrame(predictions)
            # predictions[f'{ticker}_Forecast'].index = pd.RangeIndex(forecast_window, forecast_window + len(predictions[f'{ticker}_Forecast']))
            
            # Fix: Ensure predictions is properly formatted as a DataFrame
            print(f"DEBUG: predictions type: {type(predictions)}")
            print(f"DEBUG: predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
            print(f"DEBUG: predictions content: {predictions}")
            
            if isinstance(predictions, dict):
                # If predictions is a dict, extract the forecast values
                forecast_values = predictions[f'{ticker}_Forecast']
                # Ensure it's 1D array
                if hasattr(forecast_values, 'flatten'):
                    forecast_values = forecast_values.flatten()
                elif hasattr(forecast_values, 'values'):
                    forecast_values = forecast_values.values.flatten()
                forecast_df = pd.DataFrame({f'{ticker}_Forecast': forecast_values})
            else:
                # If predictions is already the forecast values
                # Ensure it's 1D array
                if hasattr(predictions, 'flatten'):
                    predictions = predictions.flatten()
                forecast_df = pd.DataFrame({f'{ticker}_Forecast': predictions})
            
            concat_data = pd.concat([concat_data, forecast_df], join="outer", axis=1)

        print(f"{args.forecast.title()}step Forecasts Added!\n")
        data = concat_data.drop(columns="timestamp").values
        n_forecast = len(stock_tickers)

    elif args.forecast != None and args.sentiment:
        concat_data = data.iloc[forecast_window:].reset_index(drop=True)
        for ticker in stock_tickers:
            print(f"Performing {ticker} {args.forecast.title()}step Forecast")
            predictions = {}
            predictions[f'{ticker}_Forecast'] = lstm_forecast(models_folder, ticker, data, forecast_window, args.forecast.lower())
    
            # predictions[f'{ticker}_Forecast'] = pd.DataFrame(predictions)
            # predictions[f'{ticker}_Forecast'].index = pd.RangeIndex(forecast_window, forecast_window + len(predictions[f'{ticker}_Forecast']))
            
            # Fix: Ensure predictions is properly formatted as a DataFrame
            print(f"DEBUG: predictions type: {type(predictions)}")
            print(f"DEBUG: predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'no shape'}")
            print(f"DEBUG: predictions content: {predictions}")
            
            if isinstance(predictions, dict):
                # If predictions is a dict, extract the forecast values
                forecast_values = predictions[f'{ticker}_Forecast']
                # Ensure it's 1D array
                if hasattr(forecast_values, 'flatten'):
                    forecast_values = forecast_values.flatten()
                elif hasattr(forecast_values, 'values'):
                    forecast_values = forecast_values.values.flatten()
                forecast_df = pd.DataFrame({f'{ticker}_Forecast': forecast_values})
            else:
                # If predictions is already the forecast values
                # Ensure it's 1D array
                if hasattr(predictions, 'flatten'):
                    predictions = predictions.flatten()
                forecast_df = pd.DataFrame({f'{ticker}_Forecast': predictions})
            
            concat_data = pd.concat([concat_data, forecast_df], join="outer", axis=1)

        print(f"{args.forecast.title()}step Forecasts Added!\n")

        for ticker in stock_tickers:
            print(f"Analyzing {ticker} Stock Sentiment")
            sentiment_df = sentiment_analysis(news_folder, ticker)
            
            concat_data = pd.merge(concat_data, sentiment_df, left_on="timestamp", right_on="publishedAt").drop(columns="publishedAt", axis=1)
        
        print("Sentiment Features Added!\n")
        print(concat_data)
        data = concat_data.drop(columns="timestamp").values
        n_forecast = len(stock_tickers)
        n_sentiment = len(stock_tickers)

    elif args.sentiment:
        concat_data = data
        for ticker in stock_tickers:
            print(f"Analyzing {ticker} Stock Sentiment")
            sentiment_df = sentiment_analysis(news_folder, ticker)
            # print(sentiment_df)
            concat_data = pd.merge(concat_data, sentiment_df, left_on="timestamp", right_on="publishedAt").drop(columns="publishedAt", axis=1)
        
        print("Sentiment Features Added!\n")
        data = concat_data.drop(columns="timestamp").values
        n_sentiment = len(stock_tickers)

    n_timesteps, _ = data.shape

    n_train = n_timesteps
    train_data = data[:n_train]
    test_data = data[n_train:]

    # Initialize the MultiStock Environment
    env = MultiStockEnv(train_data, n_stock, n_forecast, n_sentiment, initial_investment, "DQN")
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)

    # Store the final value of the portfolio (end of episode)
    portfolio_value = []

    # After agent/environment creation and before/after predictions:
    risk_manager = None
    if config.get('risk', False):
        from ethics.risk_manager import RiskManager
        risk_manager = RiskManager()

    user_profile = None
    feedback_loop = None
    if config.get('personalization', False):
        from personalization.user_profile import UserProfile
        from personalization.feedback_loop import FeedbackLoop
        user_profile = UserProfile(risk_tolerance=config.get('risk_tolerance', 'balanced'),
                                  target_return=config.get('target_return', 0.1))
        feedback_loop = FeedbackLoop(agent, user_profile)

    ######### DDPG #########
    # Run with DDPG Agent
    if args.agent.lower() == "ddpg":
        env = MultiStockEnv(train_data, n_stock, n_forecast, n_sentiment, initial_investment, "DDPG")
        DDPGAgent(env, num_episodes)
        exit()
    ######### /DDPG #########

    ######### DQN #########
    # Run with DQN
    # play the game num_episodes times
    print("\nRunning DQN Agent...\n")
    for e in range(num_episodes):
        val = play_one_episode(agent, env, scaler, batch_size)
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}")
        portfolio_value.append(val)
        if risk_manager:
            risk_manager.update(val)
            if risk_manager.get_violations():
                print(f"[RISK/ETHICS] Violations: {risk_manager.get_violations()}")
        if feedback_loop:
            trading_result = val - (portfolio_value[-2] if len(portfolio_value) > 1 else 0)
            feedback_loop.update_agent(trading_result)
            print(f"[PERSONALIZATION] Agent params: {feedback_loop.get_agent_params()}")

    # save the weights when we are done
    # save the DQN
    agent.save(f'{models_folder}/rl/dqn.h5')

    # save the scaler
    with open(f'{models_folder}/rl/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # save portfolio value for each episode
    np.save(f'{rewards_folder}/rl/dqn.npy', portfolio_value)

    print("\nDQN Agent run complete and saved!")

    a = np.load(f'./saved_rewards/rl/dqn.npy')

    print(f"\nCumulative Portfolio Value Average: {a.mean():.2f}, Min: {a.min():.2f}, Max: {a.max():.2f}")
    plt.plot(a)
    plt.title(f"Portfolio Value Per Episode ({args.agent.upper()})")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Episodes")
    plt.show()
    ######### /DQN #########

    # Add hooks for XAI, risk, and personalization after agent/environment creation
    # Placeholder: XAI
    if config.get('xai', False):
        print('XAI module enabled (placeholder)')
        # TODO: Integrate XAI tools here
    # Placeholder: Risk/Ethics
    if config.get('risk', False):
        print('Risk/Ethics module enabled (placeholder)')
        # TODO: Integrate risk/ethics checks here
    # Placeholder: Personalization
    if config.get('personalization', False):
        print('Personalization module enabled (placeholder)')
        # TODO: Integrate personalization/feedback loop here

if __name__ == '__main__':
    main()