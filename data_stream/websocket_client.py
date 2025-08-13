import websocket
import threading
import json
import time

class RealTimeDataClient:
    def __init__(self, url, tickers):
        self.url = url
        self.tickers = tickers
        self.ws = None
        self.data = {ticker: None for ticker in tickers}
        self.thread = None
        self.running = False

    def on_message(self, ws, message):
        msg = json.loads(message)
        # Assume message contains {'ticker': ..., 'price': ..., 'timestamp': ...}
        ticker = msg.get('ticker')
        if ticker in self.data:
            self.data[ticker] = msg

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket closed")
        self.running = False

    def on_open(self, ws):
        print("WebSocket connection opened")
        # Example: subscribe to tickers
        for ticker in self.tickers:
            sub_msg = json.dumps({"type": "subscribe", "symbol": ticker})
            ws.send(sub_msg)

    def connect(self):
        self.ws = websocket.WebSocketApp(
            self.url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        self.running = True
        self.thread = threading.Thread(target=self.ws.run_forever)
        self.thread.start()
        # Wait for connection
        time.sleep(2)

    def get_latest(self, ticker):
        return self.data.get(ticker)

    def close(self):
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join()

# Example usage (for testing):
if __name__ == "__main__":
    # Use a mock or public WebSocket endpoint for demonstration
    ws_url = "wss://ws.finnhub.io?token= d1lvfopr01qksvurca8gd1lvfopr01qksvurca90"
    tickers = ["AAPL", "TSLA", "BA"]
    client = RealTimeDataClient(ws_url, tickers)
    client.connect()
    try:
        for _ in range(10):
            for ticker in tickers:
                print(client.get_latest(ticker))
            time.sleep(1)
    finally:
        client.close() 
