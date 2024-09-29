# CHAT-GPT

def TestA():
    import pandas as pd

    def loan_amortization(loan_amount, annual_interest_rate, total_payments):
        # Monthly interest rate
        monthly_interest_rate = annual_interest_rate / 12 / 100
        
        # Calculate monthly payment using the formula for an annuity
        monthly_payment = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** total_payments) / ((1 + monthly_interest_rate) ** total_payments - 1)
        
        # Initialize variables for tracking the remaining balance and amortization table
        remaining_balance = loan_amount
        amortization_table = []
        
        for payment_number in range(1, total_payments + 1):
            interest_payment = remaining_balance * monthly_interest_rate
            principal_payment = monthly_payment - interest_payment
            remaining_balance -= principal_payment
            
            # Store the details of this payment
            amortization_table.append({
                'Payment Number': payment_number,
                'Payment': round(monthly_payment, 2),
                'Principal': round(principal_payment, 2),
                'Interest': round(interest_payment, 2),
                'Remaining Balance': round(remaining_balance, 2)
            })
        
        # Convert to DataFrame for easy reading and display
        df = pd.DataFrame(amortization_table)
        return df

    # Example usage
    loan_amount = 250000  # Loan amount in dollars
    annual_interest_rate = 5  # Annual interest rate in percent
    total_payments = 360  # Total number of payments (e.g., 360 for 30 years of monthly payments)

    amortization_df = loan_amortization(loan_amount, annual_interest_rate, total_payments)
    print(amortization_df)

def TestB():
    import numpy as np
    import matplotlib.pyplot as plt

    def geometric_brownian_motion(stock_price, risk_free_rate, volatility, time_horizon, time_steps):
        # Calculate the time increment (dt) based on the total time and number of time steps
        dt = time_horizon / time_steps
        
        # Generate an array to store the simulated stock prices
        prices = np.zeros(time_steps)
        prices[0] = stock_price
        
        # Simulate the stock price for each time step
        for t in range(1, time_steps):
            # Generate a random standard normal variable (for Brownian motion)
            z = np.random.normal()
            
            # Apply the GBM formula to calculate the stock price at the next time step
            prices[t] = prices[t-1] * np.exp((risk_free_rate - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * z)
        
        return prices

    # Example usage:
    stock_price = 100  # Initial stock price
    risk_free_rate = 0.05  # Risk-free rate (5%)
    volatility = 0.2  # Volatility (20%)
    days = 365  # Time horizon in days
    time_horizon = days / 365  # Convert time to years
    time_steps = days  # Number of time steps (1 per day)

    # Simulate the stock price path
    stock_prices = geometric_brownian_motion(stock_price, risk_free_rate, volatility, time_horizon, time_steps)

    # Plot the forecasted stock price path
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices)
    plt.title('Stock Price Forecast using Geometric Brownian Motion')
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    plt.grid(True)
    plt.show()

def TestC():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Define the surface function Z = X^2 + Y^2
    def surface(X, Y):
        return X**2 + Y**2

    # Define the tangent plane at (1, 1)
    def tangent_plane(X, Y):
        return 2*X + 2*Y - 2

    # Create a grid of X and Y values
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values for the surface and tangent plane
    Z_surface = surface(X, Y)
    Z_tangent = tangent_plane(X, Y)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z_surface, cmap='viridis', alpha=0.7)

    # Plot the tangent plane
    ax.plot_surface(X, Y, Z_tangent, cmap='coolwarm', alpha=0.5)

    # Highlight the point (1, 1, 2) where the tangent plane touches the surface
    ax.scatter(1, 1, 2, color='r', s=100, label='Tangent Point (1,1,2)')

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Surface Z = X^2 + Y^2 and Tangent Plane at (1, 1)')

    # Show the plot
    plt.legend()
    plt.show()


def TestD():
    import requests
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Fetch historical data from Coinbase API
    def fetch_bitcoin_data(granularity=86400):
        url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles'
        params = {
            'granularity': granularity  # 86400 seconds = 1 day
        }
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert to datetime
            df.set_index('time', inplace=True)
            return df.sort_index()
        else:
            raise Exception(f"Failed to fetch data: {response.status_code} {response.text}")

    # Implement the trading strategy: Buy if price < average, Sell if price > average
    def trading_strategy(df):
        df['average'] = df['close'].rolling(window=20).mean()  # 20-day rolling average
        df['position'] = np.where(df['close'] < df['average'], 1, -1)  # 1 = buy, -1 = sell
        
        # Log trading decisions
        df['action'] = df['position'].diff()  # Difference to find position changes (buy/sell)
        
        buy_signals = df[df['action'] == 2]  # When we switch from sell (-1) to buy (1)
        sell_signals = df[df['action'] == -2]  # When we switch from buy (1) to sell (-1)
        
        return buy_signals, sell_signals

    # Plot the results
    def plot_trading_strategy(df, buy_signals, sell_signals):
        plt.figure(figsize=(12, 6))
        
        # Plot the closing price
        plt.plot(df.index, df['close'], label='Bitcoin Price', color='blue')
        
        # Plot the moving average
        plt.plot(df.index, df['average'], label='20-day Average', color='orange', linestyle='--')
        
        # Plot buy signals
        plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
        
        plt.title('Bitcoin Trading Strategy (Buy below Average, Sell above Average)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Main function to run the script
    def main():
        # Fetch historical Bitcoin data
        df = fetch_bitcoin_data()
        
        # Implement trading strategy
        buy_signals, sell_signals = trading_strategy(df)
        
        # Plot the strategy results
        plot_trading_strategy(df, buy_signals, sell_signals)

    
    main()

def TestE():
    import json
    import websocket
    import matplotlib.pyplot as plt
    import numpy as np

    # Global variables for storing order book data
    bids = []
    asks = []

    # WebSocket callback functions
    def on_message(ws, message):
        global bids, asks
        data = json.loads(message)

        if data['type'] == 'snapshot':
            # Initialize the order book with the snapshot data
            bids = sorted([[float(price), float(size)] for price, size in data['bids']], key=lambda x: -x[0])
            asks = sorted([[float(price), float(size)] for price, size in data['asks']], key=lambda x: x[0])

        elif data['type'] == 'l2update':
            for change in data['changes']:
                side, price, size = change
                price = float(price)
                size = float(size)

                if side == 'buy':
                    # Update bids
                    bids = update_order_book(bids, price, size, descending=True)
                elif side == 'sell':
                    # Update asks
                    asks = update_order_book(asks, price, size, descending=False)

    def update_order_book(book, price, size, descending):
        # If size is 0, remove the order, otherwise update or insert
        if size == 0:
            book = [order for order in book if order[0] != price]
        else:
            updated = False
            for order in book:
                if order[0] == price:
                    order[1] = size
                    updated = True
                    break
            if not updated:
                book.append([price, size])
            book.sort(key=lambda x: -x[0] if descending else x[0])
        return book

    def on_open(ws):
        print("WebSocket connection opened.")
        subscribe_message = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["level2"]
        }
        ws.send(json.dumps(subscribe_message))

    def on_error(ws, error):
        print(f"Error: {error}")

    def on_close(ws):
        print("WebSocket connection closed.")

    # Plotting function for the top 30 depth chart
    def plot_order_book():
        global bids, asks

        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))

        while True:
            if len(bids) > 0 and len(asks) > 0:
                # Get top 30 bids and asks
                top_bids = bids[:30]
                top_asks = asks[:30]

                # Prepare data for plotting
                bid_prices, bid_sizes = zip(*top_bids)
                ask_prices, ask_sizes = zip(*top_asks)

                # Clear the current plot
                ax.clear()

                # Plot bids (green) and asks (red)
                ax.barh(bid_prices, bid_sizes, color='green', label='Bids')
                ax.barh(ask_prices, ask_sizes, color='red', label='Asks')

                # Set plot labels and title
                ax.set_xlabel('Size')
                ax.set_ylabel('Price')
                ax.set_title('BTC-USD Order Book Depth (Top 30)')

                # Display the legend
                ax.legend()

                # Pause for a second to create an animated effect
                plt.pause(1)

    # Main function to establish WebSocket connection and plot
    def main():
        ws = websocket.WebSocketApp(
            "wss://ws-feed.exchange.coinbase.com",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Run WebSocket in a separate thread so the plot can run concurrently
        import threading
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Start plotting the order book
        plot_order_book()

    main()

def TestF():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Black-Scholes function to calculate d1 and d2
    def d1(S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def d2(S, K, T, r, sigma):
        return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    # Greeks based on Black-Scholes model
    def call_delta(S, K, T, r, sigma):
        return norm.cdf(d1(S, K, T, r, sigma))

    def put_delta(S, K, T, r, sigma):
        return -norm.cdf(-d1(S, K, T, r, sigma))

    def gamma(S, K, T, r, sigma):
        return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))

    def vega(S, K, T, r, sigma):
        return S * norm.pdf(d1(S, K, T, r, sigma)) * np.sqrt(T)

    def call_theta(S, K, T, r, sigma):
        d1_val = d1(S, K, T, r, sigma)
        d2_val = d2(S, K, T, r, sigma)
        term1 = - (S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2_val)
        return term1 - term2

    def put_theta(S, K, T, r, sigma):
        d1_val = d1(S, K, T, r, sigma)
        d2_val = d2(S, K, T, r, sigma)
        term1 = - (S * norm.pdf(d1_val) * sigma) / (2 * np.sqrt(T))
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        return term1 + term2

    def call_rho(S, K, T, r, sigma):
        return K * T * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))

    def put_rho(S, K, T, r, sigma):
        return -K * T * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))

    # Parameters for the Black-Scholes model
    S = 100  # Stock price
    T = 1/12  # 1 month until expiration (1/12 years)
    r = 0.05  # Risk-free rate 5%
    sigma = 0.30  # Volatility 30%

    # Range of strike prices
    K = np.arange(70, 131, 1)

    # Calculate the Greeks for each strike price
    call_deltas = [call_delta(S, k, T, r, sigma) for k in K]
    put_deltas = [put_delta(S, k, T, r, sigma) for k in K]
    gammas = [gamma(S, k, T, r, sigma) for k in K]
    vegas = [vega(S, k, T, r, sigma) for k in K]
    call_thetas = [call_theta(S, k, T, r, sigma) for k in K]
    put_thetas = [put_theta(S, k, T, r, sigma) for k in K]
    call_rhos = [call_rho(S, k, T, r, sigma) for k in K]
    put_rhos = [put_rho(S, k, T, r, sigma) for k in K]

    # Plotting the Greeks
    plt.figure(figsize=(10, 8))

    # Delta
    plt.subplot(2, 3, 1)
    plt.plot(K, call_deltas, label="Call Delta", color="blue")
    plt.plot(K, put_deltas, label="Put Delta", color="orange")
    plt.title("Delta")
    plt.xlabel("Strike Price")
    plt.ylabel("Delta")
    plt.legend()

    # Gamma
    plt.subplot(2, 3, 2)
    plt.plot(K, gammas, label="Gamma", color="green")
    plt.title("Gamma")
    plt.xlabel("Strike Price")
    plt.ylabel("Gamma")
    plt.legend()

    # Vega
    plt.subplot(2, 3, 3)
    plt.plot(K, vegas, label="Vega", color="purple")
    plt.title("Vega")
    plt.xlabel("Strike Price")
    plt.ylabel("Vega")
    plt.legend()

    # Theta
    plt.subplot(2, 3, 4)
    plt.plot(K, call_thetas, label="Call Theta", color="red")
    plt.plot(K, put_thetas, label="Put Theta", color="cyan")
    plt.title("Theta")
    plt.xlabel("Strike Price")
    plt.ylabel("Theta")
    plt.legend()

    # Rho
    plt.subplot(2, 3, 5)
    plt.plot(K, call_rhos, label="Call Rho", color="brown")
    plt.plot(K, put_rhos, label="Put Rho", color="pink")
    plt.title("Rho")
    plt.xlabel("Strike Price")
    plt.ylabel("Rho")
    plt.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

TestF()