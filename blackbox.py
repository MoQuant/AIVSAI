# BLACKBOX AI


def TestA():
    def loan_amortization_table(principal, annual_interest_rate, years):

        # Convert annual values to monthly values

        monthly_interest_rate = annual_interest_rate / 1200

        months = years * 12


        # Calculate monthly payment

        monthly_payment = round(principal * monthly_interest_rate * (1 + monthly_interest_rate) ** months / ((1 + monthly_interest_rate) ** months - 1), 2)


        # Initialize table

        table = []


        # Calculate and print amortization table

        for month in range(months):

            interest = round(principal * monthly_interest_rate, 2)

            principal_payment = round(monthly_payment - interest, 2)

            principal -= principal_payment

            table.append({

                'Month': month + 1,

                'Payment': monthly_payment,

                'Interest': interest,

                'Principal': principal_payment,

                'Balance': round(principal, 2)

            })


        return table


    # Example usage:

    loan_amount = 200000

    interest_rate = 6

    years = 30


    table = loan_amortization_table(loan_amount, interest_rate, years)


    print("Loan Amortization Table:")

    print("----------------------------")

    print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Month', 'Payment', 'Interest', 'Principal', 'Balance'))

    print("----------------------------")

    for row in table:

        print("{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(row['Month'], row['Payment'], row['Interest'], row['Principal'], row['Balance']))

def TestB():
    import numpy as np

    def geometric_brownian_motion(stock_price, risk_free_rate, volatility, days):

        # Calculate time in years

        time_in_years = days / 365


        # Calculate drift and diffusion coefficients

        drift = risk_free_rate - 0.5 * volatility ** 2

        diffusion = volatility * np.sqrt(time_in_years)


        # Generate random numbers for simulation

        np.random.seed(0)

        random_numbers = np.random.normal(0, 1, days)


        # Initialize array to store stock prices

        stock_prices = np.zeros(days + 1)

        stock_prices[0] = stock_price


        # Simulate stock prices using Geometric Brownian motion

        for i in range(1, days + 1):

            stock_prices[i] = stock_prices[i - 1] * np.exp((drift + diffusion * random_numbers[i - 1]) / np.sqrt(365))


        return stock_prices


    # Example usage:

    stock_price = 100

    risk_free_rate = 0.02

    volatility = 0.3

    days = 365


    stock_prices = geometric_brownian_motion(stock_price, risk_free_rate, volatility, days)


    print("Forecasted Stock Prices:")

    print("----------------------------")

    print(stock_prices)


def TestC():
    import numpy as np

    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D


    # Define the function

    def f(x, y):

        return x**2 + y**2


    # Create a 3D plot of the function

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')


    x = np.linspace(-2, 2, 100)

    y = np.linspace(-2, 2, 100)

    x, y = np.meshgrid(x, y)

    z = f(x, y)


    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')


    # Calculate the partial derivatives at the point (1, 1)

    fx = lambda x, y: 2*x

    fy = lambda x, y: 2*y


    fx_1_1 = fx(1, 1)

    fy_1_1 = fy(1, 1)


    # Calculate the normal vector to the tangent plane

    normal_vector = np.array([fx_1_1, fy_1_1, -1])


    # Calculate the point on the tangent plane

    point_on_plane = np.array([1, 1, f(1, 1)])


    # Create a tangent plane at the point (1, 1)

    x_plane = np.linspace(-2, 2, 100)

    y_plane = np.linspace(-2, 2, 100)

    x_plane, y_plane = np.meshgrid(x_plane, y_plane)

    z_plane = (normal_vector[0] * (x_plane - point_on_plane[0]) +

            normal_vector[1] * (y_plane - point_on_plane[1]) +

            point_on_plane[2])


    ax.plot_surface(x_plane, y_plane, z_plane, color='r', alpha=0.5)


    ax.set_xlabel('X')

    ax.set_ylabel('Y')

    ax.set_zlabel('Z')


    plt.show()

def TestD():
    import pandas as pd
    import numpy as np
    import requests
    import json

    # Set API endpoint and parameters url = 'https://api.pro.coinbase.com/products/BTC-USD/candles'
    params = {
        'granularity': 86400,  # 1 day
        'start': '2020-01-01T00:00:00.000Z',
        'end': '2022-09-29T00:00:00.000Z'
    }

    url = 'https://api.pro.coinbase.com/products/BTC-USD/candles'

    # Send request and get response
    response = requests.get(url, params=params)

    # Check if response was successful
    if response.status_code == 200:
        # Parse JSON response
        data = json.loads(response.text)

        # Convert data to Pandas DataFrame
        df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

        # Convert time column to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Calculate moving averages
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['ma_200'] = df['close'].rolling(window=200).mean()

        # Implement trading strategy
        df['signal'] = np.where(df['close'] < df['ma_50'], 1, 0)
        df['signal'] = np.where(df['close'] > df['ma_200'], -1, df['signal'])

        # Print trading signals
        print(df[['time', 'close', 'ma_50', 'ma_200', 'signal']])

    else:
        print('Failed to retrieve data')

def TestE():
    import websocket
    import json
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Set WebSocket API endpoint
    ws_url = 'wss://ws-feed.pro.coinbase.com'

    # Set subscription message
    subscription_message = {
        'type': 'subscribe',
        'product_ids': ['BTC-USD'],
        'channels': ['level2']
    }

    # Establish WebSocket connection
    ws = websocket.create_connection(ws_url)

    # Send subscription message
    ws.send(json.dumps(subscription_message))

    # Initialize order book data
    bids = []
    asks = []

    # Initialize plot
    fig, ax = plt.subplots()
    ax.set_title('Coinbase Level 2 Order Book')
    ax.set_xlabel('Price')
    ax.set_ylabel('Quantity')

    # Function to update plot
    def update(frame):
        global bids, asks
        ax.clear()
        ax.set_title('Coinbase Level 2 Order Book')
        ax.set_xlabel('Price')
        ax.set_ylabel('Quantity')
        ax.barh([bid[0] for bid in bids[:30]], [bid[1] for bid in bids[:30]], color='g')
        ax.barh([ask[0] for ask in asks[:30]], [ask[1] for ask in asks[:30]], color='r')
        ax.set_xlim(0, 1000)

    # Function to handle WebSocket messages
    def on_message(ws, message):
        global bids, asks
        data = json.loads(message)
        if data['type'] == 'l2update':
            for bid in data['bids']:
                bids.append([float(bid[0]), float(bid[1])])
            for ask in data['asks']:
                asks.append([float(ask[0]), float(ask[1])])
            bids.sort(reverse=True)
            asks.sort()
            bids = bids[:30]
            asks = asks[:30]

    # Set up WebSocket event handlers
    ws.on_message = on_message

    # Start animation
    ani = animation.FuncAnimation(fig, update, interval=1000)

    # Show plot
    plt.show()

    # Keep WebSocket connection open
    while True:
        ws.ping()

def TestF():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Define parameters
    S = 100  # stock price
    K = np.linspace(70, 130, 100)  # strike price
    T = 1/12  # expiration time (1 month)
    r = 0.05  # risk-free rate
    sigma = 0.3  # volatility

    # Calculate d1 and d2
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # Calculate option prices and Greeks
    C = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    P = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    delta_C = norm.cdf(d1)
    delta_P = norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    theta_C = -S*norm.pdf(d1)*sigma / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    theta_P = -S*norm.pdf(d1)*sigma / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    rho_C = K*T*np.exp(-r*T)*norm.cdf(d2)
    rho_P = -K*T*np.exp(-r*T)*norm.cdf(-d2)

    # Plot Greeks
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    axs[0, 0].plot(K, delta_C, label='Call Delta')
    axs[0, 0].plot(K, delta_P, label='Put Delta')
    axs[0, 0].set_title('Delta')
    axs[0, 0].set_xlabel('Strike Price')
    axs[0, 0].set_ylabel('Delta')
    axs[0, 0].legend()

    axs[0, 1].plot(K, gamma)
    axs[0, 1].set_title('Gamma')
    axs[0, 1].set_xlabel('Strike Price')
    axs[0, 1].set_ylabel('Gamma')

    axs[1, 0].plot(K, vega)
    axs[1, 0].set_title('Vega')
    axs[1, 0].set_xlabel('Strike Price')
    axs[1, 0].set_ylabel('Vega')

    axs[1, 1].plot(K, theta_C, label='Call Theta')
    axs[1, 1].plot(K, theta_P, label='Put Theta')
    axs[1, 1].set_title('Theta')
    axs[1, 1].set_xlabel('Strike Price')
    axs[1, 1].set_ylabel('Theta')
    axs[1, 1].legend()

    axs[2, 0].plot(K, rho_C, label='Call Rho')
    axs[2, 0].plot(K, rho_P, label='Put Rho')
    axs[2, 0].set_title('Rho')
    axs[2, 0].set_xlabel('Strike Price')
    axs[2, 0].set_ylabel('Rho')
    axs[2, 0].legend()

    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

TestF()