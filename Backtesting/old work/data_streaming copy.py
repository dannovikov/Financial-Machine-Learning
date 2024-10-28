import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import matplotlib.animation as animation


# Define the App class that interacts with the strategy
class App:
    def __init__(self, tick_size=0.25, trade_cost=0.0):
        self.tick_size = tick_size
        self.trade_cost = trade_cost
        self.last_price = None
        self.qty = 0  # Position quantity: 0 (flat), 1 (long), -1 (short)
        self.open_price = None
        self.total_profit = 0.0
        self.current_profit = 0.0

    def tickPrice(self, price):
        self.last_price = price

    def calculate_current_profit(self):
        if self.qty != 0:
            self.current_profit = (self.last_price - self.open_price) * self.qty - self.trade_cost
        else:
            self.current_profit = 0.0

    def open_long_position(self):
        self.qty = 1
        self.open_price = self.last_price + self.tick_size  # Simulate slippage
        self.calculate_current_profit()

    def close_long_position(self):
        profit = (self.last_price - self.open_price) * self.qty - self.trade_cost
        self.total_profit += profit
        self.qty = 0
        self.open_price = None
        self.current_profit = 0.0

    def open_short_position(self):
        self.qty = -1
        self.open_price = self.last_price - self.tick_size  # Simulate slippage
        self.calculate_current_profit()

    def close_short_position(self):
        profit = (self.open_price - self.last_price) * -self.qty - self.trade_cost
        self.total_profit += profit
        self.qty = 0
        self.open_price = None
        self.current_profit = 0.0


# Define the Mean Reversion Strategy
class MeanReversionStrategy:
    def __init__(self, app, boll_window=600, width=2):
        self.app = app
        self.boll_window = boll_window
        self.width = width
        self.prices = deque(maxlen=boll_window)

    def tick(self, price):
        self.app.tickPrice(price)
        self.prices.append(self.app.last_price)
        self.app.calculate_current_profit()

        if len(self.prices) == self.boll_window:
            mu = np.mean(self.prices)
            std = np.std(self.prices)
            upper_band = mu + self.width * std
            lower_band = mu - self.width * std

            if self.app.qty == 0:
                if self.app.last_price < lower_band - self.app.tick_size:
                    self.app.open_long_position()
                elif self.app.last_price > upper_band + self.app.tick_size:
                    self.app.open_short_position()
            elif self.app.qty == 1:
                if self.app.last_price >= mu:
                    self.app.close_long_position()
            elif self.app.qty == -1:
                if self.app.last_price <= mu:
                    self.app.close_short_position()


# Define the Backtester class
class Backtester:
    def __init__(self, data, strategy, sampling_interval=0.25, speed=1.0):
        self.data = data
        self.strategy = strategy
        self.sampling_interval = sampling_interval  # Interval in seconds
        self.speed = speed  # Playback speed
        self.price_history = []
        self.upper_bands = []
        self.lower_bands = []
        self.sma = []
        self.timestamps = []

    def run(self, plotting=True):
        if plotting:
            fig, ax = plt.subplots()
            plt.style.use("seaborn-v0_8-darkgrid")
            plt.tight_layout()
            (line_price,) = ax.plot([], [], label="Price", color="blue")
            (line_upper,) = ax.plot([], [], label="Upper Band", color="purple")
            (line_lower,) = ax.plot([], [], label="Lower Band", color="purple")
            (line_sma,) = ax.plot([], [], label="SMA", color="cornflowerblue")
            text_pnl = ax.text(0.02, 0.95, "", transform=ax.transAxes)
            text_std = ax.text(0.02, 0.90, "", transform=ax.transAxes)
            text_current_profit = ax.text(0.02, 0.85, "", transform=ax.transAxes)
            line_open_price = ax.axhline(y=0, color="green", linestyle="--", label="Open Price")

            def init():
                ax.set_xlim(0, self.strategy.boll_window)
                ax.set_ylim(min(self.data["price"]), max(self.data["price"]))
                return (
                    line_price,
                    line_upper,
                    line_lower,
                    line_sma,
                    text_pnl,
                    text_std,
                    text_current_profit,
                    line_open_price,
                )

            def update(frame):
                price = frame.price
                timestamp = frame.datetime
                self.strategy.tick(price)
                self.price_history.append(price)
                self.timestamps.append(timestamp)

                if len(self.strategy.prices) == self.strategy.boll_window:
                    mu = np.mean(self.strategy.prices)
                    std = np.std(self.strategy.prices)
                    upper_band = mu + self.strategy.width * std
                    lower_band = mu - self.strategy.width * std
                    self.upper_bands.append(upper_band)
                    self.lower_bands.append(lower_band)
                    self.sma.append(mu)
                else:
                    self.upper_bands.append(None)
                    self.lower_bands.append(None)
                    self.sma.append(None)

                x_axis = range(len(self.price_history))

                line_price.set_data(x_axis, self.price_history)
                line_upper.set_data(x_axis, self.upper_bands)
                line_lower.set_data(x_axis, self.lower_bands)
                line_sma.set_data(x_axis, self.sma)
                ax.relim()
                ax.autoscale_view()

                text_pnl.set_text(f"Total PnL: {self.strategy.app.total_profit:.2f}")
                if len(self.strategy.prices) == self.strategy.boll_window:
                    text_std.set_text(f"Std: {std:.2f}")
                else:
                    text_std.set_text("Std: Calculating...")

                if self.strategy.app.qty != 0:
                    text_current_profit.set_text(f"Current Profit: {self.strategy.app.current_profit:.2f}")
                    line_open_price.set_ydata(self.strategy.app.open_price)
                    line_open_price.set_visible(True)
                else:
                    text_current_profit.set_text("Current Profit: 0.00")
                    line_open_price.set_visible(False)

                ax.legend(loc="upper left")

                return (
                    line_price,
                    line_upper,
                    line_lower,
                    line_sma,
                    text_pnl,
                    text_std,
                    text_current_profit,
                    line_open_price,
                )

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=self.data.itertuples(index=False),
                init_func=init,
                interval=(self.sampling_interval / self.speed) * 1000,
                blit=False,
                repeat=False,
            )
            plt.show()
        else:
            # Run without plotting (e.g., for performance testing)
            for index, row in self.data.iterrows():
                price = row["price"]
                self.strategy.tick(price)


# Load and preprocess data
# Replace 'your_data.csv' with the path to your CSV file
# The CSV should have columns: time, price, volume, symbol
csv = r"/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Backtesting/ES-Trades-During-Market-Hours.csv"
data = pd.read_csv(csv, names=["time", "price", "volume", "symbol"], header=0)
print(data.head())

# Convert 'time' to datetime
data["datetime"] = pd.to_datetime(data["time"], unit="s")

# Set datetime as index
data.set_index("datetime", inplace=True)

# Resample data at 0.25-second intervals using the last price available
resampled_data = data["price"].resample("250L").last().fillna(method="ffill").dropna().reset_index()

# Initialize App and Strategy
app = App()
strategy = MeanReversionStrategy(app)

# Initialize Backtester with desired speed (e.g., speed=10 for 10x faster playback)
backtester = Backtester(resampled_data, strategy, sampling_interval=0.25, speed=1.0)

# Run Backtester
backtester.run()
