import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        else:
            mu = std = upper_band = lower_band = None

        return mu, std, upper_band, lower_band  # Return these for plotting


# Define the Backtester class
class Backtester:
    def __init__(self, data, strategy, sampling_interval=0.25, speed=1.0):
        self.data = data
        self.strategy = strategy
        self.sampling_interval = sampling_interval  # Interval in seconds
        self.speed = speed  # Playback speed
        self.price_history = deque(maxlen=1200)
        self.timestamps = deque(maxlen=1200)

    def run(self, plotting=True):
        if plotting:
            fig, ax = plt.subplots()
            plt.style.use("seaborn-v0_8-darkgrid")
            plt.tight_layout()

            def update(frame):
                price = frame.price
                timestamp = frame.datetime
                mu, std, upper_band, lower_band = self.strategy.tick(price)
                self.price_history.append(price)
                self.timestamps.append(timestamp)

                ax.clear()
                x_axis = range(len(self.price_history))

                # Plot price
                ax.plot(x_axis, self.price_history, label=f"Price {self.strategy.app.last_price:.2f}", color="blue")

                # Plot Bollinger Bands and SMA if available
                if len(self.strategy.prices) == self.strategy.boll_window:
                    sma = [mu] * len(x_axis)
                    upper_bands = [upper_band] * len(x_axis)
                    lower_bands = [lower_band] * len(x_axis)
                    ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_band:.2f}", color="purple")
                    ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_band:.2f}", color="purple")
                    ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
                    ax.fill_between(x_axis, lower_bands, upper_bands, color="gray", alpha=0.2)
                else:
                    upper_band = lower_band = mu = std = self.strategy.app.last_price

                # Include textual information in the legend
                ax.plot([], [], " ", label=f"PnL {self.strategy.app.total_profit:.2f}")
                if std is not None:
                    ax.plot([], [], " ", label=f"std: {std:.2f}")
                else:
                    ax.plot([], [], " ", label="std: Calculating...")

                # Plot open position if any
                if self.strategy.app.qty != 0:
                    ax.plot([], [], " ", label=f"Current Profit {self.strategy.app.current_profit:.2f}")
                    ax.hlines(
                        self.strategy.app.open_price,
                        0,
                        len(self.price_history),
                        label=f"Open Price {self.strategy.app.open_price:.2f}",
                        color="green" if self.strategy.app.qty == 1 else "red",
                        linestyles="--",
                    )

                # Set dynamic y-limits
                ylim_lower = lower_band - 2
                ylim_upper = upper_band + 2
                ax.set_ylim(ylim_lower, ylim_upper)

                # Set x-limits to show the latest data
                window_size = self.strategy.boll_window
                ax.set_xlim(max(0, len(self.price_history) - window_size), len(self.price_history) + 10)

                # Configure the legend to prevent overlapping
                ax.legend(ncol=3, loc="upper left", fontsize="small")

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=self.data.itertuples(index=False),
                interval=(self.sampling_interval / self.speed) * 1000,
                blit=False,
                repeat=False,
            )
            plt.show()
        else:
            # Run without plotting
            for frame in self.data.itertuples(index=False):
                price = frame.price
                self.strategy.tick(price)


# Load and preprocess data
# Replace 'your_data.csv' with the path to your CSV file
# The CSV should have columns: time, price, volume, symbol
data = pd.read_csv(
    "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Backtesting/ES-Trades-During-Market-Hours.csv",
    names=["time", "price", "volume", "symbol"],
    header=0,
)

# Convert 'time' to datetime
data["datetime"] = pd.to_datetime(data["time"], unit="s")

# Set datetime as index
data.set_index("datetime", inplace=True)

# Resample data at 0.25-second intervals using the last price available
resampled_data = data["price"].resample("250L").last().fillna(method="ffill").dropna().reset_index()

# Initialize App and Strategy
app = App()
strategy = MeanReversionStrategy(app)

# Initialize Backtester with desired speed
backtester = Backtester(resampled_data, strategy, sampling_interval=0.25, speed=1.0)

# Run Backtester
backtester.run()
