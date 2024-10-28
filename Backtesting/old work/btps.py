import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pytz
from collections import deque

# Read the CSV file
csv = "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Backtesting/ES-Trades-During-Market-Hours.csv"
df = pd.read_csv(csv)
df["date"] = pd.to_datetime(df["time"], unit="s").dt.date
assert df.time.is_monotonic_increasing


class Backtest:
    def __init__(self, df, time_increment):
        self.df = df
        self.time_increment = time_increment
        time_precision = len(str(time_increment).split(".")[1])
        self.next_time = round(df.time[0], time_precision)
        self.current_time = df.time[0]
        self.current_hour = datetime.fromtimestamp(self.current_time).astimezone(pytz.timezone("US/Eastern")).hour
        self.current_date = df.date[0]
        self.index = 0

        self.last_price = df.price[0]
        self.results = {}
        self.daily_trades = []

        self.qty = 0
        self.leverage = 50
        self.commission = 2.25
        self.tick_size = 0.25
        self.open_price = 0
        self.open_time = 0

        self.current_profit = 0
        self.current_min = float("inf")
        self.current_max = float("-inf")
        self.total_profit = 0
        self.max_drawdown = 0
        self.trade_id = 0

        # Visualization setup
        self.price_history = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Backtest Visualization")
        self.ax.set_xlabel("Ticks")
        self.ax.set_ylabel("Price")

    def tickPrice(self):
        if self.index >= len(self.df):
            return None

        # Handle new day trades
        if self.df.date[self.index] != self.current_date:
            self.record_daily_trades()
            self.current_date = self.df.date[self.index]
            self.next_time = round(self.df.time[self.index], len(str(self.time_increment).split(".")[1]))
            self.close_all_positions()

        # Move forward in the data to the next time increment
        while self.index < len(self.df) and self.df.time[self.index] <= self.next_time:
            self.last_price = self.df.price[self.index]
            self.price_history.append(self.last_price)  # Store price for visualization
            self.index += 1

        # Update time and hour
        self.current_time = self.next_time
        self.current_hour = datetime.fromtimestamp(self.current_time).astimezone(pytz.timezone("US/Eastern")).hour
        self.next_time += self.time_increment

        # Update plot every 1000 ticks for performance reasons
        if len(self.price_history) % 1000 == 0:
            self.plot_backtest()

        return self.last_price

    def plot_backtest(self):
        self.ax.clear()  # Clear the current axes
        x = range(len(self.price_history))
        self.ax.plot(x, self.price_history, label="Price", color="blue")

        # # Example for adding trade markers
        # for trade in self.daily_trades:
        #     color = "green" if trade["side"] == "buy" else "red"
        #     self.ax.plot(trade["time"], trade["fillPrice"], marker="o", color=color, label=trade["side"].capitalize())

        # # Group "buy" and "sell" trades
        # buy_trades = [trade for trade in self.daily_trades if trade["side"] == "buy"]
        # sell_trades = [trade for trade in self.daily_trades if trade["side"] == "sell"]

        # # Plot all "buy" trades with green markers and label them as "Buy" (only once)
        # if buy_trades:
        #     self.ax.plot([trade["time"] for trade in buy_trades], [trade["fillPrice"] for trade in buy_trades], "go", label="Buy")

        # # Plot all "sell" trades with red markers and label them as "Sell" (only once)
        # if sell_trades:
        #     self.ax.plot([trade["time"] for trade in sell_trades], [trade["fillPrice"] for trade in sell_trades], "ro", label="Sell")

        self.ax.set_title(f"Backtest Results: Time {self.current_time}")
        self.ax.set_xlabel("Ticks")
        self.ax.set_ylabel("Price")
        self.ax.legend()
        plt.draw()
        # plt.pause(0.25)  # Slight pause for real-time updates
        import time

        time.sleep(0.25)

    def calculate_current_profit(self):
        if self.qty == 0:
            self.current_profit = 0
            self.current_min = float("inf")
            self.current_max = float("-inf")
            return

        if self.qty == 1:
            self.current_profit = (self.last_price - self.open_price) * self.leverage - self.commission * 2
        elif self.qty == -1:
            self.current_profit = (self.open_price - self.last_price) * self.leverage - self.commission * 2

        if self.current_profit < self.current_min:
            self.current_min = self.current_profit
        elif self.current_profit > self.current_max:
            self.current_max = self.current_profit

    def open_long_position(self):
        self.qty = 1
        slip_price = self.last_price + self.tick_size
        self.open_price = slip_price
        self.open_time = self.current_time
        self.daily_trades.append({"id": self.trade_id, "action": "open", "side": "buy", "fillPrice": slip_price, "time": self.current_time})

    def open_short_position(self):
        self.qty = -1
        slip_price = self.last_price - self.tick_size
        self.open_price = slip_price
        self.open_time = self.current_time
        self.daily_trades.append({"id": self.trade_id, "action": "open", "side": "sell", "fillPrice": slip_price, "time": self.current_time})

    def close_long_position(self):
        self.qty = 0
        slip_price = self.last_price - self.tick_size
        self.daily_trades.append({"id": self.trade_id, "action": "close", "side": "sell", "fillPrice": slip_price, "time": self.current_time, "profit": self.current_profit, "profit_low": self.current_min, "profit_high": self.current_max})
        self.trade_id += 1

    def close_short_position(self):
        self.qty = 0
        slip_price = self.last_price + self.tick_size
        self.daily_trades.append({"id": self.trade_id, "action": "close", "side": "buy", "fillPrice": slip_price, "time": self.current_time, "profit": self.current_profit, "profit_low": self.current_min, "profit_high": self.current_max})
        self.trade_id += 1

    def close_all_positions(self):
        if self.qty == 1:
            self.close_long_position()
        elif self.qty == -1:
            self.close_short_position()

    def record_daily_trades(self):
        self.results[self.current_date] = self.daily_trades
        self.daily_trades = []


class MeanReversionStrategy:
    def __init__(self, app, boll_window=600, width=2, max_position_duration=100):
        self.app = app
        self.boll_window = boll_window
        self.width = width
        self.max_position_duration = max_position_duration
        self.prices = deque(maxlen=boll_window)

    def calculate_bands(self):
        mu = np.mean(self.prices)
        std = np.std(self.prices)
        upper_band = mu + self.width * std
        lower_band = mu - self.width * std
        return mu, upper_band, lower_band

    def tick(self):
        self.app.tickPrice()
        self.prices.append(self.app.last_price)
        self.app.calculate_current_profit()

        if len(self.prices) == self.boll_window:
            mu = np.mean(self.prices)
            std = np.std(self.prices)
            upper_band = mu + self.width * std
            lower_band = mu - self.width * std

            if self.app.qty == 0 and not self.app.current_hour >= 16:  # don't open a trade after 4pm
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

            if self.app.current_time - self.app.open_time > self.max_position_duration:
                self.app.close_all_positions()


def run_backtest(app, strategy):
    total_ticks = len(app.df)
    with tqdm(total=total_ticks) as pbar:
        while app.tickPrice() is not None:
            strategy.tick()
            pbar.update(1)


# Running the backtest
app = Backtest(df=df, time_increment=0.25)
strategy = MeanReversionStrategy(app)
run_backtest(app, strategy)
