import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import pytz
import copy
from collections import deque
from datetime import datetime
import pickle


# csv = "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Preprocessing/esm4_unix.csv"
csv = "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Preprocessing/esm4.csv"
df = pd.read_csv(csv)
dt = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("US/Eastern")
df["date"] = dt.dt.date
df["dayofweek"] = dt.dt.dayofweek
df["hour"] = dt.dt.hour
df["minute"] = dt.dt.minute


class Backtest:
    def __init__(self, df, time_increment):
        # time_increment refers to how often the current price is sampled
        self.index = 0  # index of the current price in the dataframe

        self.time_increment = time_increment
        self.time_increment_precision = len(str(time_increment).split(".")[1])

        self.last_time = df.time[0]
        self.next_time = round(df.time[0], self.time_increment_precision)

        self.current_date = df.date[0]
        self.current_hour = df.hour[0]
        self.current_minute = df.minute[0]

        self.leverage = 50
        self.commission = 2.25
        self.tick_size = 0.25
        self.trade_cost = self.commission * 2 / self.leverage

        self.last_price = df.price[0]
        self.qty = 0
        self.open_price = 0
        self.open_time = 0
        self.current_profit = 0

        self.current_min = float("inf")
        self.current_max = float("-inf")
        self.max_drawdown = 0
        self.total_profit = 0
        self.daily_total_profit = 0
        self.num_days = 1

        self.trade_id = 0
        self.results = {}
        self.daily_trades = []
        self.df = df

        # aggregating values
        # i want to store a list of daily returns, and report their mean and std every day
        self._daily_returns_list = []

    def tickPrice(self):

        if self.index + 1 >= len(self.df):
            return None

        if self.current_hour >= 16 and self.current_minute >= 30:
            print(f"date: {self.current_date}, time: {self.current_hour}:{self.current_minute}, advancing to next day")
            self._advance_to_next_trading_day()
            self.daily_total_profit = 0
            self.num_days += 1
            print(f"new date: {self.current_date}, time: {self.current_hour}:{self.current_minute}")

        # advance to the next price sampling time
        while self.df.time[self.index] <= self.next_time and self.index < len(self.df):
            self.index += 1

        self.last_price = self.df.price[self.index]

        self.last_time = self.next_time
        self.next_time += self.time_increment

        self.current_hour = self.df.hour[self.index]
        self.current_minute = self.df.minute[self.index]

        self.index += 1

        return self.last_price

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
        self.open_time = self.last_time
        self.daily_trades.append(
            {"id": self.trade_id, "action": "open", "side": "buy", "fillPrice": slip_price, "time": self.last_time}
        )

    def open_short_position(self):
        self.qty = -1
        slip_price = self.last_price - self.tick_size
        self.open_price = slip_price
        self.open_time = self.last_time
        self.daily_trades.append(
            {"id": self.trade_id, "action": "open", "side": "sell", "fillPrice": slip_price, "time": self.last_time}
        )

    def close_long_position(self):
        self.qty = 0
        slip_price = self.last_price - self.tick_size
        self.daily_trades.append(
            {
                "id": self.trade_id,
                "action": "close",
                "side": "sell",
                "fillPrice": slip_price,
                "time": self.last_time,
                "profit": self.current_profit,
                "profit_low": self.current_min,
                "profit_high": self.current_max,
            }
        )
        self.trade_id += 1
        self.total_profit += (slip_price - self.open_price) * self.leverage - self.commission * 2
        self.daily_total_profit += (slip_price - self.open_price) * self.leverage - self.commission * 2

    def close_short_position(self):
        self.qty = 0
        slip_price = self.last_price + self.tick_size
        self.daily_trades.append(
            {
                "id": self.trade_id,
                "action": "close",
                "side": "buy",
                "fillPrice": slip_price,
                "time": self.last_time,
                "profit": self.current_profit,
                "profit_low": self.current_min,
                "profit_high": self.current_max,
            }
        )
        self.trade_id += 1
        self.total_profit += (self.open_price - slip_price) * self.leverage - self.commission * 2
        self.daily_total_profit += (self.open_price - slip_price) * self.leverage - self.commission * 2

    def close_all_positions(self):
        if self.qty == 1:
            self.close_long_position()
        elif self.qty == -1:
            self.close_short_position()

    def record_daily_trades(self):
        if not self.daily_trades:
            return
        self.results[self.current_date] = copy.deepcopy(self.daily_trades)
        self._daily_returns_list.append(self.daily_total_profit)
        print(
            f"Date: {self.current_date}, Daily PnL: {self.daily_total_profit:.2f}, Num Trades: {len(self.daily_trades)}, total PnL: {self.total_profit:.2f}, num days: {self.num_days}, mean daily PnL: {self.total_profit / self.num_days:.2f}, std daily PnL: {np.std(self._daily_returns_list):.2f}"
        )
        self.daily_trades = []

    def _advance_to_next_trading_day(self):
        self.record_daily_trades()
        while self.index < len(self.df) and (self._is_weekend() or self._is_outside_trading_hours()):
            self.index += 1
        if self.index >= len(self.df):
            return None
        self.last_time = self.df.time[self.index]
        self.next_time = round(self.last_time, self.time_increment_precision)
        self.current_date = self.df.date[self.index]
        self.current_hour = self.df.hour[self.index]
        self.current_minute = self.df.minute[self.index]

    def _is_weekend(self):
        return self.df.dayofweek[self.index] >= 5

    def _is_before_trading_hours(self):
        return self.df.hour[self.index] < 9 or (self.df.hour[self.index] == 9 and self.df.minute[self.index] <= 30)

    def _is_after_trading_hours(self):
        return self.df.hour[self.index] > 16 or (self.df.hour[self.index] == 16 and self.df.minute[self.index] >= 30)

    def _is_outside_trading_hours(self):
        return self._is_before_trading_hours() or self._is_after_trading_hours()


class MeanReversionStrategy:
    def __init__(self, app, boll_window=600, history_window=4000, width=2, plotting=True, ax=None):
        self.app = app

        self.boll_window = boll_window
        self.width = width
        self.plotting = plotting
        self.ax = ax

        self.prices = deque(maxlen=boll_window)
        self.price_history = deque(maxlen=history_window)
        self.sma = deque(maxlen=history_window)
        self.upper_bands = deque(maxlen=history_window)
        self.lower_bands = deque(maxlen=history_window)

    # @profile
    def calculate_bands(self):
        mu = np.mean(self.prices)
        std = np.std(self.prices)
        upper_band = mu + self.width * std
        lower_band = mu - self.width * std
        self.sma.append(mu)
        self.upper_bands.append(upper_band)
        self.lower_bands.append(lower_band)
        return mu, upper_band, lower_band

    # @profile
    def tick(self):
        self.prices.append(self.app.last_price)
        self.price_history.append(self.app.last_price)

        self.app.calculate_current_profit()

        mu, upper_band, lower_band = self.calculate_bands()

        if len(self.prices) == self.boll_window:

            before_3_28_pm = self.app.current_hour < 15 or (
                self.app.current_hour == 15 and self.app.current_minute < 28
            )
            after_9_48_am = self.app.current_hour > 9 or (self.app.current_hour == 9 and self.app.current_minute > 48)

            if self.app.qty == 0 and after_9_48_am and before_3_28_pm:
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

        if self.plotting and len(self.prices) == self.boll_window and self.app.index % 15 == 0:
            x_axis = range(len(self.price_history))
            self.ax.clear()
            self.ax.plot(x_axis, self.price_history, label=f"Price {self.app.last_price:.2f}", color="blue")
            self.ax.plot(x_axis, self.upper_bands, label=f"Upper Band {self.upper_bands[-1]:.2f}", color="purple")
            self.ax.plot(x_axis, self.lower_bands, label=f"Lower Band {self.lower_bands[-1]:.2f}", color="purple")
            self.ax.plot(x_axis, self.sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
            self.ax.plot([], [], " ", label=f"Total PnL {self.app.total_profit:.2f}")
            self.ax.plot([], [], " ", label=f"PnL {self.app.daily_total_profit:.2f}")
            self.ax.plot([], [], " ", label=f"Dl. Avg. PnL {self.app.total_profit / self.app.num_days:.2f}")
            self.ax.plot([], [], " ", label=f"index {self.app.index}")
            self.ax.plot([], [], " ", label=f"Time {self.app.current_hour}:{self.app.current_minute}")
            if self.app.qty != 0:
                self.ax.plot([], [], " ", label=f"Current Profit {self.app.current_profit:.2f}")
                self.ax.hlines(
                    self.app.open_price,
                    0,
                    len(self.price_history),
                    label=f"Open Price {self.app.open_price:.2f}",
                    color="forestgreen" if self.app.qty == 1 else "firebrick",
                )

            if not before_3_28_pm:
                self.ax.plot([], [], " ", label="After 3:28pm, not trading")
            if not after_9_48_am:
                self.ax.plot([], [], " ", label="Before 9:48am, not trading")

            self.ax.fill_between(x_axis, self.lower_bands, self.upper_bands, color="gray", alpha=0.2)
            self.ax.set_ylim(min(self.price_history) - 5, max(self.price_history) + 5)
            self.ax.legend(ncol=5, loc="upper center")
            plt.pause(1e-8)


# Running the backtest
def run_backtest(app, strategy):
    with tqdm(total=5849263) as pbar:
        count = 0
        strategy.plotting = False
        while app.tickPrice() is not None and app.index < len(app.df):
            strategy.tick()
            pbar.update(1)
            count += 1
            # if app.index > 50000:
            # strategy.plotting = True


app = Backtest(df=df, time_increment=0.25)

fig, ax = plt.subplots()
strategy = MeanReversionStrategy(app, boll_window=2000, history_window=4000, width=1.85, plotting=False, ax=ax)

run_backtest(app, strategy)

plt.show()

with open("results.pickle", "wb") as f:
    pickle.dump(app.results, f)

# print(app.results)

# print the final report of the backtest
print(f"Total PnL: {app.total_profit:.2f}")
print(f"Number of days: {app.num_days}")
print(f"Mean daily PnL: {app.total_profit / app.num_days:.2f}")
print(f"Std daily PnL: {np.std(app._daily_returns_list):.2f}")
print(f"Unadjested Sharpe Ratio: {((app.total_profit / app.num_days)) / np.std(app._daily_returns_list):.2f}")
