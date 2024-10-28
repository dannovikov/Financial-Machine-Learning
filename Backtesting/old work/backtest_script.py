import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from tqdm import tqdm
import pytz
import pickle


class Backtest:
    def __init__(self, df, time_increment):
        time_increment_precision = len(str(time_increment).split(".")[1])
        self.next_time = round(df.time[0], time_increment_precision)
        self.last_time = df.time[0]
        self.current_hour = datetime.fromtimestamp(self.last_time).astimezone(pytz.timezone("US/Eastern")).hour
        print(self.current_hour)
        self.current_date = df.date[0]
        self.last_price = df.price[0]
        self.results = {}
        self.daily_trades = []
        self.df = df
        self.index = 0
        self.time_increment = time_increment

        self.qty = 0

        self.leverage = 50
        self.commission = 2.25

        self.tick_size = 0

        self.open_price = 0
        self.open_time = 0
        self.current_profit = 0
        self.current_min = float("inf")
        self.current_max = float("-inf")
        self.total_profit = 0
        self.max_drawdown = 0

        self.trade_id = 0

    def tickPrice(self):
        if self.index >= len(self.df):
            return None

        if self.df.date[self.index] != self.current_date:
            self.record_daily_trades()
            self.current_date = self.df.date[self.index]
            self.next_time = round(self.df.time[self.index], len(str(self.time_increment).split(".")[1]))
            self.close_all_positions()

        while self.index < len(self.df) and self.df.time[self.index] <= self.next_time:
            self.last_price = self.df.price[self.index]
            self.index += 1

        self.last_time = self.next_time
        self.current_hour = datetime.fromtimestamp(self.last_time).astimezone(pytz.timezone("US/Eastern")).hour
        self.next_time += self.time_increment

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
        self.daily_trades.append({"id": self.trade_id, "action": "open", "side": "buy", "fillPrice": slip_price, "time": self.last_time})

    def open_short_position(self):
        self.qty = -1
        slip_price = self.last_price - self.tick_size
        self.open_price = slip_price
        self.open_time = self.last_time
        self.daily_trades.append({"id": self.trade_id, "action": "open", "side": "sell", "fillPrice": slip_price, "time": self.last_time})

    def close_long_position(self):
        self.qty = 0
        slip_price = self.last_price - self.tick_size
        self.daily_trades.append({"id": self.trade_id, "action": "close", "side": "sell", "fillPrice": slip_price, "time": self.last_time, "profit": self.current_profit, "profit_low": self.current_min, "profit_high": self.current_max})
        self.trade_id += 1

    def close_short_position(self):
        self.qty = 0
        slip_price = self.last_price + self.tick_size
        self.daily_trades.append({"id": self.trade_id, "action": "close", "side": "buy", "fillPrice": slip_price, "time": self.last_time, "profit": self.current_profit, "profit_low": self.current_min, "profit_high": self.current_max})
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
    def __init__(self, app, boll_window=600, width=2, max_position_duration=10000000):
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

            if self.app.qty == 0 and not self.app.current_hour >= 16:
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

            if self.app.last_time - self.app.open_time > self.max_position_duration:
                self.app.close_all_positions()


def run_backtest(app, strategy):
    with tqdm(total=3131684) as pbar:
        while app.tickPrice() is not None:
            strategy.tick()
            pbar.update(1)


def compute_backtest_stats(results):
    total_profit = 0
    max_drawdown = 0
    max_drawup = 0
    average_daily_profit = 0
    min_daily_profit = 0
    max_daily_profit = 0
    daily_returns = []
    for day in results:
        daily_profit = 0
        for trade in results[day]:
            if trade["action"] == "close":
                total_profit += trade["profit"]
                if trade["profit_low"] < max_drawdown:
                    max_drawdown = trade["profit_low"]
                if trade["profit_high"] > max_drawup:
                    max_drawup = trade["profit_high"]
                daily_profit += trade["profit"]
        if daily_profit < min_daily_profit:
            min_daily_profit = daily_profit
        if daily_profit > max_daily_profit:
            max_daily_profit = daily_profit
        daily_returns.append(daily_profit)

    average_daily_profit = total_profit / len(app.results)

    print(f"{total_profit=}, {average_daily_profit=}, {min_daily_profit=}, {max_daily_profit=}, {max_drawdown=}, {max_drawup=}, std_of_returns={np.std(daily_returns)}")
    plt.hist(daily_returns, bins=20)
    plt.show()

    return (total_profit, max_drawdown, max_drawup, average_daily_profit, min_daily_profit, max_daily_profit, daily_returns)


def hourly_backtest_stats(results):
    hourly_results = {}
    for day in results:
        for trade in results[day]:
            if trade["action"] == "close":
                hour = datetime.fromtimestamp(trade["time"]).hour
                if hour not in hourly_results:
                    hourly_results[hour] = []
                hourly_results[hour].append(trade["profit"])

    average_hourly_profit = {}
    number_of_trades_per_hour = {}
    for hour in hourly_results:
        average_hourly_profit[hour] = np.mean(hourly_results[hour])
        number_of_trades_per_hour[hour] = len(hourly_results[hour])
    return average_hourly_profit, number_of_trades_per_hour


def after4pm_backtest_stats(results):
    after_4pm = []
    for day in results:
        for trade in results[day]:
            if trade["action"] == "close":
                hour = datetime.fromtimestamp(trade["time"]).hour
                if hour >= 16:

                    open_order = None
                    for trade2 in results[day]:
                        if trade2["id"] == trade["id"]:
                            open_order = trade2
                            break
                    after_4pm.append((open_order, trade, {"open_duration": trade["time"] - open_order["time"]}))
    return after_4pm


def plot_duration_vs_profit(results):

    trade_durations = []
    trade_profits = []

    current_open_trade = None
    for day in results:
        for trade in results[day]:
            if trade["action"] == "open":
                current_open_trade = trade
            elif trade["action"] == "close":
                trade_durations.append(trade["time"] - current_open_trade["time"])
                trade_profits.append(trade["profit"])

    plt.scatter(trade_durations, trade_profits, c=["green" if x > 0 else "red" for x in trade_profits], s=1)

    profitable_trade_indices = [i for i, x in enumerate(trade_profits) if x > 0]
    profitable_trade_durations = [trade_durations[i] for i in profitable_trade_indices]
    longest_profitable_trade_duration = max(profitable_trade_durations)

    plt.scatter([], [], c="green", s=5, label="Profitable Trade")
    plt.scatter([], [], c="red", s=5, label="Unprofitable Trade")
    plt.plot([], [], " ", label=f"Window Size = 600")
    plt.plot([], [], " ", label=f"Max Profitable Trade Duration = {int(longest_profitable_trade_duration)}")
    plt.legend(ncol=1, loc="lower right")

    plt.xlabel("Position Duration")
    plt.ylabel("Profit")
    plt.title("Position Duration vs. Profit")


def main():
    csv = "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Backtesting/ES-Trades-During-Market-Hours.csv"
    df = pd.read_csv(csv)
    assert df.time.is_monotonic_increasing

    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date
    app = Backtest(df=df, time_increment=0.25)
    strategy = MeanReversionStrategy(app)
    run_backtest(app, strategy)

    (total_profit, max_drawdown, max_drawup, average_daily_profit, min_daily_profit, max_daily_profit, daily_returns) = compute_backtest_stats(app.results)
    average_hourly_profit, number_of_trades_per_hour = hourly_backtest_stats(app.results)
    after_4pm = after4pm_backtest_stats(app.results)

    parameters = {"boll_window": strategy.boll_window, "width": strategy.width, "max_position_duration": strategy.max_position_duration}

    report = {
        "total_profit": total_profit,
        "average_daily_profit": average_daily_profit,
        "min_daily_profit": min_daily_profit,
        "max_daily_profit": max_daily_profit,
        "max_drawdown": max_drawdown,
        "max_drawup": max_drawup,
        "std_of_returns": np.std(daily_returns),
        "average_hourly_profit": average_hourly_profit,
        "number_of_trades_per_hour": number_of_trades_per_hour,
    }

    results = (parameters, report, app.results)

    with open(f"backtest_results/bw_{strategy.boll_window}_w_{strategy.width}_mpd_{strategy.max_position_duration}.pkl", "wb") as f:
        pickle.dump(results, f)

    with open(f"backtest_results/bw_600_w_2_mpd_200.pkl", "rb") as f:
        results2 = pickle.load(f)

    parameters, report, results = results2

    plot_duration_vs_profit(results)
