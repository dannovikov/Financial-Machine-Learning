import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
import pickle
import pytz
from tqdm import tqdm
from collections import deque


# Define the Backtest class
class Backtest:
    def __init__(self, df, time_increment):
        time_increment_precision = len(str(time_increment).split(".")[1]) if "." in str(time_increment) else 0
        self.next_time = round(df.time.iloc[0], time_increment_precision)
        self.last_time = df.time.iloc[0]
        self.current_hour = datetime.fromtimestamp(self.last_time).astimezone(pytz.timezone("US/Eastern")).hour
        self.current_date = df.date.iloc[0]
        self.last_price = df.price.iloc[0]
        self.results = {}
        self.daily_trades = []
        self.df = df
        self.index = 0
        self.time_increment = time_increment

        # Trading logic variables
        self.qty = 0  # Position quantity: 1 for long, -1 for short, 0 for no position

        self.leverage = 50  # Example leverage for futures trading
        self.commission = 2.25  # Commission per contract per side
        # self.tick_size = 0.25  # Fixed tick size for slippage
        self.tick_size = 0

        # Profit and drawdown variables
        self.open_price = 0
        self.open_time = 0
        self.current_profit = 0  # To track the current profit
        self.current_min = float("inf")  # To track the current drawdown
        self.current_max = float("-inf")  # To track the current profit high
        self.total_profit = 0  # To track the total profit made
        self.max_drawdown = 0

        self.trade_id = 0

    def tickPrice(self):
        if self.index >= len(self.df):
            return None

        # If index points to a new date, we need to advance the timer to the next trading day
        if self.df.date.iloc[self.index] != self.current_date:
            self.record_daily_trades()
            self.current_date = self.df.date.iloc[self.index]
            time_increment_precision = len(str(self.time_increment).split(".")[1]) if "." in str(self.time_increment) else 0
            self.next_time = round(self.df.time.iloc[self.index], time_increment_precision)
            self.close_all_positions()

        # Advance the index to keep last_price up to date with the sampling time
        while self.index < len(self.df) and self.df.time.iloc[self.index] <= self.next_time:
            self.last_price = self.df.price.iloc[self.index]
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
        if self.current_profit > self.current_max:
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


# Define the MeanReversionStrategy class
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

            if self.app.qty == 0 and not self.app.current_hour >= 16:  # Don't open a trade after 4pm
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


# Define the run_backtest function with progress updates
def run_backtest(app, strategy, boll_window, progress_position):
    total_ticks = len(app.df)
    tick_count = 0

    # Initialize a tqdm progress bar for this backtest
    with tqdm(total=total_ticks, desc=f"Boll_Window={boll_window}", position=progress_position, leave=False) as pbar:
        while app.tickPrice() is not None:
            strategy.tick()
            tick_count += 1
            pbar.update(1)
    # Print completion message
    print(f"Backtest completed for boll_window={boll_window}")


# Define the compute_backtest_stats function
def compute_backtest_stats(results):
    total_profit = 0
    max_drawdown = 0
    max_drawup = 0
    min_daily_profit = float("inf")
    max_daily_profit = float("-inf")
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

    average_daily_profit = total_profit / len(results) if len(results) > 0 else 0

    return (total_profit, max_drawdown, max_drawup, average_daily_profit, min_daily_profit, max_daily_profit, daily_returns)


# Define the function to run a single backtest
def run_single_backtest(args):
    boll_window, progress_position = args
    # Read data inside the function to avoid issues with multiprocessing
    csv = "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Backtesting/ES-Trades-During-Market-Hours.csv"
    df = pd.read_csv(csv)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date

    # Initialize the backtest and strategy
    app = Backtest(df=df, time_increment=0.25)
    strategy = MeanReversionStrategy(app, boll_window=boll_window, width=2, max_position_duration=1e20)

    # Run the backtest with progress updates
    run_backtest(app, strategy, boll_window, progress_position)

    # Save the results to a pickle file
    filename = f"results_{boll_window}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(app.results, f)

    # Compute backtest stats
    (total_profit, max_drawdown, max_drawup, average_daily_profit, min_daily_profit, max_daily_profit, daily_returns) = compute_backtest_stats(app.results)

    # Return the boll_window and the average profit
    return boll_window, average_daily_profit


if __name__ == "__main__":
    import multiprocessing
    import os

    # List of boll_window values
    boll_window_values = list(range(50, 5001, 50))  # 50 to 5000 inclusive, step 50

    num_processes = 2

    # Prepare arguments with unique progress positions
    args_list = [(boll_window, idx % num_processes + 1) for idx, boll_window in enumerate(boll_window_values)]

    # Create a manager to handle shared resources
    manager = multiprocessing.Manager()
    results = manager.list()

    # Function to collect results
    def collect_result(result):
        results.append(result)

    # Function to handle errors
    def error_callback(e):
        print(f"Error: {e}")

    # Initialize a pool with the desired number of processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the run_single_backtest function to the arguments
        for args in args_list:
            pool.apply_async(run_single_backtest, args=(args,), callback=collect_result, error_callback=error_callback)
        pool.close()
        pool.join()

    # Collect results
    boll_window_list = []
    average_profit_list = []

    for boll_window, average_profit in results:
        boll_window_list.append(boll_window)
        average_profit_list.append(average_profit)

    # Sort results by boll_window
    sorted_results = sorted(zip(boll_window_list, average_profit_list))
    boll_window_list, average_profit_list = zip(*sorted_results)

    # Plot average profit as a function of boll_window
    plt.figure(figsize=(10, 6))
    plt.plot(boll_window_list, average_profit_list, marker="o")
    plt.xlabel("Bollinger Window Size")
    plt.ylabel("Average Daily Profit")
    plt.title("Average Daily Profit vs Bollinger Window Size")
    plt.grid(True)
    plt.show()
