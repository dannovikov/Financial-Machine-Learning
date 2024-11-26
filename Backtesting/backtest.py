import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import numpy as np
import pandas as pd
import pytz
import copy
from collections import deque
from datetime import datetime
import pickle


holidays = {
    "2024-05-27": "Memorial Day",
    "2024-06-19": "Juneteenth",
}


class Backtest:
    def __init__(self, df, time_increment):
        self.df = df
        self.index = 0  # index into the dataframe
        self.strategy = None

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

        self.daily_returns = []

    def tick_price(self):

        if self.index + 1 >= len(self.df):
            return None

        if self.current_hour >= 16 and self.current_minute >= 30:
            print(f"date: {self.current_date}, time: {self.current_hour}:{self.current_minute}, advancing to next day")
            self._advance_to_next_trading_day()
            self.strategy._daily_reset()
            self.daily_total_profit = 0
            self.num_days += 1
            print(f"new date: {self.current_date}, time: {self.current_hour}:{self.current_minute}")

        if self.index >= len(self.df):
            return None

        # advance to the next price sampling time
        while self.df.time[self.index] <= self.next_time:
            self.index += 1
            if self.index >= len(self.df):
                return None

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
        trade_profit = (slip_price - self.open_price) * self.leverage - self.commission * 2

        self.daily_trades.append(
            {
                "id": self.trade_id,
                "action": "close",
                "side": "sell",
                "fillPrice": slip_price,
                "time": self.last_time,
                "profit": trade_profit,
                "profit_low": self.current_min,
                "profit_high": self.current_max,
            }
        )

        self.trade_id += 1
        self.total_profit += trade_profit
        self.daily_total_profit += trade_profit

    def close_short_position(self):
        self.qty = 0
        slip_price = self.last_price + self.tick_size
        trade_profit = (self.open_price - slip_price) * self.leverage - self.commission * 2
        self.daily_trades.append(
            {
                "id": self.trade_id,
                "action": "close",
                "side": "buy",
                "fillPrice": slip_price,
                "time": self.last_time,
                "profit": trade_profit,
                "profit_low": self.current_min,
                "profit_high": self.current_max,
            }
        )
        self.trade_id += 1
        self.total_profit += trade_profit
        self.daily_total_profit += trade_profit

    def close_all_positions(self):
        if self.qty == 1:
            self.close_long_position()
        elif self.qty == -1:
            self.close_short_position()

    def record_daily_trades(self):
        if not self.daily_trades:
            print(f"Date: {self.current_date}, No trades today")
            return
        self.results[self.current_date] = copy.deepcopy(self.daily_trades)
        self.daily_returns.append(self.daily_total_profit)
        print(
            f"Date: {self.current_date}, Daily PnL: {self.daily_total_profit:.2f}, Num Trades: {len(self.daily_trades)}, total PnL: {self.total_profit:.2f}, num days: {self.num_days}, mean daily PnL: {self.total_profit / self.num_days:.2f}, std daily PnL: {np.std(self.daily_returns):.2f}"
        )
        self.daily_trades = []

    def _advance_to_next_trading_day(self):
        self.record_daily_trades()
        while self.index < len(self.df) and (self._is_weekend() or self._is_outside_trading_hours()):
            self.index += 1
        if self.index >= len(self.df):
            return None
        self.last_price = self.df.price[self.index]
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
        return self._is_before_trading_hours() or self._is_after_trading_hours() or self.df.date[self.index].strftime("%Y-%m-%d") in holidays


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

        self.slope = 0

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
            before_3_28_pm = self.app.current_hour < 15 or (self.app.current_hour == 15 and self.app.current_minute < 28)
            after_9_48_am = self.app.current_hour > 9 or (self.app.current_hour == 9 and self.app.current_minute > 48)
            if self.app.qty == 0 and after_9_48_am and before_3_28_pm:
                # if self.app.last_price < lower_band - self.app.tick_size:
                if self.app.last_price < lower_band:
                    self.app.open_long_position()
                # elif self.app.last_price > upper_band + self.app.tick_size:
                elif self.app.last_price > upper_band:
                    self.app.open_short_position()
            elif self.app.qty == 1:
                if self.app.last_price >= mu:
                    self.app.close_long_position()
            elif self.app.qty == -1:
                if self.app.last_price <= mu:
                    self.app.close_short_position()

        if self.plotting and len(self.prices) == self.boll_window and self.app.index % 2 == 0:
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
            self.ax.plot([], [], " ", label=f"Time {self.app.current_hour}:{self.app.current_minute:02}")
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
            # plt.pause(0.15)

    def _daily_reset(self):
        self.prices.clear()


class TrendFollowingStrategy:
    def __init__(self, app, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015, entry_macd=0.50, exit_cooldown=60, plotting=True, ax=None):
        self.app = app
        self.slow_alpha = slow_alpha
        self.fast_alpha = fast_alpha
        self.take_profit_alpha = take_profit_alpha
        self.entry_macd = entry_macd
        self.plotting = plotting
        self.ax = ax

        self.history = 8000
        self.price_history = deque(maxlen=self.history)
        self.slow_emas = deque(maxlen=self.history)
        self.fast_emas = deque(maxlen=self.history)
        self.take_profit_emas = deque(maxlen=self.history)

        self.slow_emas.append(app.last_price)
        self.fast_emas.append(app.last_price)
        self.take_profit_emas.append(app.last_price)
        self.price_history.append(app.last_price)

        self.color_map = mcolors.LinearSegmentedColormap.from_list("slope", [(0, "firebrick"), (0.5, "grey"), (1, "forestgreen")])
        self.stop_slope_length = 250
        self.entry_slope_length = 2000
        self.entry_line = None
        self.stop_line = None

        # self.exit_cooldown = 15 * 4  # 15 seconds at 4 ticks per second
        self.exit_cooldown = exit_cooldown
        self.cooldown = max(
            self.entry_slope_length, self.stop_slope_length
        )  # set to allow for slope to accumulate enough ticks. TODO: make this a parameter and check for it

    def tick(self):
        self.app.calculate_current_profit()

        slow_ema = self.slow_alpha * self.app.last_price + (1 - self.slow_alpha) * self.slow_emas[-1]
        fast_ema = self.fast_alpha * self.app.last_price + (1 - self.fast_alpha) * self.fast_emas[-1]
        take_profit_ema = self.take_profit_alpha * self.app.last_price + (1 - self.take_profit_alpha) * self.take_profit_emas[-1]

        self.price_history.append(self.app.last_price)
        self.slow_emas.append(slow_ema)
        self.fast_emas.append(fast_ema)
        self.take_profit_emas.append(take_profit_ema)

        before_4_12_pm = self.app.current_hour < 16 or (self.app.current_hour == 16 and self.app.current_minute < 12)
        after_9_36_am = self.app.current_hour > 9 or (self.app.current_hour == 9 and self.app.current_minute > 36)

        if len(self.price_history) >= max(self.entry_slope_length, self.stop_slope_length):
            self.entry_line = np.polyfit(range(self.entry_slope_length), list(self.price_history)[-self.entry_slope_length :], 1)
            self.stop_line = np.polyfit(range(self.stop_slope_length), list(self.price_history)[-self.stop_slope_length :], 1)

        if self.cooldown > 0:
            self.cooldown -= 1

        if self.cooldown == 0:

            if self.app.qty == 0 and after_9_36_am and before_4_12_pm:
                if fast_ema > slow_ema + self.entry_macd and self.entry_line[0] > 2e-3 and self.stop_line[0] > 0:
                    self.app.open_long_position()
                elif fast_ema < slow_ema - self.entry_macd and self.entry_line[0] < -2e-3 and self.stop_line[0] < 0:
                    self.app.open_short_position()
            elif self.app.qty == 1:
                if fast_ema < take_profit_ema or self.stop_line[0] < 0:  # -0.1 / 1000:
                    self.app.close_long_position()
                    self.cooldown = self.exit_cooldown
            elif self.app.qty == -1:
                if fast_ema > take_profit_ema or self.stop_line[0] > 0:  # 0.1 / 1000:
                    self.app.close_short_position()
                    self.cooldown = self.exit_cooldown

        if self.plotting and self.app.index % 1 == 0:
            x_axis = range(len(self.price_history))
            self.ax.clear()
            self.ax.plot(x_axis, self.price_history, label=f"Price {self.app.last_price:.2f}", color="blue")
            self.ax.plot(x_axis, self.slow_emas, label=f"{self.slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
            self.ax.plot(x_axis, self.fast_emas, label=f"{self.fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
            if self.take_profit_alpha != self.slow_alpha:
                self.ax.plot(
                    x_axis,
                    self.take_profit_emas,
                    label=f"{self.take_profit_alpha:.4f} Take Profit EMA {take_profit_ema:.2f}",
                    color="lightcoral",
                )
            self.ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            if self.app.qty != 0:
                self.ax.plot([], [], " ", label=f"Current Profit {self.app.current_profit:.2f}")
                self.ax.hlines(
                    self.app.open_price,
                    0,
                    len(self.price_history),
                    label=f"Open Price {self.app.open_price:.2f}",
                    color="forestgreen" if self.app.qty == 1 else "firebrick",
                )
            self.ax.plot([], [], " ", label=f"Total PnL {self.app.total_profit:.2f}")
            self.ax.plot([], [], " ", label=f"PnL {self.app.daily_total_profit:.2f}")
            self.ax.plot([], [], " ", label=f"Dl. Avg. PnL {self.app.total_profit / self.app.num_days:.2f}")
            self.ax.plot([], [], " ", label=f"Time {self.app.current_hour}:{self.app.current_minute:02}")

            # slope of the last 100 prices
            if len(self.price_history) > max(self.entry_slope_length, self.stop_slope_length):
                entry_slope = self.entry_line[0]
                entry_bias = self.entry_line[1]

                stop_slope = self.stop_line[0]
                stop_bias = self.stop_line[1]

                # self.ax.plot([], [], " ", label=f"Slope ({self.slope_length}) {slope:.8f}")
                # plot a line of the slope over the past slope_length prices, using the polyfit parameters

                self.ax.plot(
                    np.arange(self.entry_slope_length) + len(self.price_history) - self.entry_slope_length,
                    [entry_slope * i + entry_bias for i in range(self.entry_slope_length)],
                    label=f"Slope (2000) {entry_slope * 1000:.2f}",
                    color=self.color_map(mcolors.Normalize(vmin=-2.5, vmax=2.5)(entry_slope * 1000)),
                )

                self.ax.plot(
                    np.arange(self.stop_slope_length) + len(self.price_history) - self.stop_slope_length,
                    [stop_slope * i + stop_bias for i in range(self.stop_slope_length)],
                    label=f"Stop Slope ({self.stop_slope_length}) {stop_slope * 1000:.2f}",
                    color=self.color_map(mcolors.Normalize(vmin=-2.5, vmax=2.5)(stop_slope * 1000)),
                )

            else:
                self.ax.plot(
                    [], [], " ", label=f"Slope Calibrating {len(self.price_history)} / {max(self.entry_slope_length, self.stop_slope_length)}"
                )
            if not before_4_12_pm:
                self.ax.plot([], [], " ", label="After 4:12pm, not trading")
            if not after_9_36_am:
                self.ax.plot([], [], " ", label="Before 9:36am, not trading")

            self.ax.set_ylim(min(self.price_history) - 2, max(self.price_history) + 2)
            self.ax.legend(ncol=1, loc="upper left")
            # plt.pause(0.15)
            plt.pause(1e-8)
        else:
            print(
                f"{self.app.index / 6510009 + 1:.2f}% -- Profit: {self.app.total_profit:.2f} Sharpe: {np.mean(self.app.daily_returns) / np.std(self.app.daily_returns):.2f}"
            )

    def _daily_reset(self):
        self.slow_emas[-1] = self.app.last_price
        self.fast_emas[-1] = self.app.last_price
        self.take_profit_emas[-1] = self.app.last_price


def run_backtest(app, strategy):
    app.strategy = strategy
    with tqdm(total=6510009) as pbar:
        # with tqdm(total=1370551) as pbar:  # this is through 2024-04-04
        strategy.plotting = False
        while app.tick_price() is not None and app.index < len(app.df):
            strategy.tick()
            pbar.update(1)
            if app.index > 0:
                strategy.plotting = True
            # # if date > 2024-04-04, break
            # if app.current_date > datetime.strptime("2024-04-04", "%Y-%m-%d").date():
            #     break
            # # strategy.plotting = True
            # # if app.index > 90000:
            # #     strategy.plotting = True


# # Load the data
# csv = "/Users/dan/Documents/Finance/Programs/Financial Machine Learning/Preprocessing/esm4.csv"
# df = pd.read_csv(csv)

# # df = pd.read_csv("dt_esm4.csv")

# # Construct the date and time columns from unix timestamps
# print(f"Building date and time columns...")
# dt = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("US/Eastern")
# df["date"] = dt.dt.date
# df["dayofweek"] = dt.dt.dayofweek
# df["hour"] = dt.dt.hour
# df["minute"] = dt.dt.minute

# # save the csv
# # df.to_csv("dt_esm4.csv", index=False)
# with open("dt_esm4.pkl", "wb") as f:
#     pickle.dump(df, f)

print("loading data...")
with open("dt_esm4.pkl", "rb") as f:
    df = pickle.load(f)

print("done.")

# Backtest the strategy
fig, ax = plt.subplots()
app = Backtest(df=df, time_increment=0.25)


# strategy = MeanReversionStrategy(app, boll_window=2500, history_window=2500, width=1.5, plotting=True, ax=ax)
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0005, fast_alpha=0.001, take_profit_alpha=0.0005, entry_macd=0.85, plotting=True, ax=ax)# -0.02 sharp mu -45.84 Std: 1869.59
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0001, fast_alpha=0.001, take_profit_alpha=0.0001, entry_macd=0.85, plotting=False, ax=ax) #-0 sharp mu -8 std 2088
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0001, fast_alpha=0.001, take_profit_alpha=0.0001, entry_macd=1.1, plotting=True, ax=ax) # 0.01 sharp mu 25 std 2081
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0001, fast_alpha=0.0005, take_profit_alpha=0.0001, entry_macd=1.1, plotting=True, ax=ax) #-0.01/-21/2029
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0001, fast_alpha=0.0005, take_profit_alpha=0.0001, entry_macd=1.7, plotting=True, ax=ax) #-0.03/-58/2017
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0001, fast_alpha=0.0004, take_profit_alpha=0.0001, entry_macd=1.0, plotting=True, ax=ax) #-0.04/-81/2092
# strategy = TrendFollowingStrategy(app, slow_alpha=0.0001, fast_alpha=0.001, take_profit_alpha=0.0001, entry_macd=1.35, plotting=True, ax=ax) #0.02/31.5/2085
# strategy = TrendFollowingStrategy(
#     app, slow_alpha=0.0001, fast_alpha=0.001, take_profit_alpha=0.0001, entry_macd=1, exit_cooldown=0, ax=ax
# )  # 0.01/11.6/2072

strategy = TrendFollowingStrategy(app, slow_alpha=0.001, fast_alpha=0.005, take_profit_alpha=0.001, entry_macd=0.75, exit_cooldown=0, ax=ax)

run_backtest(app, strategy)
plt.show()

# results = {}

# for boll_window in np.linspace(400, 5000, 47):
# #     print(f"Running backtest with boll_window: {boll_window}")
# #     fig, ax = plt.subplots()
# #     app = Backtest(df=df, time_increment=0.25)
# #     strategy = MeanReversionStrategy(app, boll_window=int(boll_window), history_window=int(boll_window), width=1.5, plotting=False, ax=ax)
# #     run_backtest(app, strategy)
# #     plt.close(fig)
# #     print(f"Total PnL: {app.total_profit:.2f}")
# #     print(f"Number of days: {app.num_days}")
# #     print(f"Mean daily PnL: {app.total_profit / app.num_days:.2f}")
# #     print(f"Std daily PnL: {np.std(app.daily_returns):.2f}")
# #     print(f"Unadjested Sharpe Ratio: {((app.total_profit / app.num_days)) / np.std(app.daily_returns):.2f}")
# #     print(f"Daily returns: {app.daily_returns}")
# #     results[boll_window] = {
# #         "total_profit": app.total_profit,
# #         "num_days": app.num_days,
# #         "mean_daily_pnl": app.total_profit / app.num_days,
# #         "std_daily_pnl": np.std(app.daily_returns),
# #         "sharpe_ratio": ((app.total_profit / app.num_days)) / np.std(app.daily_returns),
# #         "daily_returns": app.daily_returns,
# #     }


# print(f"Results: {results}")


# save the results
# with open("results.pickle", "wb") as f:
#     #     pickle.dump(app.results, f)
#     pickle.dump(results, f)

# print the final report of the backtest


print(f"Total PnL: {app.total_profit:.2f}")
print(f"Number of days: {app.num_days}")
print(f"Mean daily PnL: {app.total_profit / app.num_days:.2f}")
print(f"Std daily PnL: {np.std(app.daily_returns):.2f}")
print(f"Unadjusted Sharpe Ratio: {((app.total_profit / app.num_days)) / np.std(app.daily_returns):.2f}")
print(f"Daily returns: {app.daily_returns}")

# lets plot the daily returns as a histogram
plt.hist(app.daily_returns, bins=20)
plt.show()
# plot daily returns as a bar chart
# plt.bar(range(len(app.daily_returns)), app.daily_returns)
# lets use dates for the x-axis
plt.bar([str(date) for date in app.results.keys()], app.daily_returns)
plt.xticks(rotation=90)
plt.show()
