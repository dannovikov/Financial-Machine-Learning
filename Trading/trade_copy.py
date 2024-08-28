from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import threading
import time


class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

        self.tick_types = {
            "BID_SIZE": 0,
            "BID_PRICE": 1,
            "ASK_PRICE": 2,
            "ASK_SIZE": 3,
            "LAST_PRICE": 4,
            "LAST_SIZE": 5,
            "VOLUME": 8,
        }

        self.order_id = 100  # To track orders

        self.bid_price = 0
        self.bid_size = 0
        self.ask_price = 0
        self.ask_size = 0
        self.last_price = 0
        self.last_size = 0

        self.histlen = 3600
        self.price_history = deque(maxlen=self.histlen)

        # mean reversion data structures
        self.mr_window = 600
        self.mr_sma = deque(maxlen=self.histlen)
        self.mr_upper_band = deque(maxlen=self.histlen)
        self.mr_lower_band = deque(maxlen=self.histlen)

        # trend following data structures
        self.tf_short_window = 80
        self.tf_long_window = 160
        self.tf_sma_short = deque(maxlen=self.histlen)
        self.tf_sma_long = deque(maxlen=self.histlen)

        self.mean_side = None
        self.qty = 0
        self.open_price = 0
        self.open_dir = None
        self.open_time = 0
        self.max_open_time = float("inf")

        self.leverage = 5  # 50  # ES futures leverage
        self.commission = 0.62  # 2.25  # 0.62  # ES commission per contract per side
        self.trade_cost = self.commission * 2 / self.leverage  # for setting buy/sell bands

        self.current_profit = 0
        self.guarenteed_profit = 0
        self.total_profit = 0
        self._last_permid = None

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.order_id = orderId
        print(f"Next valid order ID: {orderId}")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        if status == "Filled":
            if self.qty != 0:  # qty having been changed by the open order
                self.open_price = avgFillPrice
                self.open_dir = "BUY" if self.qty > 0 else "SELL"
            else:
                subprofit = avgFillPrice - self.open_price
                if self.open_dir == "SELL":
                    subprofit *= -1
                subprofit *= self.leverage
                profit = subprofit - self.commission
                if self._last_permid != permId:
                    print(f"Order profit = {subprofit} - {self.commission} = {profit}")
                    print(f"PnL: {self.total_profit} -> {self.total_profit + profit}")
                    self.total_profit += profit
                    self._last_permid = permId

    def openOrder(self, orderId, contract, order, orderState):
        pass

    def historicalTicks(self, reqId, ticks, done):
        print("Historical Tick Data. ReqId:", reqId)
        for tick in ticks:
            print(f"Tick: {tick.time}, Price: {tick.price}, Size: {tick.size}")

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == self.tick_types["LAST_PRICE"]:
            self.last_price = price
            self.price_history.append(price)
            # Update mean reversion data structures
            mr_price_window = list(self.price_history)[-self.mr_window :]
            mr_mu = np.mean(mr_price_window)
            mr_std = np.std(mr_price_window)
            self.mr_sma.append(mr_mu)
            self.mr_upper_band.append(mr_mu + 2 * mr_std)
            self.mr_lower_band.append(mr_mu - 2 * mr_std)
            # Update trend following data structures
            tf_short_window = list(self.price_history)[-self.tf_short_window :]
            tf_long_window = list(self.price_history)[-self.tf_long_window :]
            self.tf_sma_short.append(np.mean(tf_short_window))
            self.tf_sma_long.append(np.mean(tf_long_window))

        elif tickType == self.tick_types["BID_PRICE"]:
            self.bid_price = price
        elif tickType == self.tick_types["ASK_PRICE"]:
            self.ask_price = price

    def tickSize(self, reqId, tickType, size):
        if tickType == self.tick_types["LAST_SIZE"]:
            self.last_size = size
        elif tickType == self.tick_types["BID_SIZE"]:
            self.bid_size = size
        elif tickType == self.tick_types["ASK_SIZE"]:
            self.ask_size = size

    def subscribe_to_market_data(self, contract):
        """Subscribe to live market data."""
        self.reqMktData(reqId=1, contract=contract, genericTickList="", snapshot=False, regulatorySnapshot=False, mktDataOptions=[])

    def place_order(self, contract, order):
        self.placeOrder(self.order_id, contract, order)
        self.order_id += 1

    def buy(self, contract, qty):
        order = self.create_order("BUY", qty)
        self.place_order(contract, order)

    def sell(self, contract, qty):
        order = self.create_order("SELL", qty)
        self.place_order(contract, order)

    def create_order(self, action, qty, price=None):
        order = Order()
        order.action = action
        order.totalQuantity = qty
        order.orderType = "LMT" if price else "MKT"
        if price:
            order.lmtPrice = price
        return order


def run_loop():
    app.run()


# def mean_reversion(app):
#     fig, ax = plt.subplots()
#     ax.set_title("Real-Time Bollinger Bands")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Price")

#     # window = 600  # at 4 ticks per second

#     # histlen = 2400
#     # sma = deque(maxlen=histlen)
#     # upper_band = deque(maxlen=histlen)
#     # lower_band = deque(maxlen=histlen)
#     # price_history = deque(maxlen=histlen)

#     def tick(_):
#         prices = list(app.price_history)[-app.mr_window :]

#         # --- Strategy ---
#         # mu = np.mean(prices)
#         # std = np.std(prices)
#         # sma.append(mu)
#         # upper_band.append(mu + 2 * std)
#         # lower_band.append(mu - 2 * std)

#         if app.qty == 1:
#             app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission
#             app.open_time += 1
#         elif app.qty == -1:
#             app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission
#             app.open_time += 1

#         # time limit condition for closing position
#         # if app.open_time > app.max_open_time:
#         #     if app.qty == 1:
#         #         app.sell(contract, 1)
#         #         app.qty = 0
#         #         print("Time is up. Selling 1 contract.")
#         #         app.open_time = 0
#         #     elif app.qty == -1:
#         #         app.buy(contract, 1)
#         #         app.qty = 0
#         #         print("Time is up. Buying 1 contract.")
#         #         app.open_time = 0

#         # open cross sma condition for closing position
#         # if len(prices) == window:
#         #     if app.qty == 1:
#         #         if app.open_price >= mu:
#         #             app.sell(contract, 1)
#         #             app.qty = 0
#         #             app.open_time = 0
#         #             print("Open price crossed the mean. Selling 1 contract.")
#         #     elif app.qty == -1:
#         #         if app.open_price <= mu:
#         #             app.buy(contract, 1)
#         #             app.qty = 0
#         #             app.open_time = 0
#         #             print("Open price crossed the mean. Buying 1 contract.")

#         if len(prices) == app.mr_window:
#             if app.qty == 0:
#                 if app.last_price < app.mr_lower_band[-1] - app.trade_cost:
#                     app.qty = 1
#                     app.buy(contract, 1)
#                     print("Price is under the lower band. Buying 1 contract.")
#                 elif app.last_price > app.mr_upper_band[-1] + app.trade_cost:
#                     app.qty = -1
#                     app.sell(contract, 1)
#                     print("Price is over the upper band. Selling 1 contract.")
#             elif app.qty == 1:
#                 if app.last_price >= mu:
#                     app.qty = 0
#                     app.sell(contract, 1)
#                     app.open_time = 0
#                     print("Price is above the mean. Selling 1 contract.")
#             elif app.qty == -1:
#                 if app.last_price <= mu:
#                     app.qty = 0
#                     app.buy(contract, 1)
#                     app.open_time = 0
#                     print("Price is below the mean. Buying 1 contract.")

#         # --- Plotting ---
#         ax.clear()
#         x = range(len(app.price_history))
#         ax.plot(x, app.price_history, label=f"Price {app.last_price:.2f}", color="blue")
#         ax.plot(x, app.mr_upper_band, label=f"Upper Band {app.mr_upper_band[-1]:.2f}", color="purple")
#         ax.plot(x, app.mr_lower_band, label=f"Lower Band {app.mr_lower_band[-1]:.2f}", color="purple")
#         ax.plot(x, [u + app.trade_cost for u in app.mr_upper_band], label=f"Sell Band {app.mr_upper_band[-1] + app.trade_cost:.2f}", color="mediumpurple")
#         ax.plot(x, [l - app.trade_cost for l in app.mr_lower_band], label=f"Buy Band {app.mr_lower_band[-1] - app.trade_cost:.2f}", color="mediumpurple")
#         ax.plot(x, app.mr_sma, label=f"SMA {app.mr_sma[-1]:.2f}", color="cornflowerblue")
#         # ax.hlines(mu, 0, len(price_history), label=f"Mean {mu:.2f}", color="royalblue")
#         if app.qty != 0:
#             ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")  # Invisible line with profit label
#             ax.hlines(app.open_price, 0, len(app.mr_price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
#         ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")  # Invisible line with total profit label
#         ax.fill_between(x, app.mr_lower_band, app.mr_upper_band, color="gray", alpha=0.2)
#         ax.set_ylim(app.mr_lower_band[-1] - app.trade_cost - 2, app.mr_upper_band[-1] + app.trade_cost + 2)
#         ax.legend(ncol=3, loc="upper center")
#         # -----------------

#     ani = animation.FuncAnimation(fig, tick, interval=250, cache_frame_data=False)
#     plt.show()


# def trend_following(app):
#     # --- Trend Following using Moving Averages ---
#     fig, ax = plt.subplots()
#     ax.set_title("Real-Time Trend Following")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Price")

#     # histlen = 3600
#     # sma_short = deque(maxlen=histlen)
#     # sma_long = deque(maxlen=histlen)

#     # short_window = 80
#     # long_window = 160

#     def tick(_):
#         # prices = app.price_history

#         # --- Strategy ---
#         # sma_short.append(np.mean(list(prices)[-short_window:]))
#         # sma_long.append(np.mean(list(prices)[-long_window:]))

#         if app.qty == 1:
#             app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission
#             app.guarenteed_profit = (app.tf_sma_long[-1] - app.open_price) * app.leverage - app.commission
#             app.open_time += 1
#         elif app.qty == -1:
#             app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission
#             app.guarenteed_profit = (app.open_price - app.tf_sma_long[-1]) * app.leverage - app.commission
#             app.open_time += 1

#         if len(prices) > app.tf_long_window:
#             if app.qty == 0:
#                 if app.tf_sma_short[-1] - app.tf_sma_long[-1] > 0.25:
#                     app.qty = 1
#                     app.buy(contract, 1)
#                     print("Open: Short SMA crossed above Long SMA. Buying 1 contract.")
#                     time.sleep(0.5)
#                 elif app.tf_sma_short[-1] - app.tf_sma_long[-1] < -0.25:
#                     app.qty = -1
#                     app.sell(contract, 1)
#                     print("Open: Short SMA crossed below Long SMA. Selling 1 contract.")
#                     time.sleep(0.5)
#             elif app.qty == 1:
#                 # if sma_short[-1] - sma_long[-1] < -0.25:
#                 if app.tf_sma_short[-1] < app.tf_sma_long[-1]:
#                     app.qty = 0
#                     app.sell(contract, 1)
#                     print("Close: Short SMA crossed below Long SMA. Selling 1 contract.")
#                     time.sleep(0.5)

#             elif app.qty == -1:
#                 # if sma_short[-1] - sma_long[-1] > 0.25:
#                 if app.tf_sma_short[-1] > app.tf_sma_long[-1]:
#                     app.qty = 0
#                     app.buy(contract, 1)
#                     print("Close: Short SMA crossed above Long SMA. Buying 1 contract.")
#                     time.sleep(0.5)

#         # --- Plotting ---
#         ax.clear()
#         x = range(len(app.price_history))
#         ax.plot(x, app.price_history, label=f"Price {app.last_price:.2f}", color="blue")
#         ax.plot(x, app.tf_sma_short, label=f"{app.tf_short_window} SMA {app.tf_sma_short[-1]:.2f}", color="gold")
#         ax.plot(x, app.tf_sma_long, label=f"{app.tf_long_window} SMA {app.tf_sma_long[-1]:.2f}", color="purple")
#         if app.qty != 0:
#             ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
#             ax.hlines(app.open_price, 0, len(app.price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
#         ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
#         ax.legend(ncol=3, loc="upper center")
#         ax.set_ylim(min(app.price_history) - 1, max(app.price_history) + 1)
#         # -----------------

#     ani = animation.FuncAnimation(fig, tick, interval=250, cache_frame_data=False)
#     plt.show()


import matplotlib.pyplot as plt
import time


class MeanReversionStrategy:
    def __init__(self):
        self.running = False

    def start(self, app, contract):
        self.running = True
        fig, ax = plt.subplots()
        ax.set_title("Real-Time Bollinger Bands")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        while self.running:
            prices = list(app.price_history)[-app.mr_window :]

            if app.qty == 1:
                app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission
                app.open_time += 1
            elif app.qty == -1:
                app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission
                app.open_time += 1

            if len(prices) == app.mr_window:
                if app.qty == 0:
                    if app.last_price < app.mr_lower_band[-1] - app.trade_cost:
                        app.qty = 1
                        app.buy(contract, 1)
                        print("Price is under the lower band. Buying 1 contract.")
                    elif app.last_price > app.mr_upper_band[-1] + app.trade_cost:
                        app.qty = -1
                        app.sell(contract, 1)
                        print("Price is over the upper band. Selling 1 contract.")
                elif app.qty == 1:
                    if app.last_price >= np.mean(prices):
                        app.qty = 0
                        app.sell(contract, 1)
                        app.open_time = 0
                        print("Price is above the mean. Selling 1 contract.")
                elif app.qty == -1:
                    if app.last_price <= np.mean(prices):
                        app.qty = 0
                        app.buy(contract, 1)
                        app.open_time = 0
                        print("Price is below the mean. Buying 1 contract.")

            # --- Plotting ---
            ax.clear()
            x = range(len(app.price_history))
            ax.plot(x, app.price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x, app.mr_upper_band, label=f"Upper Band {app.mr_upper_band[-1]:.2f}", color="purple")
            ax.plot(x, app.mr_lower_band, label=f"Lower Band {app.mr_lower_band[-1]:.2f}", color="purple")
            ax.plot(x, app.mr_sma, label=f"SMA {app.mr_sma[-1]:.2f}", color="cornflowerblue")

            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")  # Invisible line with profit label
                ax.hlines(app.open_price, 0, len(app.price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")  # Invisible line with total profit label
            ax.fill_between(x, app.mr_lower_band, app.mr_upper_band, color="gray", alpha=0.2)
            ax.set_ylim(app.mr_lower_band[-1] - app.trade_cost - 2, app.mr_upper_band[-1] + app.trade_cost + 2)
            ax.legend(ncol=3, loc="upper center")

            plt.pause(0.25)  # Pause for 250 ms to control the update rate

            # Allowing external termination
            if not self.running:
                break

        plt.close(fig)

    def stop(self):
        self.running = False


class TrendFollowingStrategy:
    def __init__(self):
        self.running = False

    def start(self, app, contract):
        self.running = True
        fig, ax = plt.subplots()
        ax.set_title("Real-Time Trend Following")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        while self.running:
            if app.qty == 1:
                app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission
                app.guarenteed_profit = (app.tf_sma_long[-1] - app.open_price) * app.leverage - app.commission
                app.open_time += 1
            elif app.qty == -1:
                app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission
                app.guarenteed_profit = (app.open_price - app.tf_sma_long[-1]) * app.leverage - app.commission
                app.open_time += 1

            if len(app.price_history) > app.tf_long_window:
                if app.qty == 0:
                    if app.tf_sma_short[-1] > app.tf_sma_long[-1] + 0.25:
                        app.qty = 1
                        app.buy(contract, 1)
                        print("Open: Short SMA crossed above Long SMA. Buying 1 contract.")
                        time.sleep(0.5)
                    elif app.tf_sma_short[-1] < app.tf_sma_long[-1] - 0.25:
                        app.qty = -1
                        app.sell(contract, 1)
                        print("Open: Short SMA crossed below Long SMA. Selling 1 contract.")
                        time.sleep(0.5)
                elif app.qty == 1:
                    if app.tf_sma_short[-1] < app.tf_sma_long[-1]:
                        app.qty = 0
                        app.sell(contract, 1)
                        print("Close: Short SMA crossed below Long SMA. Selling 1 contract.")
                        time.sleep(0.5)

                elif app.qty == -1:
                    if app.tf_sma_short[-1] > app.tf_sma_long[-1]:
                        app.qty = 0
                        app.buy(contract, 1)
                        print("Close: Short SMA crossed above Long SMA. Buying 1 contract.")
                        time.sleep(0.5)

            # --- Plotting ---
            ax.clear()
            x = range(len(app.price_history))
            ax.plot(x, app.price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x, app.tf_sma_short, label=f"{app.tf_short_window} SMA {app.tf_sma_short[-1]:.2f}", color="gold")
            ax.plot(x, app.tf_sma_long, label=f"{app.tf_long_window} SMA {app.tf_sma_long[-1]:.2f}", color="purple")

            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.hlines(app.open_price, 0, len(app.price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.legend(ncol=3, loc="upper center")
            ax.set_ylim(min(app.price_history) - 1, max(app.price_history) + 1)

            plt.pause(0.25)  # Pause for 250 ms to control the update rate

            # Allowing external termination
            if not self.running:
                break

        plt.close(fig)

    def stop(self):
        self.running = False


app = IBApi()
app.connect("127.0.0.1", 7497, 0)  # Ensure TWS is running on this port with Paper Trading login
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()
time.sleep(1)  # Sleep to ensure connection is established


# Define the ES continuous futures contract
contract = Contract()
contract.symbol = "MES"  # "MES"
contract.secType = "FUT"
contract.exchange = "CME"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = "202409"

# Subscribe to live market data for ES
print("Subscribing to market data")
app.subscribe_to_market_data(contract)

# Wait for price data
while not app.last_price:
    wait_time = 0
    print(f"Waiting for price data... {wait_time} s\r", end="")
    time.sleep(1)
print(" " * 50, end="\r")


# mean_reversion(app)
# trend_following(app)

import threading


class StrategyManager:
    def __init__(self, app, contract):
        self.app = app
        self.contract = contract
        self.current_strategy_thread = None
        self.current_strategy = None

    def set_strategy(self, strategy_class):
        if self.current_strategy_thread and self.current_strategy_thread.is_alive():
            print(f"Stopping {self.current_strategy.__class__.__name__} strategy.")
            self.current_strategy.stop()
            self.current_strategy_thread.join()  # Wait for the thread to finish
        self.current_strategy = strategy_class()
        print(f"Starting {strategy_class.__name__} strategy.")
        self.current_strategy_thread = threading.Thread(target=self.current_strategy.start, args=(self.app, self.contract))
        self.current_strategy_thread.start()

    def switch_strategy(self, new_strategy_class):
        self.set_strategy(new_strategy_class)


# class MeanReversionStrategy:
#     def __init__(self):
#         self.running = False

#     def start(self, app, contract):
#         self.running = True
#         mean_reversion(app)

#     def stop(self):
#         self.running = False
#         plt.close()  # Close the Matplotlib window


# class TrendFollowingStrategy:
#     def __init__(self):
#         self.running = False

#     def start(self, app, contract):
#         self.running = True
#         trend_following(app)

#     def stop(self):
#         self.running = False
#         plt.close()  # Close the Matplotlib window


# Initialize strategies
mean_reversion_strategy = MeanReversionStrategy
trend_following_strategy = TrendFollowingStrategy

# Initialize Strategy Manager
manager = StrategyManager(app, contract)

# Start with mean reversion strategy
manager.set_strategy(mean_reversion_strategy)

# Switch strategy after some time (e.g., 5 seconds)
time.sleep(5)
manager.switch_strategy(trend_following_strategy)
