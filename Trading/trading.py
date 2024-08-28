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
        self.price_history = deque(maxlen=3600)

        self.mean_side = None
        self.qty = 0
        self.open_price = 0
        self.open_dir = None
        self.open_time = 0
        self.max_open_time = float("inf")

        self.leverage = 50  # 5  # ES futures leverage
        self.commission = 2.25  # 0.62  # ES commission per contract per side
        self.trade_cost = self.commission * 2 / self.leverage  # for setting buy/sell bands

        self.stop_loss = -2 * self.leverage  # stop loss in dollars
        self.max_open_time = 300  # max number of ticks to hold a position

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
                profit = subprofit - self.commission*2
                if self._last_permid != permId:
                    print(f"Order profit = {subprofit} - {self.commission*2} = {profit}", end = "\t")
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
    
    def close_position(self):
        if self.qty == 1:
            self.sell(contract, 1)
            self.qty = 0
            print("Closing long position. Selling 1 contract.")
        elif self.qty == -1:
            self.buy(contract, 1)
            self.qty = 0
            print("Closing short position. Buying 1 contract.")

    def close_long_position(self):
        assert self.qty == 1, "Close long position called when not long"
        self.sell(contract, 1)
        self.qty = 0
        self.open_time = 0
        print("Closing long position. Selling 1 contract.")

    def close_short_position(self):
        assert self.qty == -1, "Close short position called when not short"
        self.buy(contract, 1)
        self.qty = 0
        self.open_time = 0
        print("Closing short position. Buying 1 contract.")

    def open_long_position(self):
        assert app.qty == 0, "Open long position called when not flat"
        self.qty = 1
        self.buy(contract, 1)
        print("Opening long position. Buying 1 contract.")

    def open_short_position(self):
        assert self.qty == 0, "Open short position called when not flat"
        self.qty = -1
        self.sell(contract, 1)
        print("Opening short position. Selling 1 contract.")


def run_loop():
    app.run()

def mean_reversion(app, window=600, width=2):
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Bollinger Bands")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    # window = 600  # at 4 ticks per second
    prices = deque(maxlen=window)

    histlen = 2400
    sma = deque(maxlen=histlen)
    upper_band = deque(maxlen=histlen)
    lower_band = deque(maxlen=histlen)
    price_history = deque(maxlen=histlen)

    def tick():
        prices.append(app.last_price)
        price_history.append(app.last_price)

        # --- Strategy ---
        mu = np.mean(prices)
        std = np.std(prices)

        sma.append(mu)
        upper_band.append(mu + width * std)
        lower_band.append(mu - width * std)

        if app.qty != 0:
            app.open_time += 1
            if app.qty == 1:
                app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission*2
            elif app.qty == -1:
                app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission*2
            if app.current_profit < app.stop_loss:
                app.close_position()
            if app.open_time > app.max_open_time:
                app.close_position()

        if len(prices) == window:

            if app.qty == 0:
                if app.last_price < lower_band[-1] - app.trade_cost:
                    print("Price is under the lower band...")
                    app.open_long_position()

                elif app.last_price > upper_band[-1] + app.trade_cost:
                    print("Price is over the upper band...")
                    app.open_short_position()

            elif app.qty == 1:
                if app.last_price >= mu:
                    print("Price crossed above the mean...")
                    app.close_long_position()

            elif app.qty == -1:
                if app.last_price <= mu:
                    print("Price crossed below the mean...")
                    app.close_short_position()


        # -------- Plotting --------
        x = range(len(price_history))
        buy_band  = [l - app.trade_cost for l in lower_band]
        sell_band = [u + app.trade_cost for u in upper_band]
        ax.clear()
        ax.plot(x,  price_history,  label=f"Price {app.last_price:.2f}", color="blue")
        ax.plot(x,   upper_band,    label=f"Upper Band {upper_band[-1]:.2f}", color="purple")
        ax.plot(x,   lower_band,    label=f"Lower Band {lower_band[-1]:.2f}", color="purple")
        ax.plot(x,    sell_band,    label=f"Sell Band {upper_band[-1] + app.trade_cost:.2f}", color="mediumpurple")
        ax.plot(x,    buy_band,     label=f"Buy Band {lower_band[-1] - app.trade_cost:.2f}", color="mediumpurple")
        ax.plot(x,      sma,        label=f"SMA {mu:.2f}", color="cornflowerblue")
        ax.plot([],[],  " ",        label=f"PnL {app.total_profit:.2f}")  
        ax.plot([],[],  " ",        label=f"Std: {std:.2f}")
        if app.qty != 0:
            ax.plot([],[], " ", label=f"Current Profit {app.current_profit:.2f}")  # Invisible line with profit label
            ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
        ax.fill_between(x, lower_band, upper_band, color="gray", alpha=0.2)
        ax.set_ylim(lower_band[-1] - app.trade_cost - 2, upper_band[-1] + app.trade_cost + 2)
        ax.legend(ncol=3, loc="upper center")


    # ani = animation.FuncAnimation(fig, tick, interval=250, cache_frame_data=False) # type: ignore
    while True:
        tick()
        plt.pause(0.25)
    plt.show()


def mean_reversion_with_shutoff_at_high_volatility(app, window=600, max_std=1.0):
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Bollinger Bands with Volatility Shutoff")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    prices = deque(maxlen=window)

    histlen = 2400
    sma = deque(maxlen=histlen)
    upper_band = deque(maxlen=histlen)
    lower_band = deque(maxlen=histlen)
    price_history = deque(maxlen=histlen)

    def tick(_):
        prices.append(app.last_price)
        price_history.append(app.last_price)

        # --- Strategy ---
        mu = np.mean(prices)
        std = np.std(prices)
        sma.append(mu)
        upper_band.append(mu + 2 * std)
        lower_band.append(mu - 2 * std)

 
        if app.qty != 0:
            if app.qty == 1:
                app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission * 2
                app.open_time += 1
            elif app.qty == -1:
                app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission * 2
                app.open_time += 1
            if app.current_profit < app.stop_loss:
                app.close_position()


        if len(prices) == window:
            if std < max_std:
                if app.qty == 0:
                    if app.last_price < lower_band[-1] - app.trade_cost:
                        app.qty = 1
                        app.buy(contract, 1)
                        print("Price is under the lower band. Buying 1 contract.")
                    elif app.last_price > upper_band[-1] + app.trade_cost:
                        app.qty = -1
                        app.sell(contract, 1)
                        print("Price is over the upper band. Selling 1 contract.")
                elif app.qty == 1:
                    if app.last_price >= mu:
                        app.qty = 0
                        app.sell(contract, 1)
                        app.open_time = 0
                        print("Price is above the mean. Selling 1 contract.")
                elif app.qty == -1:
                    if app.last_price <= mu:
                        app.qty = 0
                        app.buy(contract, 1)
                        app.open_time = 0
                        print("Price is below the mean. Buying 1 contract.")
            else:
                if app.qty == 1:
                    app.sell(contract, 1)
                    app.qty = 0
                    app.open_time = 0
                    print("Volatility is too high. Selling 1 contract.")
                elif app.qty == -1:
                    app.buy(contract, 1)
                    app.qty = 0
                    app.open_time = 0
                    print("Volatility is too high. Buying 1 contract.")

        # --- Plotting ---
        ax.clear()
        x = range(len(price_history))
        ax.plot(x, price_history, label=f"Price {app.last_price:.2f}", color="blue")
        ax.plot(x, upper_band, label=f"Upper Band {upper_band[-1]:.2f}", color="purple")
        ax.plot(x, lower_band, label=f"Lower Band {lower_band[-1]:.2f}", color="purple")
        ax.plot(x, [u + app.trade_cost for u in upper_band], label=f"Sell Band {upper_band[-1] + app.trade_cost:.2f}", color="mediumpurple")
        ax.plot(x, [l - app.trade_cost for l in lower_band], label=f"Buy Band {lower_band[-1] - app.trade_cost:.2f}", color="mediumpurple")
        ax.plot(x, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
        # ax.hlines(mu, 0, len(price_history), label=f"Mean {mu:.2f}", color="royalblue")
        if app.qty != 0:
            ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
            ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
        ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
        ax.plot([],[], " ", label=f"std: {std:.2f}")
        ax.fill_between(x, lower_band, upper_band, color="gray", alpha=0.2)
        ax.set_ylim(lower_band[-1] - app.trade_cost - 2, upper_band[-1] + app.trade_cost + 2)
        ax.legend(ncol=3, loc="upper center")
        # -----------------

    ani = animation.FuncAnimation(fig, tick, interval=250, cache_frame_data=False)
    plt.show()


def trend_following(app):
    # --- Trend Following using Moving Averages ---
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Trend Following")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    histlen = 3600
    prices = deque(maxlen=histlen)
    sma_short = deque(maxlen=histlen)
    sma_long = deque(maxlen=histlen)
    price_history = deque(maxlen=histlen)

    short_window = 50
    long_window = 200

    def tick(_):
        prices.append(app.last_price)
        price_history.append(app.last_price)

        # --- Strategy ---
        sma_short.append(np.mean(list(prices)[-short_window:]))
        sma_long.append(np.mean(list(prices)[-long_window:]))

        if app.qty == 1:
            app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission*2
            app.guarenteed_profit = (sma_long[-1] - app.open_price) * app.leverage - app.commission*2
            app.open_time += 1
        elif app.qty == -1:
            app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission*2
            app.guarenteed_profit = (app.open_price - sma_long[-1]) * app.leverage - app.commission*2
            app.open_time += 1

        if len(prices) > long_window:
            if app.qty == 0:
                if sma_short[-1] - sma_long[-1] > 0.25:
                    app.qty = 1
                    app.buy(contract, 1)
                    print("Open: Short SMA crossed above Long SMA. Buying 1 contract.")
                    time.sleep(0.5)
                elif sma_short[-1] - sma_long[-1] < -0.25:
                    app.qty = -1
                    app.sell(contract, 1)
                    print("Open: Short SMA crossed below Long SMA. Selling 1 contract.")
                    time.sleep(0.5)
            elif app.qty == 1:
                if sma_short[-1] < sma_long[-1]:
                    app.qty = 0
                    app.sell(contract, 1)
                    print("Close: Short SMA crossed below Long SMA. Selling 1 contract.")
                    time.sleep(0.5)

            elif app.qty == -1:
                if sma_short[-1] > sma_long[-1]:
                    app.qty = 0
                    app.buy(contract, 1)
                    print("Close: Short SMA crossed above Long SMA. Buying 1 contract.")
                    time.sleep(0.5)

        # --- Plotting ---
        ax.clear()
        x = range(len(price_history))
        ax.plot(x, price_history, label=f"Price {app.last_price:.2f}", color="blue")
        ax.plot(x, sma_short, label=f"{short_window} SMA {sma_short[-1]:.2f}", color="gold")
        ax.plot(x, sma_long, label=f"{long_window} SMA {sma_long[-1]:.2f}", color="purple")
        if app.qty != 0:
            ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
            ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
        ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
        ax.legend(ncol=3, loc="upper center")
        ax.set_ylim(min(price_history) - 1, max(price_history) + 1)
        # -----------------

    ani = animation.FuncAnimation(fig, tick, interval=250, cache_frame_data=False)
    plt.show()


def mean_reversion_and_trend_following_switching(app):
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Mean Reversion and Trend Following Switching")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    mr_window = 600  
    tf_short_window = 50
    tf_long_window = 200
    up = down = 2

    histlen = 3600
    prices = deque(maxlen=histlen)
    upper_band = deque(maxlen=histlen)
    lower_band = deque(maxlen=histlen)
    sma_mr = deque(maxlen=histlen)
    sma_short = deque(maxlen=histlen)
    sma_long = deque(maxlen=histlen)
    stds = deque(maxlen=histlen)

    app.current_strategy = "mean_reversion"
    stategy_switch_threshold = 0.50

    def tick(_):
        prices.append(app.last_price)

        # --- Strategy Data Structures ---
        mu_mr = np.mean(list(prices)[-mr_window:])
        std = np.std(list(prices)[-mr_window:])
        upper_band.append(mu_mr + up * std)
        lower_band.append(mu_mr - down * std)
        stds.append(std)
        sma_mr.append(mu_mr)
        sma_short.append(np.mean(list(prices)[-tf_short_window:]))
        sma_long.append(np.mean(list(prices)[-tf_long_window:]))
        std_short = np.std(list(prices)[-tf_short_window:])
        std_long = np.std(list(prices)[-tf_long_window:])
        # ---------------------------------

        if app.qty == 1:
            app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission*2
        elif app.qty == -1:
            app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission*2

        if len(prices) > mr_window:
            if app.current_strategy == "mean_reversion":
                if std >= stategy_switch_threshold:
                    if app.qty == 1:
                        app.sell(contract, 1)
                        app.qty = 0
                        print("Switching to Trend Following. Selling 1 contract.")
                        time.sleep(0.5)
                    elif app.qty == -1:
                        app.buy(contract, 1)
                        app.qty = 0
                        print("Switching to Trend Following. Buying 1 contract.")
                        time.sleep(0.5)
                    app.current_strategy = "trend_following"
                    print("Switching to Trend Following.")
                else:
                    # logic for mean reversion
                    if app.qty == 0: # logic for opening a position
                        if app.last_price < lower_band[-1] - app.trade_cost:
                            app.qty = 1
                            app.buy(contract, 1)
                            print("Price is under the lower band. Buying 1 contract.")
                        elif app.last_price > upper_band[-1] + app.trade_cost:
                            app.qty = -1
                            app.sell(contract, 1)
                            print("Price is over the upper band. Selling 1 contract.")
                    elif app.qty == 1: # logic for closing a long position
                        if app.last_price >= mu_mr:
                            app.qty = 0
                            app.sell(contract, 1)
                            print("Price is above the mean. Selling 1 contract.")
                    elif app.qty == -1: # logic for closing a short position
                        if app.last_price <= mu_mr:
                            app.qty = 0
                            app.buy(contract, 1)
                            print("Price is below the mean. Buying 1 contract.")
            elif app.current_strategy == "trend_following":
                if std < stategy_switch_threshold:
                    if app.qty == 1:
                        app.sell(contract, 1)
                        app.qty = 0
                        print("Switching to Mean Reversion. Selling 1 contract.")
                        time.sleep(0.5)
                    elif app.qty == -1:
                        app.buy(contract, 1)
                        app.qty = 0
                        print("Switching to Mean Reversion. Buying 1 contract.")
                        time.sleep(0.5)
                    app.current_strategy = "mean_reversion"
                    print("Switching to Mean Reversion.")
                else:
                    # logic for trend following
                    if app.qty == 0: # logic for opening a position
                        if sma_short[-1] - sma_long[-1] > 0.25:
                            app.qty = 1
                            app.buy(contract, 1)
                            print("Open: Short SMA crossed above Long SMA. Buying 1 contract.")
                            time.sleep(0.5)
                        elif sma_short[-1] - sma_long[-1] < -0.25:
                            app.qty = -1
                            app.sell(contract, 1)
                            print("Open: Short SMA crossed below Long SMA. Selling 1 contract.")
                            time.sleep(0.5)
                    elif app.qty == 1: # logic for closing a long position
                        if sma_short[-1] < sma_long[-1]:
                            app.qty = 0
                            app.sell(contract, 1)
                            print("Close: Short SMA crossed below Long SMA. Selling 1 contract.")
                            time.sleep(0.5)
                    elif app.qty == -1: # logic for closing a short position
                        if sma_short[-1] > sma_long[-1]:
                            app.qty = 0
                            app.buy(contract, 1)
                            print("Close: Short SMA crossed above Long SMA. Buying 1 contract.")
                            time.sleep(0.5)

        # --- Plotting ---
        ax.clear()
        x = range(len(prices))
        ax.plot(x, prices, label=f"Price {app.last_price:.2f}", color="blue")
        ax.plot(x, upper_band, label=f"Upper Band {upper_band[-1]:.2f}", color="purple")
        ax.plot(x, lower_band, label=f"Lower Band {lower_band[-1]:.2f}", color="purple")
        ax.plot(x, [u + app.trade_cost for u in upper_band], label=f"Sell Band {upper_band[-1] + app.trade_cost:.2f}", color="mediumpurple")
        ax.plot(x, [l - app.trade_cost for l in lower_band], label=f"Buy Band {lower_band[-1] - app.trade_cost:.2f}", color="mediumpurple")
        ax.plot(x, sma_mr, label=f"{mr_window} SMA {mu_mr:.2f}", color="cornflowerblue")
        ax.plot(x, sma_short, label=f"{tf_short_window} SMA {sma_short[-1]:.2f}", color="lightcoral")
        ax.plot(x, sma_long, label=f"{tf_long_window} SMA {sma_long[-1]:.2f}", color="lightseagreen")
        ax.fill_between(x, lower_band, upper_band, color="gray", alpha=0.2)
        if app.qty != 0:
            ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
            ax.hlines(app.open_price, 0, len(prices), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
        ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
        ax.plot([],[], " ", label=f"Strategy: {"MR" if app.current_strategy == "mean_reversion" else "TF"}")
        ax.plot([],[], " ", label=f"{mr_window} Std: {std:.2f}")
        ax.plot([],[], " ", label=f"{tf_short_window} Std: {std_short:.2f}")
        ax.legend(ncol=4, loc="upper center")
        ax.set_ylim(min(prices) - 3, max(prices) + 3)
        # -----------------

    ani = animation.FuncAnimation(fig, tick, interval=250, cache_frame_data=False)
    plt.show()

                    


app = IBApi()
app.connect("127.0.0.1", 7497, 0)  # Ensure TWS is running on this port with Paper Trading login
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()
time.sleep(1)  # Sleep to ensure connection is established


# Define the ES continuous futures contract
contract = Contract()
contract.symbol = "ES"  # "MES"
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


mean_reversion(app, 600, 2)
# mean_reversion_with_shutoff_at_high_volatility(app, 300)
# trend_following(app)
# mean_reversion_and_trend_following_switching(app)
# mean_reversion(app, 4 * 60 * 5)
