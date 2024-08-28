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

        self.mean_side = None
        self.qty = 0
        self.open_price = 0
        self.open_dir = None
        self.open_time = 0
        self.max_open_time = float("inf")

        self.leverage = 50  # ES futures leverage
        self.commission = 2.25  # 0.62  # ES commission per contract per side
        self.trade_cost = self.commission * 2 / self.leverage  # for setting buy/sell bands

        self.current_profit = 0
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


def mean_reversion(app):
    fig, ax = plt.subplots()
    ax.set_title("Real-Time Bollinger Bands")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    window = 480  # at 4 ticks per second
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

        if app.qty == 1:
            app.current_profit = (app.last_price - app.open_price) * app.leverage - app.commission
            app.open_time += 1
        elif app.qty == -1:
            app.current_profit = (app.open_price - app.last_price) * app.leverage - app.commission
            app.open_time += 1

        # time limit condition for closing position
        if app.open_time > app.max_open_time:
            if app.qty == 1:
                app.sell(contract, 1)
                app.qty = 0
                app.mean_side = "over"
                print("Time is up. Selling 1 contract.")
                app.open_time = 0
            elif app.qty == -1:
                app.buy(contract, 1)
                app.qty = 0
                app.mean_side = "under"
                print("Time is up. Buying 1 contract.")
                app.open_time = 0

        # sma cross condition for closing position
        if len(prices) == window:
            # use the open price and the sma
            # close when the open price crosses the sma
            # if qty is 1, then we started above the sma and close when its below
            # and vice versa

            if app.qty == 1:
                if app.open_price < mu:
                    app.sell(contract, 1)
                    app.qty = 0
                    app.open_time = 0
                    print("Open price crossed the mean. Selling 1 contract.")

        if len(prices) == window:
            if app.qty == 0:
                if app.last_price < lower_band[-1] - app.trade_cost:
                    app.buy(contract, 1)
                    app.qty = 1
                    print("Price is under the lower band. Buying 1 contract.")
                elif app.last_price > upper_band[-1] + app.trade_cost:
                    app.sell(contract, 1)
                    app.qty = -1
                    print("Price is over the upper band. Selling 1 contract.")
            elif app.qty == 1:
                if app.last_price >= mu:
                    app.sell(contract, 1)
                    app.qty = 0
                    app.open_time = 0
                    print("Price is above the mean. Selling 1 contract.")
            elif app.qty == -1:
                if app.last_price <= mu:
                    app.buy(contract, 1)
                    app.qty = 0
                    app.open_time = 0
                    print("Price is below the mean. Buying 1 contract.")

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
            ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")  # Invisible line with profit label
            ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
        ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")  # Invisible line with total profit label
        ax.fill_between(x, lower_band, upper_band, color="gray", alpha=0.2)
        ax.set_ylim(lower_band[-1] - app.trade_cost - 2, upper_band[-1] + app.trade_cost + 2)
        ax.legend(ncol=3, loc="upper center")
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


mean_reversion(app)
