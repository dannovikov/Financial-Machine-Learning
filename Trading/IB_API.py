from ibapi.client import EClient
from ibapi.order_state import OrderState
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from collections import deque
import time


class IBApi(EWrapper, EClient):
    def __init__(self, contract):
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

        # Current price and volume
        self.bid_price = 0
        self.bid_size = 0
        self.ask_price = 0
        self.ask_size = 0
        self.last_price = 0
        self.last_size = 0

        # Contract details
        self.contract = contract
        self.leverage = 50  # 5  # ES futures leverage
        self.commission = 2.25  # 0.62  # ES commission per contract per side
        self.trade_cost = self.commission * 2 / self.leverage  # for setting buy/sell bands

        # Position details
        self.qty = 0
        self.prev_qty = 0
        self.open_price = 0
        self.open_dir = None
        self.open_time = 0
        self._last_permid = None
        self.order_id = 0  # To track orders

        # Profit details
        self.current_profit = 0
        self.guarenteed_profit = 0
        self.total_profit = 0

        self.stop_loss = -2 * self.leverage  # stop loss in dollars
        self.max_open_time = 300  # max number of ticks to hold a position

        self.last_three_open_times = deque([1e9, 1e9, 1e9], maxlen=3)

        self.cooldown = 0
        self.strategy = "trend_following"

        # measure how often our order_price != fill_price
        self.order_price = 0
        self.slipped_trades = 0
        self.total_trades = 0

    def nextValidId(self, orderId):
        super().nextValidId(orderId)
        self.order_id = orderId
        print(f"Next valid order ID: {orderId}")

    def set_qty(self, new_qty):
        self.prev_qty = self.qty  # Track current position before change
        self.qty = new_qty  # Update to reflect new position

    def orderStatus(
        self,
        orderId,
        status,
        filled,
        remaining,
        avgFillPrice,
        permId,
        parentId,
        lastFillPrice,
        clientId,
        whyHeld,
        mktCapPrice,
    ):
        if status == "Filled":
            if permId == self._last_permid:
                return
            self._last_permid = permId
            if self.prev_qty == -1 and self.qty == 1:
                # Reversed from short to long
                profit = (self.open_price - avgFillPrice) * self.leverage - self.commission * 2
                self.total_profit += profit
                self.open_price = avgFillPrice  # Set new open price
                print(f"Reversed from short to long. Profit: {profit:.2f}, Total PnL: {self.total_profit:.2f}")

            elif self.prev_qty == 1 and self.qty == -1:
                # Reversed from long to short
                profit = (avgFillPrice - self.open_price) * self.leverage - self.commission * 2
                self.total_profit += profit
                self.open_price = avgFillPrice  # Set new open price
                print(f"Reversed from long to short. Profit: {profit:.2f}, Total PnL: {self.total_profit:.2f}")

            elif self.qty == 1 and self.prev_qty == 0:
                # Opened a new long position
                self.open_price = avgFillPrice
                if self.order_price != avgFillPrice:
                    self.slipped_trades += 1
                print(f"Opened long position at {avgFillPrice}")

            elif self.qty == -1 and self.prev_qty == 0:
                # Opened a new short position
                self.open_price = avgFillPrice
                if self.order_price != avgFillPrice:
                    self.slipped_trades += 1
                print(f"Opened short position at {avgFillPrice}")

            elif self.qty == 0 and self.prev_qty == 1:
                # Closed long position
                profit = (avgFillPrice - self.open_price) * self.leverage - self.commission * 2
                self.total_profit += profit
                self.open_price = 0  # Reset since position is closed
                if self.order_price != avgFillPrice:
                    self.slipped_trades += 1
                print(f"Closed long position. Profit: {profit:.2f}, Total PnL: {self.total_profit:.2f}")

            elif self.qty == 0 and self.prev_qty == -1:
                # Closed short position
                profit = (self.open_price - avgFillPrice) * self.leverage - self.commission * 2
                self.total_profit += profit
                self.open_price = 0  # Reset since position is closed
                if self.order_price != avgFillPrice:
                    self.slipped_trades += 1
                print(f"Closed short position. Profit: {profit:.2f}, Total PnL: {self.total_profit:.2f}")

            self.total_trades += 1

    def pnl(self, reqId: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float):
        print(f"Daily PnL: {dailyPnL}, Unrealized PnL: {unrealizedPnL}, Realized PnL: {realizedPnL}")

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
        self.reqMktData(
            reqId=1, contract=contract, genericTickList="", snapshot=False, regulatorySnapshot=False, mktDataOptions=[]
        )

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
        order.orderType = "LMT" if price else "MKT"
        order.totalQuantity = qty
        if price:
            order.lmtPrice = price
        return order

    def close_position(self):
        if self.qty == 1:
            self.set_qty(0)  # Update qty to 0 to close the long position
            self.sell(self.contract, 1)
            print("Closing long position. Selling 1 contract.")
        elif self.qty == -1:
            self.set_qty(0)  # Update qty to 0 to close the short position
            self.buy(self.contract, 1)
            print("Closing short position. Buying 1 contract.")
        self.reset_open_time()

    def close_long_position(self):
        assert self.qty == 1, "Close long position called when not long"
        self.set_qty(0)
        self.sell(self.contract, 1)
        self.reset_open_time()
        print("Closing long position. Selling 1 contract.")

    def close_short_position(self):
        assert self.qty == -1, "Close short position called when not short"
        self.set_qty(0)
        self.buy(self.contract, 1)
        self.reset_open_time()
        print("Closing short position. Buying 1 contract.")

    def open_long_position(self):
        assert self.qty == 0, "Open long position called when not flat"
        self.set_qty(1)  # Update qty to 1 for a long position
        self.buy(self.contract, 1)
        print("Opening long position. Buying 1 contract.")

    def open_short_position(self):
        assert self.qty == 0, "Open short position called when not flat"
        self.set_qty(-1)  # Update qty to -1 for a short position
        self.sell(self.contract, 1)
        print("Opening short position. Selling 1 contract.")

    def switch_side(self):
        if self.qty == 1:
            self.set_qty(-1)  # Update qty to -1 to reverse to a short position
            self.sell(self.contract, 2)  # Sell 2 contracts: 1 to close long, 1 to open short
            print("Switching to short position. Selling 2 contracts.")
        elif self.qty == -1:
            self.set_qty(1)  # Update qty to 1 to reverse to a long position
            self.buy(self.contract, 2)  # Buy 2 contracts: 1 to close short, 1 to open long
            print("Switching to long position. Buying 2 contracts.")
        self.reset_open_time()

    def wait_for_prices(self):
        while not self.last_price:
            wait_time = 0
            print(f"Waiting for price data... {wait_time} s\r", end="")
            time.sleep(1)
            wait_time += 1
        print(" " * 50, end="\r")

    def update_current_profit(self):
        if self.qty == 1:
            self.current_profit = (self.last_price - self.open_price) * self.leverage - self.commission * 2
        elif self.qty == -1:
            self.current_profit = (self.open_price - self.last_price) * self.leverage - self.commission * 2
        else:
            self.current_profit = 0

    def reset_open_time(self):
        self.last_three_open_times.append(self.open_time)
        self.open_time = 0
