from IB_API import IBApi
from ibapi.contract import Contract
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import threading
import time


def compute_rsi(prices, window=150):
    if len(prices) < window:
        return -1
    delta = np.diff(list(prices)[-window:])
    gain = delta[delta > 0]
    loss = -delta[delta < 0]
    avg_gain = np.sum(gain)
    avg_loss = np.sum(loss)
    if avg_loss != 0:
        rs = avg_gain / avg_loss
    else:
        rs = 0
    rsi = 100 - (100 / (1 + rs))
    return rsi


def mean_reversion(app, boll_window=600, rsi_window=200, width=2):
    history = 2400
    cooldown = 100

    prices = deque(maxlen=boll_window)
    price_history = deque(maxlen=history)
    upper_bands = deque(maxlen=history)
    lower_bands = deque(maxlen=history)
    sma = deque(maxlen=history)

    def tick(ax, plotting=True):
        app.update_current_profit()
        prices.append(app.last_price)
        price_history.append(app.last_price)

        mu = np.mean(prices)
        std = np.std(prices)
        upper_band = mu + width * std
        lower_band = mu - width * std
        rsi = compute_rsi(prices, rsi_window)

        sma.append(mu)
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)

        if app.qty != 0:
            app.open_time += 1
            if app.current_profit < app.stop_loss:
                print("Stop loss hit.")
                app.close_position()
                app.cooldown = cooldown
            if app.open_time > app.max_open_time:
                print("Times up.")
                app.close_position()
                app.cooldown = cooldown

        if len(prices) == boll_window and app.cooldown == 0:
            if app.qty == 0:
                if app.last_price < lower_band:
                    app.open_long_position()
                elif app.last_price > upper_band:
                    app.open_short_position()
            elif app.qty == 1:
                if app.last_price >= mu:
                    app.close_long_position()
            elif app.qty == -1:
                if app.last_price <= mu:
                    app.close_short_position()

        if app.cooldown > 0:
            app.cooldown -= 1

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.plot([], [], " ", label=f"std: {std:.2f}")
            ax.plot([], [], " ", label=f"RSI: {rsi:.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
            if app.cooldown > 0:
                ax.plot([], [], " ", label=f"Cooldown {app.cooldown}/{cooldown}")
            ax.fill_between(x_axis, lower_bands, upper_bands, color="gray", alpha=0.2)
            ax.set_ylim(lower_bands[-1] - app.trade_cost - 2, upper_bands[-1] + app.trade_cost + 2)
            ax.legend(ncol=3, loc="upper center")

    fig, ax = plt.subplots()
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def trend_following(app, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015, entry_threshold=0.50, exit_cooldown=60):
    history = 1200
    open_cooldown = 4

    price_history = deque(maxlen=history)
    slow_emas = deque(maxlen=history)
    fast_emas = deque(maxlen=history)
    take_profit_emas = deque(maxlen=history)
    slow_emas.append(app.last_price)
    fast_emas.append(app.last_price)
    take_profit_emas.append(app.last_price)
    price_history.append(app.last_price)

    def tick(ax, plotting=True):
        app.update_current_profit()
        price_history.append(app.last_price)

        slow_ema = slow_alpha * app.last_price + (1 - slow_alpha) * slow_emas[-1]
        fast_ema = fast_alpha * app.last_price + (1 - fast_alpha) * fast_emas[-1]
        take_profit_ema = take_profit_alpha * app.last_price + (1 - take_profit_alpha) * take_profit_emas[-1]
        slow_emas.append(slow_ema)
        fast_emas.append(fast_ema)
        take_profit_emas.append(take_profit_ema)

        if app.qty != 0:
            app.open_time += 1
        #     if app.current_profit < app.stop_loss:
        #         print("Stop loss hit.")
        #         app.close_position()
        #         app.cooldown = cooldown
        #     if app.open_time > app.max_open_time:
        #         print("Times up.")
        #         app.close_position()
        #         app.cooldown = cooldown

        if len(price_history) > 150 and app.cooldown == 0:
            if app.qty == 0:
                if fast_ema > slow_ema + entry_threshold:
                    app.open_long_position()
                    app.cooldown = open_cooldown
                elif fast_ema < slow_ema - entry_threshold:
                    app.open_short_position()
                    app.cooldown = open_cooldown
            elif app.qty == 1:
                # if fast_ema < slow_ema:
                if app.last_price < take_profit_ema:
                    app.close_long_position()
                    app.cooldown = exit_cooldown
            elif app.qty == -1:
                # if fast_ema > slow_ema:
                if app.last_price > take_profit_ema:
                    app.close_short_position()
                    app.cooldown = exit_cooldown

        if app.cooldown > 0:
            app.cooldown -= 1

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x_axis, slow_emas, label=f"{slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
            ax.plot(x_axis, fast_emas, label=f"{fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
            ax.plot(x_axis, take_profit_emas, label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}", color="lightcoral")
            ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
            if app.cooldown > 0:
                ax.plot([], [], " ", label=f"Cooldown {app.cooldown}")
            ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
            ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()
    # set figsize
    fig.set_size_inches(2, 6)
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def TF_MR_switch_on_rapid_close(app, boll_window=300, rsi_window=300, width=2, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015, entry_threshold=0.50, exit_cooldown=60):
    history = 1200
    open_cooldown = 4
    cooldown = 100

    prices = deque(maxlen=boll_window)
    price_history = deque(maxlen=history)
    upper_bands = deque(maxlen=history)
    lower_bands = deque(maxlen=history)
    sma = deque(maxlen=history)
    slow_emas = deque(maxlen=history)
    fast_emas = deque(maxlen=history)
    take_profit_emas = deque(maxlen=history)

    slow_emas.append(app.last_price)
    fast_emas.append(app.last_price)
    take_profit_emas.append(app.last_price)
    price_history.append(app.last_price)
    sma.append(app.last_price)
    upper_bands.append(app.last_price)
    lower_bands.append(app.last_price)

    def tick(ax, plotting=True):
        app.update_current_profit()
        prices.append(app.last_price)
        price_history.append(app.last_price)

        mu = np.mean(prices)
        std = np.std(prices)
        upper_band = mu + width * std
        lower_band = mu - width * std
        rsi = compute_rsi(prices, rsi_window)

        sma.append(mu)
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)

        slow_ema = slow_alpha * app.last_price + (1 - slow_alpha) * slow_emas[-1]
        fast_ema = fast_alpha * app.last_price + (1 - fast_alpha) * fast_emas[-1]
        take_profit_ema = take_profit_alpha * app.last_price + (1 - take_profit_alpha) * take_profit_emas[-1]
        slow_emas.append(slow_ema)
        fast_emas.append(fast_ema)
        take_profit_emas.append(take_profit_ema)

        if app.qty != 0:
            app.open_time += 1

        if app.cooldown > 0:
            app.cooldown -= 1

        if app.strategy == "mean_reversion":
            if len(prices) == boll_window and app.cooldown == 0:
                if app.qty == 0:
                    if app.last_price < lower_band:
                        app.open_long_position()
                    elif app.last_price > upper_band:
                        app.open_short_position()
                elif app.qty == 1:
                    if app.last_price >= mu:
                        app.close_long_position()
                elif app.qty == -1:
                    if app.last_price <= mu:
                        app.close_short_position()
        elif app.strategy == "trend_following":
            if len(price_history) > boll_window and app.cooldown == 0:
                if app.qty == 0:
                    if fast_ema > slow_ema + entry_threshold:
                        app.open_long_position()
                        app.cooldown = open_cooldown
                    elif fast_ema < slow_ema - entry_threshold:
                        app.open_short_position()
                        app.cooldown = open_cooldown
                elif app.qty == 1:
                    if app.last_price < take_profit_ema:
                        app.close_long_position()
                        app.cooldown = exit_cooldown
                elif app.qty == -1:
                    if app.last_price > take_profit_ema:
                        app.close_short_position()
                        app.cooldown = exit_cooldown

        if app.strategy == "trend_following" and np.mean(app.last_three_open_times) < 60:
            app.strategy = "mean_reversion"
            app.close_position()
            print("Switching to mean reversion strategy.")

        if plotting:
            if app.strategy == "mean_reversion":
                x_axis = range(len(price_history))
                ax.clear()
                ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
                ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
                ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")
                ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
                ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
                ax.plot([], [], " ", label=f"std: {std:.2f}")
                ax.plot([], [], " ", label=f"RSI: {rsi:.2f}")
                ax.plot([], [], " ", label="Strat: MR")
                if app.qty != 0:
                    ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                    ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                    ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
                if app.cooldown > 0:
                    ax.plot([], [], " ", label=f"Cooldown {app.cooldown}/{cooldown}")
                ax.fill_between(x_axis, lower_bands, upper_bands, color="gray", alpha=0.2)
                ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
                ax.legend(ncol=1, loc="upper left")

            elif app.strategy == "trend_following":
                x_axis = range(len(price_history))
                ax.clear()
                ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
                ax.plot(x_axis, slow_emas, label=f"{slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
                ax.plot(x_axis, fast_emas, label=f"{fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
                ax.plot(x_axis, take_profit_emas, label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}", color="lightcoral")
                ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
                ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
                ax.plot([], [], " ", label=f"{app.last_three_open_times}")
                ax.plot([], [], " ", label="Strat: TF")
                if app.qty != 0:
                    ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                    ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                    ax.hlines(app.open_price, 0, len(price_history), label=f"Open Price {app.open_price:.2f}", color="forestgreen" if app.qty == 1 else "firebrick")
                if app.cooldown > 0:
                    ax.plot([], [], " ", label=f"Cooldown {app.cooldown}")
                ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
                ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


# Define the ES continuous futures contract
contract = Contract()
contract.symbol = "ES"  # "MES"
contract.secType = "FUT"
contract.exchange = "CME"
contract.currency = "USD"
contract.lastTradeDateOrContractMonth = "202409"

app = IBApi(contract)
app.connect("127.0.0.1", 7497, 0)  # Ensure TWS is running on this port with Paper Trading login
api_thread = threading.Thread(target=app.run, daemon=True)
api_thread.start()
time.sleep(1)  # Sleep to ensure connection is established


# Subscribe to live market data for ES
print("Subscribing to market data")
app.subscribe_to_market_data(contract)

app.wait_for_prices()

# app.open_long_position()
# time.sleep(1)
# app.switch_side()
# time.sleep(1)
# app.switch_side()
# time.sleep(1)
# app.close_position()
# time.sleep(1)


# mean_reversion(app, 600, 50, 2)
# trend_following(app)
TF_MR_switch_on_rapid_close(app)
