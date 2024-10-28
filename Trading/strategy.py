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
    bid_history = deque(maxlen=history)
    ask_history = deque(maxlen=history)
    upper_bands = deque(maxlen=history)
    lower_bands = deque(maxlen=history)
    sma = deque(maxlen=history)

    def tick(ax, plotting=True):
        app.update_current_profit()
        prices.append(app.last_price)
        price_history.append(app.last_price)
        bid_history.append(app.bid_price)
        ask_history.append(app.ask_price)

        mu = np.mean(prices)
        std = np.std(prices)
        upper_band = mu + width * std + app.trade_cost
        lower_band = mu - width * std - app.trade_cost
        # rsi = compute_rsi(prices, rsi_window)

        sma.append(mu)
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)

        if app.qty != 0:
            app.open_time += 1

        if len(prices) == boll_window and app.cooldown == 0:
            if app.qty == 0:
                # if app.last_price < lower_band:
                if app.ask_price < lower_band:
                    app.order_price = app.ask_price
                    app.open_long_position()
                # elif app.last_price > upper_band:
                elif app.bid_price > upper_band:
                    app.order_price = app.bid_price
                    app.open_short_position()
            elif app.qty == 1:
                if app.last_price >= mu:
                    app.order_price = app.last_price
                    app.close_long_position()
            elif app.qty == -1:
                if app.last_price <= mu:
                    app.order_price = app.last_price
                    app.close_short_position()

        if app.cooldown > 0:
            app.cooldown -= 1

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            # ax.plot(x_axis, bid_history, label=f"Bid {app.bid_price:.2f}", color="lightblue")
            # ax.plot(x_axis, ask_history, label=f"Ask {app.ask_price:.2f}", color="lightblue")

            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")

            ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.plot([], [], " ", label=f"std: {std:.2f}")
            # ax.plot([], [], " ", label=f"RSI: {rsi:.2f}")
            ax.plot([], [], " ", label=f"Slippage rate: {app.slipped_trades/(app.total_trades+1e-8):.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
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


def mean_reversion_at_the_bands(app, boll_window=600, rsi_window=200, width=2):
    history = 2400
    cooldown = 100

    prices = deque(maxlen=boll_window)
    price_history = deque(maxlen=history)
    bid_history = deque(maxlen=history)
    ask_history = deque(maxlen=history)
    upper_bands = deque(maxlen=history)
    lower_bands = deque(maxlen=history)
    sma = deque(maxlen=history)

    def tick(ax, plotting=True):
        app.update_current_profit()
        prices.append(app.last_price)
        price_history.append(app.last_price)
        bid_history.append(app.bid_price)
        ask_history.append(app.ask_price)

        mu = np.mean(prices)
        std = np.std(prices)
        upper_band = mu + width * std + app.trade_cost
        lower_band = mu - width * std - app.trade_cost
        # rsi = compute_rsi(prices, rsi_window)

        sma.append(mu)
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)

        if app.qty != 0:
            app.open_time += 1

        if len(prices) == boll_window and app.cooldown == 0:
            if app.qty == 0:
                # if app.last_price < lower_band:
                if app.last_price < lower_band:
                    app.order_price = app.ask_price
                    app.open_long_position()
                # elif app.last_price > upper_band:
                elif app.bid_price > upper_band:
                    app.last_price = app.bid_price
                    app.open_short_position()
            elif app.qty == 1:
                if app.last_price >= upper_band:
                    app.order_price = app.last_price
                    app.close_long_position()
            elif app.qty == -1:
                if app.last_price <= lower_band:
                    app.order_price = app.last_price
                    app.close_short_position()

        if app.cooldown > 0:
            app.cooldown -= 1

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            # ax.plot(x_axis, bid_history, label=f"Bid {app.bid_price:.2f}", color="lightblue")
            # ax.plot(x_axis, ask_history, label=f"Ask {app.ask_price:.2f}", color="lightblue")

            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")

            ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.plot([], [], " ", label=f"std: {std:.2f}")
            # ax.plot([], [], " ", label=f"RSI: {rsi:.2f}")
            ax.plot([], [], " ", label=f"Slippage rate: {app.slipped_trades/(app.total_trades+1e-8):.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
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


def mean_reversion_with_ema(app, boll_window=600, width=2, alpha=0.01):
    history = 1200
    cooldown = 100

    prices = deque(maxlen=boll_window)
    price_history = deque(maxlen=history)
    bid_history = deque(maxlen=history)
    ask_history = deque(maxlen=history)
    upper_bands = deque(maxlen=history)
    lower_bands = deque(maxlen=history)
    emas = deque(maxlen=history)
    mu = app.last_price

    def tick(ax, plotting=True):
        nonlocal mu
        app.update_current_profit()
        prices.append(app.last_price)
        price_history.append(app.last_price)
        bid_history.append(app.bid_price)
        ask_history.append(app.ask_price)

        # mu = np.mean(prices)
        mu = alpha * app.last_price + (1 - alpha) * mu
        std = np.std(prices)
        upper_band = mu + width * std + app.trade_cost
        lower_band = mu - width * std - app.trade_cost
        # rsi = compute_rsi(prices, rsi_window)

        emas.append(mu)
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)

        if app.qty != 0:
            app.open_time += 1

        if len(prices) == boll_window and app.cooldown == 0:
            if app.qty == 0:
                # if app.last_price < lower_band:
                if app.ask_price < lower_band:
                    app.order_price = app.ask_price
                    app.open_long_position()
                # elif app.last_price > upper_band:
                elif app.bid_price > upper_band:
                    app.order_price = app.bid_price
                    app.open_short_position()
            elif app.qty == 1:
                if app.last_price >= mu:
                    app.order_price = app.last_price
                    app.close_long_position()
            elif app.qty == -1:
                if app.last_price <= mu:
                    app.order_price = app.last_price
                    app.close_short_position()

        if app.cooldown > 0:
            app.cooldown -= 1

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            # ax.plot(x_axis, bid_history, label=f"Bid {app.bid_price:.2f}", color="lightblue")
            # ax.plot(x_axis, ask_history, label=f"Ask {app.ask_price:.2f}", color="lightblue")

            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")

            ax.plot(x_axis, emas, label=f"EMA {mu:.2f}", color="cornflowerblue")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.plot([], [], " ", label=f"std: {std:.2f}")
            # ax.plot([], [], " ", label=f"RSI: {rsi:.2f}")
            ax.plot([], [], " ", label=f"Slippage rate: {app.slipped_trades/(app.total_trades+1e-8):.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
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


def trend_following_with_cooldown(
    app, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015, entry_threshold=0.50, exit_cooldown=60
):
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
            ax.plot(
                x_axis,
                take_profit_emas,
                label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}",
                color="lightcoral",
            )
            ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
            if app.cooldown > 0:
                ax.plot([], [], " ", label=f"Cooldown {app.cooldown}")
            ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
            ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()
    # set figsize
    # fig.set_size_inches(2, 6)
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def trend_following(app, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015, entry_macd=0.50):
    history = 1200
    price_history = deque(maxlen=history)
    slow_emas = deque(maxlen=history)
    fast_emas = deque(maxlen=history)
    take_profit_emas = deque(maxlen=history)
    slow_emas.append(app.last_price)
    fast_emas.append(app.last_price)
    take_profit_emas.append(app.last_price)
    price_history.append(app.last_price)

    _exit_cooldown = 15 * 4  # 15 seconds at 4 ticks per second
    _update_slow_ema_on_trend_stop = False

    def tick(ax, plotting=True):
        nonlocal _update_slow_ema_on_trend_stop
        app.update_current_profit()

        slow_ema = slow_alpha * app.last_price + (1 - slow_alpha) * slow_emas[-1]
        fast_ema = fast_alpha * app.last_price + (1 - fast_alpha) * fast_emas[-1]
        take_profit_ema = take_profit_alpha * app.last_price + (1 - take_profit_alpha) * take_profit_emas[-1]
        if _update_slow_ema_on_trend_stop:
            slow_ema = fast_ema
            _update_slow_ema_on_trend_stop = False

        price_history.append(app.last_price)
        slow_emas.append(slow_ema)
        fast_emas.append(fast_ema)
        take_profit_emas.append(take_profit_ema)

        if app.cooldown > 0:
            app.cooldown -= 1

        if len(price_history) > 150 and app.cooldown == 0:
            if app.qty == 0:
                if fast_ema > slow_ema + entry_macd:
                    app.open_long_position()
                elif fast_ema < slow_ema - entry_macd:
                    app.open_short_position()
            elif app.qty == 1:
                if fast_ema < take_profit_ema:
                    app.close_long_position()
                    app.cooldown = _exit_cooldown
                    _update_slow_ema_on_trend_stop = True
            elif app.qty == -1:
                if fast_ema > take_profit_ema:
                    app.close_short_position()
                    app.cooldown = _exit_cooldown
                    _update_slow_ema_on_trend_stop = True

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x_axis, slow_emas, label=f"{slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
            ax.plot(x_axis, fast_emas, label=f"{fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
            if take_profit_alpha != slow_alpha:
                ax.plot(
                    x_axis,
                    take_profit_emas,
                    label=f"{take_profit_alpha:.4f} Take Profit EMA {take_profit_ema:.2f}",
                    color="lightcoral",
                )
            ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
            ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
            ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()

    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def crypto_trend_following_with_no_naked_shorts(
    app, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015, entry_macd=0.50
):
    history = 1200
    price_history = deque(maxlen=history)
    slow_emas = deque(maxlen=history)
    fast_emas = deque(maxlen=history)
    take_profit_emas = deque(maxlen=history)
    slow_emas.append(app.last_price)
    fast_emas.append(app.last_price)
    take_profit_emas.append(app.last_price)
    price_history.append(app.last_price)

    _exit_cooldown = 15 * 4  # 15 seconds at 4 ticks per second

    # start by buying 100 dollars worth of the crypto
    app.open_long_position()

    def tick(ax, plotting=True):
        app.update_current_profit()

        slow_ema = slow_alpha * app.last_price + (1 - slow_alpha) * slow_emas[-1]
        fast_ema = fast_alpha * app.last_price + (1 - fast_alpha) * fast_emas[-1]
        take_profit_ema = take_profit_alpha * app.last_price + (1 - take_profit_alpha) * take_profit_emas[-1]

        price_history.append(app.last_price)
        slow_emas.append(slow_ema)
        fast_emas.append(fast_ema)
        take_profit_emas.append(take_profit_ema)

        if app.cooldown > 0:
            app.cooldown -= 1

        if len(price_history) > 150 and app.cooldown == 0:
            if app.qty == 0:
                if fast_ema > slow_ema + entry_macd:
                    app.open_long_position()
                elif fast_ema < slow_ema - entry_macd:
                    app.open_short_position()
            elif app.qty == 1:
                if fast_ema < take_profit_ema:
                    app.close_long_position()
                    app.cooldown = _exit_cooldown
            elif app.qty == -1:
                if fast_ema > take_profit_ema:
                    app.close_short_position()
                    app.cooldown = _exit_cooldown

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x_axis, slow_emas, label=f"{slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
            ax.plot(x_axis, fast_emas, label=f"{fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
            ax.plot(
                x_axis,
                take_profit_emas,
                label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}",
                color="lightcoral",
            )
            ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
            ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
            ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()

    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def TF_MR_switch_on_rapid_close(
    app,
    boll_window=300,
    rsi_window=300,
    width=2,
    slow_alpha=0.01,
    fast_alpha=0.1,
    take_profit_alpha=0.015,
    entry_threshold=0.50,
    exit_cooldown=60,
):
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
                    ax.hlines(
                        app.open_price,
                        0,
                        len(price_history),
                        label=f"Open Price {app.open_price:.2f}",
                        color="forestgreen" if app.qty == 1 else "firebrick",
                    )
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
                ax.plot(
                    x_axis,
                    take_profit_emas,
                    label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}",
                    color="lightcoral",
                )
                ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
                ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
                ax.plot([], [], " ", label=f"{app.last_three_open_times}")
                ax.plot([], [], " ", label="Strat: TF")
                if app.qty != 0:
                    ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                    ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                    ax.hlines(
                        app.open_price,
                        0,
                        len(price_history),
                        label=f"Open Price {app.open_price:.2f}",
                        color="forestgreen" if app.qty == 1 else "firebrick",
                    )
                if app.cooldown > 0:
                    ax.plot([], [], " ", label=f"Cooldown {app.cooldown}")
                ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
                ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def trend_with_bollinger_breakout(app, boll_window=150, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015):
    history = 1200
    open_cooldown = 4
    cooldown = 15

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
        upper_band = mu + 2.1 * std
        lower_band = mu - 2.1 * std

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

        if app.qty == 0:
            if fast_ema > slow_ema and fast_ema > upper_band:
                app.open_long_position()
                app.cooldown = open_cooldown
            elif fast_ema < slow_ema and fast_ema < lower_band:
                app.open_short_position()
                app.cooldown = open_cooldown
        elif app.qty == 1:
            if app.last_price < take_profit_ema:
                app.close_long_position()
                app.cooldown = cooldown
        elif app.qty == -1:
            if app.last_price > take_profit_ema:
                app.close_short_position()
                app.cooldown = cooldown

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
            ax.plot(x_axis, slow_emas, label=f"{slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
            ax.plot(x_axis, fast_emas, label=f"{fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
            ax.plot(
                x_axis,
                take_profit_emas,
                label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}",
                color="lightcoral",
            )
            ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.plot([], [], " ", label="Strat: TF with Bollinger")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
            if app.cooldown > 0:
                ax.plot([], [], " ", label=f"Cooldown {app.cooldown}/{cooldown}")
            ax.fill_between(x_axis, lower_bands, upper_bands, color="gray", alpha=0.2)
            ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
            ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def mean_rev_within_boll_and_trend_on_breakout(
    app, boll_window=500, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.015
):
    history = 1200
    open_cooldown = 4
    cooldown = 15

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
        upper_band = mu + 2 * std
        lower_band = mu - 2 * std

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

        if app.qty == 0 and len(price_history) > boll_window:
            if fast_ema > slow_ema:
                if fast_ema > upper_band:  # this will never execute
                    app.strategy = "trend_following"
                    app.open_long_position()
                    app.cooldown = open_cooldown
                elif app.last_price > upper_band:
                    app.strategy = "mean_reversion"
                    app.open_short_position()
                    app.cooldown = open_cooldown
            elif fast_ema < slow_ema:
                if fast_ema < lower_band:
                    app.strategy = "trend_following"
                    app.open_short_position()
                    app.cooldown = open_cooldown
                elif app.last_price < lower_band:
                    app.strategy = "mean_reversion"
                    app.open_long_position()
                    app.cooldown = open_cooldown
        elif app.qty == 1:
            if app.strategy == "trend_following":
                if app.last_price < take_profit_ema:
                    app.close_long_position()
                    app.cooldown = cooldown
            elif app.strategy == "mean_reversion":
                if app.last_price > mu:
                    app.close_long_position()
                    app.cooldown = cooldown
                elif fast_ema < lower_band - 0.50:
                    app.switch_side()
                    app.cooldown = cooldown
        elif app.qty == -1:
            if app.strategy == "trend_following":
                if app.last_price > take_profit_ema:
                    app.close_short_position()
                    app.cooldown = cooldown
            elif app.strategy == "mean_reversion":
                if app.last_price < mu:
                    app.close_short_position()
                    app.cooldown = cooldown
                elif fast_ema > upper_band + 0.50:
                    app.switch_side()
                    app.cooldown = cooldown

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            ax.plot(x_axis, price_history, label=f"Price {app.last_price:.2f}", color="blue")
            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")
            ax.plot(x_axis, sma, label=f"SMA {mu:.2f}", color="cornflowerblue")
            ax.plot(x_axis, slow_emas, label=f"{slow_alpha} Slow EMA {slow_ema:.2f}", color="cornflowerblue")
            ax.plot(x_axis, fast_emas, label=f"{fast_alpha} Fast EMA {fast_ema:.2f}", color="navajowhite")
            ax.plot(
                x_axis,
                take_profit_emas,
                label=f"{take_profit_alpha} Take Profit EMA {take_profit_ema:.2f}",
                color="lightcoral",
            )
            ax.plot([], [], " ", label=f"MACD {fast_ema - slow_ema:.2f}")
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            ax.plot([], [], " ", label=f"Strat: {app.strategy}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
            if app.cooldown > 0:
                ax.plot([], [], " ", label=f"Cooldown {app.cooldown}/{cooldown}")
            ax.fill_between(x_axis, lower_bands, upper_bands, color="gray", alpha=0.2)
            ax.set_ylim(min(price_history) - 2, max(price_history) + 2)
            ax.legend(ncol=1, loc="upper left")

    fig, ax = plt.subplots()
    while True:
        tick(ax)
        plt.pause(0.25)
    plt.show()


def mean_reversion_with_ema_deviation_sensitive(app, ema_alpha_init, boll_window=300):
    # This version of the strategy will feature an ema alpha that is sensitive to the deviation of the price from the ema
    # a large move away from the ema will cause the alpha to decrease, and vice versa
    # the idea being that large moves are mean reversions over a larger period of time
    # if the alpha is too sensitive, it will close out a loss before it can recover

    history = 1200
    width = 2

    prices = deque(maxlen=boll_window)
    price_history = deque(maxlen=history)
    bid_history = deque(maxlen=history)
    ask_history = deque(maxlen=history)
    upper_bands = deque(maxlen=history)
    lower_bands = deque(maxlen=history)
    emas = deque(maxlen=history)
    emas.append(app.last_price)
    prices.append(app.last_price)
    price_history.append(app.last_price)
    bid_history.append(app.bid_price)
    ask_history.append(app.ask_price)
    upper_bands.append(app.last_price)
    lower_bands.append(app.last_price)

    ema = app.last_price
    alpha = ema_alpha_init
    alpha_max = ema_alpha_init
    alpha_min = ema_alpha_init / 10

    def tick(ax, plotting=True):
        nonlocal alpha
        app.update_current_profit()
        ema = alpha * app.last_price + (1 - alpha) * emas[-1]
        std = np.std(prices) + 1e-6
        upper_band = ema + width * std + app.trade_cost
        lower_band = ema - width * std - app.trade_cost

        prices.append(app.last_price)
        price_history.append(app.last_price)
        bid_history.append(app.bid_price)
        ask_history.append(app.ask_price)
        emas.append(ema)
        upper_bands.append(upper_band)
        lower_bands.append(lower_band)

        # Update alpha based on deviation from ema
        deviation = abs(app.last_price - ema)
        # if deviation <= 2 * std:
        #     alpha = ema_alpha_init
        # else:
        # alpha = alpha_min + (alpha_max - alpha_min) * (1 - np.exp(-deviation / std))
        # alpha should be negatively proportional to deviation, such that when we deviate a lot, we are LESS sensitive
        k = 2  # k is a sensitivity parameter, the higher the K, the more rigid the alpha is
        alpha = alpha_max / (1 + k * (deviation / std))

        # print(f"\n\n{alpha=:.4f}, {deviation=:.2f}, {std=:.2f}, {ema=:.2f},\n\n")
        if app.qty != 0:
            app.open_time += 1

        if len(prices) == boll_window:
            if app.qty == 0:
                if app.ask_price < lower_band:
                    app.open_long_position()
                elif app.bid_price > upper_band:
                    app.open_short_position()
            elif app.qty == 1:
                if app.last_price >= ema:
                    app.close_long_position()
            elif app.qty == -1:
                if app.last_price <= ema:
                    app.close_short_position()

        if plotting:
            x_axis = range(len(price_history))
            ax.clear()
            # Bid History
            ax.plot(x_axis, bid_history, label=f"Bid {app.bid_price:.2f}", color="blue")
            # Ask History
            ax.plot(x_axis, ask_history, label=f"Ask {app.ask_price:.2f}", color="darkblue")
            # Upper Band
            ax.plot(x_axis, upper_bands, label=f"Upper Band {upper_bands[-1]:.2f}", color="purple")
            # Lower Band
            ax.plot(x_axis, lower_bands, label=f"Lower Band {lower_bands[-1]:.2f}", color="purple")
            # EMA
            ax.plot(x_axis, emas, label=f"EMA {ema:.2f}", color="cornflowerblue")
            # PnL
            ax.plot([], [], " ", label=f"PnL {app.total_profit:.2f}")
            # Std
            ax.plot([], [], " ", label=f"std: {std:.2f}")
            # Alpha
            ax.plot([], [], " ", label=f"Alpha: {alpha:.4f}")
            if app.qty != 0:
                ax.plot([], [], " ", label=f"Current Profit {app.current_profit:.2f}")
                ax.plot([], [], " ", label=f"Open Time {app.open_time}/{app.max_open_time}")
                ax.hlines(
                    app.open_price,
                    0,
                    len(price_history),
                    label=f"Open Price {app.open_price:.2f}",
                    color="forestgreen" if app.qty == 1 else "firebrick",
                )
            ax.fill_between(x_axis, lower_bands, upper_bands, color="gray", alpha=0.2)
            ax.set_ylim(lower_bands[-1] - app.trade_cost - 2, upper_bands[-1] + app.trade_cost + 2)
            ax.legend(ncol=3, loc="upper center")

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
contract.lastTradeDateOrContractMonth = "202412"

# # Define the BTC cryptocurrency contract
# contract = Contract()
# contract.symbol = "BTC"
# contract.secType = "CRYPTO"
# contract.exchange = "ZEROHASH"
# contract.currency = "USD"

# # Define the EUR.USD forex contract
# contract = Contract()
# contract.symbol = "EUR"
# contract.secType = "CASH"
# contract.exchange = "IDEAL"  # for virtual trading. Use "IDEALPRO" for real trading
# contract.currency = "USD"


app = IBApi(contract)
app.connect("127.0.0.1", 7497, 0)  # Ensure TWS is running on this port with Paper Trading login
api_thread = threading.Thread(target=app.run, daemon=True)
api_thread.start()
time.sleep(1)  # Sleep to ensure connection is established


# Subscribe to live market data for ES
print("Subscribing to market data")
app.subscribe_to_market_data(contract)

print("Waiting for prices")
app.wait_for_prices()
print("Prices received")

# app.open_long_position()
# app.close_long_position()

# mean_reversion(app, 300, 300, 2)
# TF_MR_switch_on_rapid_close(app)
# trend_with_bollinger_breakout(app, 150, 0.01, 0.1, 0.015)
# mean_rev_within_boll_and_trend_on_breakout(app, 300, 0.01, 0.1, 0.015)
# mean_reversion(app, 2400, "N/A", 2)
# mean_reversion_with_ema(app, 2400, 2)

# mean_reversion_with_ema_deviation_sensitive(app, 0.01, 300)

# crypto_trend_following_with_no_naked_shorts(
#     app, slow_alpha=0.01, fast_alpha=0.1, take_profit_alpha=0.025, entry_macd=0.70
# )

# mean_reversion(app, 600, None, 2)
# mean_reversion_at_the_bands(app, 600, None, 2)


slow = 0.002
fast = 0.0314
take_profit = fast / 3
trend_following(app, slow_alpha=slow, fast_alpha=fast, take_profit_alpha=take_profit, entry_macd=0.82)
