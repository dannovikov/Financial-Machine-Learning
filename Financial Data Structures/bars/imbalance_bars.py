from datetime import datetime, timedelta
import pytz


def tick_imbalance_bars(data: dict, alpha: float, et_init=100, verbose=True) -> dict:
    """Get tick imbalance bars for a price series

    As described in AFML 2.3.2.1 "Tick Imbalance Bars", TIBs represent
    the exceeding accumulation of signed trades w.r.t past expectations.

    Args:
        data: a dictionary containing "price" and "volume" lists
        alpha: decay factor for EWMA
        et_init: initial value for expected number of ticks per bar
        verbose: whether to display a progress bar

    Returns:
        A dictionary containing tick imbalance bars, which recast the price
        series into bars sized by information content.
    """
    tqdm = _handle_verbose(verbose)

    def make_bar(start_idx, end_idx, data, bars):
        T = end_idx - start_idx
        bars["start_idx"].append(start_idx)
        bars["end_idx"].append(end_idx)
        bars["open"].append(data["price"][start_idx])
        bars["high"].append(max(data["price"][start_idx:end_idx]) if T > 0 else data["price"][end_idx])
        bars["low"].append(min(data["price"][start_idx:end_idx]) if T > 0 else data["price"][end_idx])
        bars["close"].append(data["price"][end_idx])
        bars["volume"].append(sum(data["volume"][start_idx:end_idx]) if T > 0 else data["volume"][end_idx])

    def compute_bar_pb(b_hist):
        bar_pb = 0
        for b in b_hist:
            if b == 1:
                bar_pb += 1
        bar_pb /= len(b_hist)
        return bar_pb

    bars = {
        "start_idx": [0],
        "end_idx": [1],
        "open": [data["price"][0]],
        "high": [data["price"][0]],
        "low": [data["price"][0]],
        "close": [data["price"][0]],
        "volume": [data["volume"][0]],
    }

    p = data["price"]
    b = 1
    b_last = 1
    b_hist = []
    theta = 0
    et = et_init
    pb = 1

    for t in range(1, len(p)):
        dp = p[t] - p[t - 1]
        if dp == 0:
            b = b_last
        else:
            b = 1 if dp > 0 else -1
            b_last = b
        theta += b
        b_hist.append(b)

        if abs(theta) >= et * abs(2 * pb - 1):
            T = t - bars["end_idx"][-1]
            make_bar(bars["end_idx"][-1], t, data, bars)

            et = alpha * T + (1 - alpha) * et
            bar_pb = compute_bar_pb(b_hist)
            pb = alpha * bar_pb + (1 - alpha) * pb

            b_hist = []
            theta = 0

    return bars


def volume_imbalance_bars(data: dict, alpha: float, et_init=100, max_bar_seconds=60 * 60, verbose=True) -> dict:
    """Get volume imbalance bars for a price series

    As described in AFML 2.3.2.2 "Volume/Dollar Imbalance Bars", VIBs represent
    the exceeding accumulation of signed volume w.r.t past expectations.

    Args:
        data: a dictionary containing "time", "price" and "volume" lists
        alpha: decay factor for EWMA
        et_init: initial value for expected number of ticks per bar
        max_bar_seconds: maximum number of seconds for a bar
        verbose: whether to display a progress bar

    Returns:
        A dictionary containing volume imbalance bars, which recast the price
        series into bars sized by information content.
    """

    tqdm = _handle_verbose(verbose)
    max_bar_seconds = timedelta(seconds=max_bar_seconds)  # type: ignore

    def make_bar(start_idx, end_idx, data, bars):
        # Creates a new bar from start_idx to end_idx
        # print(f"Making a new bar from {start_idx} to {end_idx}")
        bars["start_idx"].append(start_idx)
        bars["end_idx"].append(end_idx)
        bars["open"].append(data["price"][start_idx])
        bars["high"].append(max(data["price"][start_idx:end_idx]))
        bars["low"].append(min(data["price"][start_idx:end_idx]))
        bars["close"].append(data["price"][end_idx])
        bars["volume"].append(sum(data["volume"][start_idx:end_idx]))

    def compute_bar_v(b_hist, v_hist):
        bar_vp = 0
        bar_vm = 0
        bar_pb = 0
        for i in range(len(b_hist)):
            if b_hist[i] == 1:
                bar_vp += v_hist[i]
                bar_pb += 1
        bar_pb /= len(b_hist)
        bar_vm = sum(v_hist) - bar_vp
        return bar_vp, bar_vm

    bars = {
        "start_idx": [0],
        "end_idx": [0],
        "open": [data["price"][0]],
        "high": [data["price"][0]],
        "low": [data["price"][0]],
        "close": [data["price"][0]],
        "volume": [data["volume"][0]],
    }

    p = data["price"]  # series of prices
    v = data["volume"]  # series of volumes
    times = data["time"]  # series of timmestamps

    b = 1  # current tick direction
    b_last = b  # last tick direction

    b_hist = []  # history of tick directions
    v_hist = []  # history of volumes
    theta = 0  # tick imbalance

    et = et_init  # expected number of ticks per bar
    vp = et // 2  # expected volume of positive ticks
    vm = et - vp  # expected volume of negative ticks

    last_bar_time = data["time"][0]

    c1, c2 = 0, 0

    for t in tqdm(range(1, len(p)), desc=f"a={alpha:.2f}, et_init={et_init:.2f}"):
        # compute tick direction
        dp = p[t] - p[t - 1]
        if dp == 0:
            b = b_last
        else:
            b = 1 if dp > 0 else -1
        b_last = b

        # update tick imbalance
        theta += b * v[t]
        b_hist.append(b)
        v_hist.append(v[t])

        current_time = times[t]

        # check if a new bar is created
        if abs(theta) >= et * abs(vp - vm) or current_time - last_bar_time > max_bar_seconds:
            if current_time - last_bar_time > max_bar_seconds:
                c2 += 1
            else:
                c1 += 1
            T = t - bars["end_idx"][-1]
            make_bar(bars["end_idx"][-1], t, data, bars)
            # update expectations
            bar_vp, bar_vm = compute_bar_v(b_hist, v_hist)
            vp = alpha * bar_vp + (1 - alpha) * vp
            vm = alpha * bar_vm + (1 - alpha) * vm
            et = alpha * T + (1 - alpha) * et
            # reset tick imbalance
            last_bar_time = times[t]
            b_hist = []
            v_hist = []
            theta = 0
    print(f"Created {c1} bars due to theta and {c2} bars due to time")
    return bars


def _handle_verbose(verbose):
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore
        except ImportError as e:
            print("Please install tqdm for progress bar. Defaulting to verbose=False.")

            def tqdm(x, *args, **kwargs):
                for i in x:
                    yield i

    else:

        def tqdm(x, *args, **kwargs):
            for i in x:
                yield i

    return tqdm


if __name__ == "__main__":
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import numpy as np
    import os

    print(os.getcwd())

    filtered_csv = "Historical Data/ES-Futures-Ticks-20230807-20240806.trades.filtered.csv"  # now input

    data = {"time": [], "price": [], "volume": [], "symbol": []}
    with open(filtered_csv, "r") as f:
        for i, line in enumerate(tqdm(f, total=1000000)):
            if i == 0:
                continue
            if i == 1000000:
                break
            line = line.strip().split(",")
            data["time"].append(line[0])
            data["price"].append(float(line[1]))
            data["volume"].append(float(line[2]))
            data["symbol"].append(line[3])
    # lets try many values for alpha and et_init
    # lets vary alpha from 0.01 to 0.05, 10 values
    # lets vary et_init from 1 to 25, 10 values
    alphas = np.linspace(0.01, 0.03, 10)
    et_inits = np.linspace(1, 25, 10)
    results = {}
    results2 = {}
    # bars = volume_imbalance_bars(data, alpha=0.05, et_init=10, verbose=True)
    count = 0
    for a in tqdm(alphas):
        for e in et_inits:
            # print(f"{count}/{len(alphas) * len(et_inits)}")
            count += 1
            bars = volume_imbalance_bars(data, alpha=a, et_init=e, verbose=False)
            # results[(a, e)] = len(bars["start_idx"])
            results[(a, len(bars["start_idx"]))] = sum(
                [bars["start_idx"][i + 1] - bars["start_idx"][i] for i in range(len(bars["start_idx"]) - 1)]
            ) / len(bars["start_idx"])
            results2[(a, e)] = len(bars["start_idx"])

    for r in results.items():
        print(r)
    # 3d plot of results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = [x[0] for x in results.keys()]
    y = [x[1] for x in results.keys()]
    z = list(results.values())
    ax.scatter(x, y, z)
    # plot bars as a function of alpha and et_init
    # ax.plot_trisurf(x, y, z, cmap="viridis")
    ax.set_xlabel("alpha")
    ax.set_ylabel("et_init")
    ax.set_zlabel("num_bars")  # type: ignore
    plt.show()

    # # a new 2d plot of results.keys[1] vs results.values()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x = [x[1] for x in results.keys()]
    # y = list(results.values())
    # ax.scatter(x, y)
    # ax.set_ylabel("index gap")
    # ax.set_xlabel("number of bars")

    # plt.show()

    # plot results1 values as a function of results2 values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = list(results2.values())
    y = list(results.values())
    ax.scatter(x, y)
    ax.set_ylabel("mean bar size")
    ax.set_xlabel("number of bars")
    plt.show()
