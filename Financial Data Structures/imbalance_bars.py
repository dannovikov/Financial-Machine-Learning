import pandas as pd


def tick_imbalance_bars(data: pd.DataFrame, alpha: float, et_init=100) -> pd.DataFrame:
    """Get tick imbalance bars for a price series

    Implementation described in AFML 2.3.2.1 "Tick Imbalance Bars"

    Args:
        data: a dataframe containing "price" and "volume" columns
        alpha: decay factor for EWMA
        et_init: initial value for expected number of ticks per bar

    Returns:
        A dataframe containing tick imbalance bars, which recast the price
        series into bars sized by information content.
    """

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

    return pd.DataFrame(bars)


def volume_imbalance_bars(data: pd.DataFrame, alpha: float, et_init=100) -> pd.DataFrame:
    """Get volume imbalance bars for a price series

    Implementation described in AFML 2.3.2.2 "Volume/Dollar Imbalance Bars"

    Args:
        data: a dataframe containing "price" and "volume" columns
        alpha: decay factor for EWMA
        et_init: initial value for expected number of ticks per bar

    Returns:
        A dataframe containing tick imbalance bars, which recast the price
        series into bars sized by information content.
    """

    def make_bar(start_idx, end_idx, data, bars):
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
        for i in range(len(b_hist)):
            if b == 1:
                bar_vp += v_hist[i]
            else:
                bar_vm += v_hist[i]
        bar_vp /= len(b_hist)
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

    p = data["price"]
    v = data["volume"]
    b = 1
    b_last = 1
    b_hist = []
    v_hist = []
    theta = 0

    et = 1
    vp = 1
    vm = 0

    for t in range(1, len(p)):
        dp = p[t] - p[t - 1]
        if dp == 0:
            b = b_last
        else:
            b = 1 if dp > 0 else -1
            b_last = b
        theta += b * v[t]
        b_hist.append(b)

        if abs(theta) >= et * abs(vp - vm):
            T = t - bars["end_idx"][-1]
            # make bar
            make_bar(bars["end_idx"][-1], t, data, bars)
            # update ewma
            et = alpha * T + (1 - alpha) * et
            bar_vp, bar_vm = compute_bar_v(b_hist, v_hist)
            vp = alpha * bar_vp + (1 - alpha) * vp
            vm = alpha * bar_vm + (1 - alpha) * vm

            # reset
            b_hist = []
            v_hist = []
            theta = 0

    return pd.DataFrame(bars)
