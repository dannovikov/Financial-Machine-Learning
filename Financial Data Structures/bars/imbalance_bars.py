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


def volume_imbalance_bars(data: dict, alpha: float, et_init=100, verbose=True) -> dict:
    """Get volume imbalance bars for a price series

    As described in AFML 2.3.2.2 "Volume/Dollar Imbalance Bars", VIBs represent
    the exceeding accumulation of signed volume w.r.t past expectations.

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
        # Creates a new bar from start_idx to end_idx
        print(f"Making a new bar from {start_idx} to {end_idx}")
        bars["start_idx"].append(start_idx)
        bars["end_idx"].append(end_idx)
        bars["open"].append(data["price"][start_idx])
        bars["high"].append(max(data["price"][start_idx:end_idx]))
        bars["low"].append(min(data["price"][start_idx:end_idx]))
        bars["close"].append(data["price"][end_idx])
        bars["volume"].append(sum(data["volume"][start_idx:end_idx]))

    def compute_bar_v(b_hist, v_hist):
        # Compute the volume of positive and negative ticks
        bar_vp = 0
        bar_vm = 0
        for i in range(len(b_hist)):
            if b_hist[i] == 1:
                bar_vp += v_hist[i]
            else:
                bar_vm += v_hist[i]
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

    b = 1  # current tick direction
    b_last = b  # last tick direction

    b_hist = []  # history of tick directions
    v_hist = []  # history of volumes
    theta = 0  # tick imbalance

    et = et_init  # expected number of ticks per bar
    vp = 1  # expected volume of positive ticks
    vm = 0  # expected volume of negative ticks

    for t in tqdm(range(1, len(p))):
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

        # check if a new bar is created
        if abs(theta) >= et * abs(2 * vp - (vp + vm)):
            T = t - bars["end_idx"][-1]
            make_bar(bars["end_idx"][-1], t, data, bars)
            # update expectations
            bar_vp, bar_vm = compute_bar_v(b_hist, v_hist)
            vp = alpha * bar_vp + (1 - alpha) * vp
            vm = alpha * bar_vm + (1 - alpha) * vm
            et = alpha * T + (1 - alpha) * et
            # reset tick imbalance
            b_hist = []
            v_hist = []
            theta = 0

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
