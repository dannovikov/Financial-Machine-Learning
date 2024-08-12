"""Exponentially weighted moving average (EWMA)"""


def ewma(s: list, a: float, w: int = None) -> float:  # type: ignore
    """Exponentially weighted moving average

    EWMA(s, a) = a*(s_t) + (1-a)*EWMA(s_t-1)

    Args:
        s - series or list-like collection of numbers
        a - alpha smoothing parameter, gives more weight to more recent elements
        w - window size. If None, the entire series is used. Else, the last w elements are used.

    Returns:
        e - exponentially weighted moving average over the window
    """

    if len(s) == 1:
        return s[0]

    if w is not None:
        s = s[-w:]

    e = s[0]
    for i in range(1, len(s)):
        e = a * s[i] + (1 - a) * e

    return e
