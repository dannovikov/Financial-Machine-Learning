"""Exponentially weighted moving average (EWMA)"""


def ewma(s: list, a: float) -> list:
    """Exponentially weighted moving average

    EWMA(s, a) = a*(s_t) + (1-a)*EWMA(s_t-1)

    Args:
        s - series or list-like window of data
        a - alpha smoothing parameter, gives more weight to more recent elements

    Returns:
        a list of EWMA values for the window
    """

    if len(s) == 1:
        return s
    ewma_values = [s[0]]
    for i in range(1, len(s)):
        ewma_values.append(a * s[i] + (1 - a) * ewma_values[i - 1])
    return ewma_values
