import pandas as pd
import plotly.graph_objects as go


def plot_bars_on_prices(price_df, bars_df):
    """
    Plot tick imbalance bars on the price chart using Plotly.

    Args:
        price_df: a dataframe containing "price" and "volume" columns.
        bars_df: a dataframe containing tick imbalance bars with columns
                 "start_idx", "end_idx", "open", "high", "low", "close", "volume".
    """
    # Create candlestick chart for tick imbalance bars
    candlestick = go.Candlestick(
        x=bars_df["end_idx"],  # Use end_idx for x values
        open=bars_df["open"],
        high=bars_df["high"],
        low=bars_df["low"],
        close=bars_df["close"],
        increasing_line_color="green",
        decreasing_line_color="red",
        name="Tick Imbalance Bars",
    )

    # Create line chart for price data
    line = go.Scatter(x=list(range(len(price_df["price"]))), y=price_df["price"], mode="lines", name="Price")

    # Create figure
    fig = go.Figure()

    # Add line chart
    fig.add_trace(line)

    # Add candlestick chart
    fig.add_trace(candlestick)

    # Add rectangles to reflect the number of ticks in the bar width
    fig.update_layout(
        title="Tick Imbalance Bars on Price Chart",
        xaxis_title="Ticks",
        yaxis_title="Price",
        xaxis=dict(rangeslider=dict(visible=False)),
        shapes=[
            dict(
                x0=row["start_idx"],
                x1=row["end_idx"],
                y0=row["low"],
                y1=row["high"],
                type="rect",
                xref="x",
                yref="y",
                line=dict(color="green" if row["open"] <= row["close"] else "red"),
            )
            for idx, row in bars_df.iterrows()
        ],
    )

    # Show the figure
    fig.show()


# Example usage
# plot_tick_imbalance_bars_on_prices(df, bars)
