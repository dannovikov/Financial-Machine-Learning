import yfinance as yf
import numpy as np
import numpy_financial as npf


def get_financials(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.financials.T  # Get financials transposed
    print("Financials data:")
    print(financials)
    print("Available columns:", financials.columns)
    return financials


def get_revenue_and_expenses(financials):
    revenue_column = "Total Revenue"
    expenses_column = "Total Expenses"

    revenue = financials[revenue_column].astype(float)
    expenses = financials[expenses_column].astype(float)

    # Only use the last 3 years of data
    revenue = revenue[-3:]
    expenses = expenses[-3:]

    print("Revenue data:")
    print(revenue)
    print("Expenses data:")
    print(expenses)
    return revenue, expenses


def compute_average_growth_rate(data):
    growth_rates = []
    for i in range(1, len(data)):
        growth_rate = (data[i] - data[i - 1]) / data[i - 1]
        growth_rates.append(growth_rate)
    average_growth_rate = np.mean(growth_rates)
    print("Growth Rates:", growth_rates)
    print("Average Growth Rate:", average_growth_rate)
    return average_growth_rate


def forecast_revenue_and_expenses(revenue, expenses, growth_rate_revenue, growth_rate_expenses):
    forecasted_revenue = []
    forecasted_expenses = []
    last_revenue = revenue[-1]
    last_expenses = expenses[-1]

    for i in range(1, 6):
        last_revenue *= 1 + growth_rate_revenue
        forecasted_revenue.append(last_revenue)

    for i in range(6, 11):
        last_revenue *= 1 + growth_rate_revenue / 2
        forecasted_revenue.append(last_revenue)

    for i in range(11, 31):
        last_revenue *= 1 + growth_rate_revenue / 4
        forecasted_revenue.append(last_revenue)

    for i in range(1, 31):
        last_expenses *= 1 + growth_rate_expenses
        forecasted_expenses.append(last_expenses)

    print("Forecasted Revenue:", forecasted_revenue)
    print("Forecasted Expenses:", forecasted_expenses)
    return forecasted_revenue, forecasted_expenses


def compute_forecasted_net_income(forecasted_revenue, forecasted_expenses):
    forecasted_net_income = np.array(forecasted_revenue) - np.array(forecasted_expenses)
    print("Forecasted Net Income:", forecasted_net_income)
    return forecasted_net_income


def compute_npv(forecasted_net_income, discount_rate=0.15):
    npv = npf.npv(discount_rate, forecasted_net_income)
    print("NPV:", npv)
    return npv


def compute_share_value(ticker, npv):
    stock = yf.Ticker(ticker)
    shares_outstanding = stock.info.get("sharesOutstanding", 0)
    print("Shares Outstanding:", shares_outstanding)
    if shares_outstanding == 0:
        raise ValueError("Shares outstanding data is not available.")
    share_value = npv / shares_outstanding
    return share_value


def main(ticker):
    financials = get_financials(ticker)
    revenue, expenses = get_revenue_and_expenses(financials)
    revenue = revenue[~np.isnan(revenue)]
    expenses = expenses[~np.isnan(expenses)]

    if len(revenue) < 2 or len(expenses) < 2:
        raise ValueError("Not enough valid data available. At least 2 years of revenue and expense data are required.")

    growth_rate_revenue = compute_average_growth_rate(revenue)
    growth_rate_expenses = compute_average_growth_rate(expenses)
    forecasted_revenue, forecasted_expenses = forecast_revenue_and_expenses(
        revenue, expenses, growth_rate_revenue, growth_rate_expenses
    )
    forecasted_net_income = compute_forecasted_net_income(forecasted_revenue, forecasted_expenses)
    npv = compute_npv(forecasted_net_income, 0.15)
    share_value = compute_share_value(ticker, npv)
    return share_value


# Example usage
ticker = "LMND"  # Replace with desired ticker
try:
    share_value = main(ticker)
    print(f"The estimated share value for {ticker} is: ${share_value:.2f}")
except Exception as e:
    print(f"An error occurred: {e}")
