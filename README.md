# Financial-Machine-Learning

Here are implementations of the book *Advances in Financial Machine Learning* by Dr. Marcos Lopez de Prado, along with other quantitative finance ideas and techniques I come across, like automated intrinsic value estimation of companies, and statistical arbitrage on spreads of cointegrated pairs of stock. 

Contents:
- Financial Data Structures (AFML Chapter 2)
    - Information Driven Bars
        - Imbalance Bars
            - Tick Imbalance Bars
            - Volume/Dollar Imbalance Bars
        - Runs Bars
            - Tick Runs Bars
            - Volume/Dollar Runs Bars


- Trading
    - Basic trading strategies for mean reversion and trend following
    - API class for integration with Interactive Brokers to send orders and get real time market data
    - Visualizing the trading strategies, watch how they run on realtime data
 
- Backtesting
    - Simulating strategy performance on historical data 
    - Variable data sampling rate, i.e. sample the price every 1/4 second.
    - Modeling slippage, commissions, and transaction costs
    - Visualizing the model on historical data, as if it were live (variable playback speed)
    - Testing new strategies
  
