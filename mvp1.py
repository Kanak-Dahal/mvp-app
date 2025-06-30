import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import time

st.title('Modern Portfolio Theory')
st.header('Portfolio Analysis and Computation of Minimum Variance Portfolio')

tickers = [
    'Select','RELIANCE','TATAMOTORS','HDFCBANK','INFY','HINDUNILVR','TCS','ICICIBANK','LT','MARUTI',
    'ASIANPAINT','KOTAKBANK','HCLTECH','ITC','SBIN','BAJFINANCE','AXISBANK','WIPRO','SUNPHARMA','ONGC',
    'ADANIPORTS','BHARTIARTL','ULTRACEMCO','TATAPOWER','TECHM','JSWSTEEL','POWERGRID','NTPC','HINDALCO',
    'GRASIM','DIVISLAB','DRREDDY','CIPLA','BPCL','IOC','GAIL','HAVELLS','TATACHEM','ADANIGREEN','ADANIPOWER',
    'ADANIGAS','ADANIPORTS'
]

dataframe = pd.DataFrame()

# === Yfinance retry wrapper ===
def safe_yf_download(ticker, interval="1d", period="5y", max_retries=3):
    tries = 0
    while tries < max_retries:
        try:
            comp = yf.Ticker(ticker)
            return comp.history(interval=interval, period=period)['Close']
        except yf.shared._exceptions.YFRateLimitError:
            tries += 1
            wait = 10 * tries
            st.warning(f"Rate limit hit for {ticker}. Retrying in {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
            return pd.Series(dtype=float)
    st.error(f"Failed to fetch {ticker} after {max_retries} retries.")
    return pd.Series(dtype=float)

# Sidebar
client_selected = st.sidebar.multiselect('Select stocks for portfolio analysis', tickers[1:])
st.sidebar.write('You selected:', client_selected)

if client_selected:
    st.sidebar.write('Fetching stock data...')

    for ticker in client_selected:
        df = safe_yf_download(f"{ticker}.NS")
        if not df.empty:
            dataframe[ticker] = df

    if dataframe.empty:
        st.error("No data could be fetched for the selected stocks.")
        st.stop()

    client_selected_df = dataframe[client_selected]

    # stock specific data
    stock_mean_returns = client_selected_df.pct_change().mean() * 252
    stock_std_dev = client_selected_df.pct_change().std() * np.sqrt(252)

    for ticker in client_selected_df.columns:
        st.subheader(ticker)
        st.write(f"Mean Return: {stock_mean_returns[ticker]:.2%}")
        st.write(f"Standard Deviation: {stock_std_dev[ticker]:.2%}")
        st.line_chart(client_selected_df[ticker])

    # portfolio optimization
    def get_min_variance_weights(returns):
        num_assets = returns.shape[1]
        cov_matrix = returns.cov()
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        result = minimize(portfolio_volatility, init_guess,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        return result.x

    returns = client_selected_df.pct_change().dropna()
    weights = get_min_variance_weights(returns)

    st.subheader('Optimal Portfolio Weights')
    st.markdown(
        "<span style='white-space: nowrap;'>" +
        ",     ".join(f" {ticker} :  {weight:.2%} " for ticker, weight in zip(client_selected, weights)) +
        "</span>",
        unsafe_allow_html=True
    )

    # portfolio performance
    def portfolio_return(weights, returns):
        return np.sum(returns.mean() * weights) * 252

    def portfolio_volatility(weights, returns):
        cov_matrix = returns.cov() * 252
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def sharpe_ratio(weights, returns, risk_free_rate=0.0677):
        port_return = portfolio_return(weights, returns)
        port_volatility_val = portfolio_volatility(weights, returns)
        return (port_return - risk_free_rate) / port_volatility_val

    pret = portfolio_return(weights, returns)
    stdev = portfolio_volatility(weights, returns)
    sharpe = sharpe_ratio(weights, returns)

    # efficient frontier
    st.subheader("Efficient Frontier Visualisation")

    num_portfolios = 5000
    num_assets = len(client_selected)
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        random_weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        port_return = portfolio_return(random_weights, returns)
        port_volatility_val = portfolio_volatility(random_weights, returns)
        port_sharpe = sharpe_ratio(random_weights, returns)
        results[0, i] = port_volatility_val
        results[1, i] = port_return
        results[2, i] = port_sharpe

    max_sharpe_idx = results[2].argmax()
    max_sharpe_volatility = results[0, max_sharpe_idx]
    max_sharpe_return = results[1, max_sharpe_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
    ax.scatter(stdev, pret, marker='*', color='red', s=200, label='Minimum Variance Portfolio')
    ax.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='violet', s=200, label='Maximum Sharpe Portfolio')
    ax.set_xlabel('Volatility (Std. Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    fig.colorbar(scatter, label='Sharpe Ratio')
    st.pyplot(fig)

    st.subheader('Portfolio Performance Metrics')
    st.write(f"Expected Annual Return: {pret:.2%}")
    st.write(f"Annualized Volatility: {stdev:.2%}")
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

    # === Nifty comparison with retry ===
    nifty_df = safe_yf_download("^NSEI")
    if nifty_df.empty:
        st.error("Could not fetch Nifty data. Try again later.")
        st.stop()

    nifty_returns = nifty_df.pct_change().dropna()
    portfolio_daily_returns = (returns @ weights)
    portfolio_cum = (1 + portfolio_daily_returns).cumprod()
    nifty_cum = (1 + nifty_returns).cumprod()

    compare_df = pd.DataFrame({
        'Portfolio': portfolio_cum,
        'Nifty': nifty_cum
    })

    st.subheader('Portfolio vs Nifty Performance')
    st.line_chart(compare_df)

    # === LSTM Forecast Portfolio on cumulative returns ===
    st.subheader("Forecasting Portfolio Cumulative Returns")

    # [all the LSTM block stays identical to your code]
    # no change needed there since error was yfinance
    # so you can keep your exact LSTM code as-is below
    # --------------------------------------------------
    # but if you want I can optimize its speed or do changes
    # let me know

else:
    st.info("Please select stocks from the sidebar to perform the portfolio analysis.")
