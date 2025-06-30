# 📈 Modern Portfolio Theory & Forecasting with LSTM

A Streamlit-based interactive web application that applies Modern Portfolio Theory to optimize a user-selected stock portfolio, visualize the efficient frontier, and forecast its cumulative returns using advanced LSTM deep learning models. The app also compares the optimized portfolio’s performance to the Nifty 50 benchmark, making it a powerful tool for personal investing research and educational purposes.

---

## 🚀 Features

✅ Interactive stock selection from major NIFTY 50 constituents  
✅ Automated fetching of 5 years of historical data from Yahoo Finance  
✅ Computation of key portfolio performance metrics:
- Annualized mean return  
- Annualized volatility  
- Sharpe ratio  

✅ Minimum Variance Portfolio computation using numerical optimization  
✅ Generation of an efficient frontier with 5000 random portfolios  
✅ Identification and plotting of the maximum Sharpe ratio portfolio  
✅ Comparison of portfolio cumulative returns against the Nifty 50 index  
✅ 30-day cumulative returns forecasting for both portfolio and Nifty using LSTM (Long Short-Term Memory) models  
✅ Clean, interactive visualizations built with matplotlib and Streamlit  
✅ Deployable on platforms like Streamlit Cloud or your local machine

---

## 🛠️ Tech Stack

- **Python 3.10+**  
- pandas, numpy, scipy  
- matplotlib  
- scikit-learn  
- TensorFlow / Keras (for LSTM forecasting)  
- Streamlit  
- yfinance

---

## 📦 Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/modern-portfolio-lstm.git
cd modern-portfolio-lstm
