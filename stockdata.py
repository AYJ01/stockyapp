"""
Stock Forecasting App (Streamlit + Prophet + yfinance)

Install:  pip install streamlit yfinance prophet plotly pandas numpy
Run:      streamlit run stock_forecast_app.py
"""
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

st.set_page_config(page_title="Stock Forecasting App", page_icon="📈", layout="wide")

POPULAR = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "NKE",
    "MCD", "PYPL", "UBER", "SPOT", "RBLX", "BABA", "MSI",
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def mae(actual, pred):
    return float(np.mean(np.abs(actual - pred)))


def to_price(values, log_transform):
    """Convert model-space values back to price space."""
    return np.exp(values) if log_transform else values


def build_model(changepoint_prior, log_transform):
    # Stock data has no intraday info, so daily seasonality is switched off.
    # In log space, additive seasonality == multiplicative in price space.
    return Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive" if log_transform else "multiplicative",
        changepoint_prior_scale=changepoint_prior,
        interval_width=0.80,
    )


def to_train_frame(df, log_transform):
    out = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    if log_transform:
        out["y"] = np.log(out["y"])
    return out


# ----------------------------------------------------------------------------
# Cached data / model functions
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start: str, end: str):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    # Newer yfinance versions return MultiIndex columns like ("Close", "AAPL")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    return df


@st.cache_resource(ttl=3600, show_spinner=False)
def fit_and_forecast(train: pd.DataFrame, horizon: int, changepoint_prior: float, log_transform: bool):
    model = build_model(changepoint_prior, log_transform)
    model.fit(train)
    # freq="B" = business days, so the forecast skips weekends
    future = model.make_future_dataframe(periods=horizon, freq="B")
    forecast = model.predict(future)
    return model, forecast


@st.cache_data(ttl=3600, show_spinner=False)
def walk_forward_backtest(train: pd.DataFrame, fold_len: int, folds: int,
                          changepoint_prior: float, log_transform: bool):
    """Train on data up to a cutoff, predict the next `fold_len` days, compare
    against what actually happened. Repeated over several cutoffs."""
    rows, n = [], len(train)
    for k in range(folds, 0, -1):
        cut = n - k * fold_len
        if cut < 250:
            continue
        tr, te = train.iloc[:cut], train.iloc[cut:cut + fold_len]
        model = build_model(changepoint_prior, log_transform)
        model.fit(tr)
        pred = to_price(model.predict(te[["ds"]])["yhat"].to_numpy(), log_transform)
        actual = to_price(te["y"].to_numpy(), log_transform)
        naive = np.full_like(actual, to_price(tr["y"].iloc[-1], log_transform))
        rows.append({
            "Train cutoff": tr["ds"].iloc[-1].date(),
            "Prophet MAPE %": mape(actual, pred),
            "Naive MAPE %": mape(actual, naive),
            "Prophet MAE": mae(actual, pred),
            "Naive MAE": mae(actual, naive),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Popular tickers", POPULAR)
    custom = st.text_input("...or type any ticker", placeholder="e.g. NVDA, RELIANCE.NS")
    ticker = (custom.strip() or choice).upper()

    start = st.date_input(
        "History start",
        value=date(2015, 1, 1),
        min_value=date(2000, 1, 1),
        max_value=date.today() - timedelta(days=365),
    )
    horizon = st.slider("Forecast horizon (trading days)", 20, 504, 126,
                        help="~21 trading days = 1 month, ~252 = 1 year")

    with st.expander("Model settings"):
        log_transform = st.checkbox(
            "Log-transform prices", value=True,
            help="Models percentage moves instead of dollar moves. Usually better for stocks.")
        changepoint_prior = st.select_slider(
            "Trend flexibility", options=[0.01, 0.05, 0.1, 0.3, 0.5], value=0.05,
            help="Higher = trend reacts faster to recent changes but overfits more.")

    with st.expander("Backtest settings"):
        run_backtest = st.checkbox("Run walk-forward backtest", value=True)
        fold_len = st.slider("Fold length (trading days)", 20, 120, 60)
        n_folds = st.slider("Number of folds", 1, 5, 3)

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
st.title("📈 Stock Forecasting App")
st.caption("Prophet-based forecasts with a built-in backtest. Educational use only, not financial advice.")

try:
    with st.spinner(f"Loading {ticker}..."):
        df = load_data(ticker, str(start), str(date.today() + timedelta(days=1)))
except Exception as e:
    st.error(f"Error loading data for {ticker}: {e}")
    st.stop()

if df is None:
    st.error(f"No data found for ticker '{ticker}'. Check the symbol (Indian stocks need .NS or .BO).")
    st.stop()
if len(df) < 300:
    st.warning("Not enough history to fit a reliable model (need at least ~300 trading days). "
               "Pick an earlier start date or a different ticker.")
    st.stop()

# ----------------------------------------------------------------------------
# Fit model
# ----------------------------------------------------------------------------
train = to_train_frame(df, log_transform)
with st.spinner("Fitting model..."):
    model, fc_raw = fit_and_forecast(train, horizon, changepoint_prior, log_transform)

fc = fc_raw.copy()
for col in ("yhat", "yhat_lower", "yhat_upper"):
    fc[col] = to_price(fc[col], log_transform)

last_date = df["Date"].iloc[-1]
last_price = float(df["Close"].iloc[-1])
future_fc = fc[fc["ds"] > last_date]
end_row = future_fc.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{ticker} last close", f"{last_price:,.2f}")
c2.metric(f"Forecast on {end_row['ds'].date()}", f"{end_row['yhat']:,.2f}",
          f"{(end_row['yhat'] / last_price - 1) * 100:+.1f}%")
c3.metric("80% range (low)", f"{end_row['yhat_lower']:,.2f}")
c4.metric("80% range (high)", f"{end_row['yhat_upper']:,.2f}")

tab_hist, tab_fc, tab_bt, tab_comp = st.tabs(
    ["Historical data", "Forecast", "Backtest (accuracy)", "Components"])

# ----------------------------------------------------------------------------
# Tab 1: historical
# ----------------------------------------------------------------------------
with tab_hist:
    d = df.copy()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["MA50"], name="50-day MA", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["MA200"], name="200-day MA", line=dict(width=1.2)))
    fig.update_layout(title=f"{ticker} price history", template="plotly_dark",
                      xaxis_rangeslider_visible=True, height=550)
    st.plotly_chart(fig)

    if "Volume" in d.columns:
        vol = go.Figure(go.Bar(x=d["Date"], y=d["Volume"], name="Volume"))
        vol.update_layout(title="Volume", template="plotly_dark", height=250)
        st.plotly_chart(vol)

    with st.expander("Raw data (last 10 rows)"):
        st.dataframe(df.tail(10), hide_index=True)

# ----------------------------------------------------------------------------
# Tab 2: forecast
# ----------------------------------------------------------------------------
with tab_fc:
    hist_tail = df.tail(750)
    fit_tail = fc[(fc["ds"] >= hist_tail["Date"].iloc[0]) & (fc["ds"] <= last_date)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["Close"], name="Actual close",
                             line=dict(color="#4c9be8")))
    fig.add_trace(go.Scatter(x=fit_tail["ds"], y=fit_tail["yhat"], name="Model fit",
                             line=dict(color="gray", dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=future_fc["ds"], y=future_fc["yhat_upper"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_fc["ds"], y=future_fc["yhat_lower"], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(255,165,0,0.2)",
                             name="80% interval"))
    fig.add_trace(go.Scatter(x=future_fc["ds"], y=future_fc["yhat"], name="Forecast",
                             line=dict(color="orange", width=2)))
    fig.update_layout(title=f"{ticker}: {horizon}-trading-day forecast", template="plotly_dark",
                      xaxis_rangeslider_visible=True, height=550)
    st.plotly_chart(fig)

    out = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={
        "ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower (80%)", "yhat_upper": "Upper (80%)"})
    st.dataframe(out.tail(15), hide_index=True)
    st.download_button("Download forecast as CSV", out.to_csv(index=False),
                       file_name=f"{ticker}_forecast.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# Tab 3: backtest
# ----------------------------------------------------------------------------
with tab_bt:
    if not run_backtest:
        st.info("Enable the walk-forward backtest in the sidebar.")
    else:
        with st.spinner("Running backtest (fits one model per fold)..."):
            bt = walk_forward_backtest(train, fold_len, n_folds, changepoint_prior, log_transform)
        if bt.empty:
            st.warning("Not enough history for the chosen fold settings.")
        else:
            st.dataframe(bt.style.format({c: "{:.2f}" for c in bt.columns if c != "Train cutoff"}),
                         hide_index=True)
            p, n = bt["Prophet MAPE %"].mean(), bt["Naive MAPE %"].mean()
            b1, b2 = st.columns(2)
            b1.metric("Prophet avg MAPE", f"{p:.2f}%")
            b2.metric("Naive baseline avg MAPE", f"{n:.2f}%", help="Naive = 'price stays at last known value'")
            if p < n:
                st.success("Prophet beat the naive baseline on average for this stock and these settings.")
            else:
                st.warning("Prophet did NOT beat the naive 'no change' baseline here. "
                           "Treat this forecast as a trend illustration, not a prediction.")

# ----------------------------------------------------------------------------
# Tab 4: components
# ----------------------------------------------------------------------------
with tab_comp:
    if log_transform:
        st.caption("Components are shown in log-price space (additive effects = percentage effects).")
    st.pyplot(model.plot_components(fc_raw))
