# app.py
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# -------------------- CONFIG --------------------
PRICES_DIR = "Necessary_CSVs"   # per-ticker CSVs with columns: Date, AdjClose (or Close)
TEST_START = pd.Timestamp("2024-10-01")
TEST_END   = pd.Timestamp("2025-03-31")
INITIAL_INV = 10000

st.set_page_config(page_title="SPX Momentum vs Reversal (Q4'24–Q1'25)",
                   layout="wide")

# -------------------- LOADERS --------------------
@st.cache_data(show_spinner=False)
def list_available_tickers():
    files = [f for f in os.listdir(PRICES_DIR) if f.endswith(".csv")]
    return sorted({os.path.splitext(f)[0] for f in files})

@st.cache_data(show_spinner=False)
def load_prices_from_folder(tickers):
    """Load per-ticker CSVs into a wide DataFrame (index=Date, columns=tickers)."""
    frames = []
    for t in tickers:
        p = os.path.join(PRICES_DIR, f"{t}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, parse_dates=["Date"])
        col = "AdjClose" if "AdjClose" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            continue
        df = df[["Date", col]].rename(columns={col: t}).set_index("Date")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index.name = "Date"
    return wide

@st.cache_data(show_spinner=False)
def load_spy_series(start_date: str, end_date: str) -> pd.Series:
    """
    Load SPY Adj Close between start_date and end_date (YYYY-MM-DD).
    Use local SPY.csv if present; otherwise fetch via yfinance (cached).
    """
    local = os.path.join(PRICES_DIR, "SPY.csv")
    if os.path.exists(local):
        df = pd.read_csv(local, parse_dates=["Date"])
        col = "AdjClose" if "AdjClose" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            raise ValueError("SPY.csv must have 'AdjClose' or 'Close' column.")
        s = df.set_index("Date")[col].rename("SPY")
        return s.loc[pd.to_datetime(start_date) - pd.Timedelta(days=5) :
                     pd.to_datetime(end_date)   + pd.Timedelta(days=5)]
    else:
        df = yf.download(
            "SPY",
            start=pd.to_datetime(start_date) - pd.Timedelta(days=5),
            end  =pd.to_datetime(end_date)   + pd.Timedelta(days=5),
            progress=False, auto_adjust=False, threads=False
        )
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return df[col].rename("SPY")

# -------------------- UTILS --------------------
def compute_cum(x: pd.Series) -> float:
    return (1 + x.dropna()).prod() - 1

def max_drawdown(ret: pd.Series) -> float:
    wealth = (1 + ret.fillna(0)).cumprod()
    dd = wealth / wealth.cummax() - 1
    return dd.min()

def partition_groups(formation: pd.Series, mode: str):
    """
    Return (top_index, bottom_index, ranks) based on selected grouping.
    10%/25% use percentile ranks. 50% uses median split with safe fallback.
    """
    ranks = formation.rank(pct=True)
    if mode == "Decile (Top 10% vs Bottom 10%)":
        top = ranks[ranks >= 0.9].index
        bot = ranks[ranks <= 0.1].index
    elif mode == "Quartile (Top 25% vs Bottom 25%)":
        top = ranks[ranks >= 0.75].index
        bot = ranks[ranks <= 0.25].index
    else:  # "Half (Top 50% vs Bottom 50%)"
        med = formation.median()
        top = formation[formation >= med].index
        bot = formation[formation <  med].index
        # Fallback if ties make one side empty
        if len(top) == 0 or len(bot) == 0:
            ranks2 = formation.rank(pct=True, method="average")
            top = ranks2[ranks2 >= 0.5].index
            bot = ranks2[ranks2 <  0.5].index
    return top, bot, ranks

def port_stats(series):
    std = series.std()
    if std == 0 or np.isnan(std):
        return np.nan, np.nan, np.nan
    ann_vol = std * np.sqrt(252)
    sharpe  = (series.mean() / std) * np.sqrt(252)
    return ann_vol, sharpe, max_drawdown(series)

# -------------------- SIDEBAR (with Run button) --------------------
with st.sidebar.form("controls"):
    st.header("Parameters")

    lookback_choice = st.selectbox(
        "Formation lookback",
        ["1 month", "3 months", "6 months", "12 months"],
        index=1
    )
    lb_map = {"1 month":1, "3 months":3, "6 months":6, "12 months":12}
    LOOKBACK_MONTHS = lb_map[lookback_choice]

    group_mode = st.selectbox(
        "Grouping",
        ["Decile (Top 10% vs Bottom 10%)",
         "Quartile (Top 25% vs Bottom 25%)",
         "Half (Top 50% vs Bottom 50%)"],
        index=0
    )

    run = st.form_submit_button("Run")

# Don’t run heavy work until user clicks
if not run:
    st.title("Momentum vs Reversal — S&P 500 (Point‑in‑Time) Q4’24–Q1’25")
    st.info("Select your lookback and grouping in the sidebar, then click **Run**.")
    st.stop()

formation_end   = TEST_START - pd.Timedelta(days=1)
formation_start = TEST_START - pd.DateOffset(months=LOOKBACK_MONTHS)

st.sidebar.write(f"Formation: **{formation_start.date()} → {formation_end.date()}**")
st.sidebar.write(f"Test: **{TEST_START.date()} → {TEST_END.date()}**")

# -------------------- LOAD DATA --------------------
all_files = list_available_tickers()
if not all_files:
    st.error(f"No CSVs found in `{PRICES_DIR}/`. Commit your per‑ticker files first.")
    st.stop()

# Exclude benchmark (and any extras you might keep in the folder)
EXCLUDE = {"SPY"}
tickers = [t for t in all_files if t not in EXCLUDE]

prices_all = load_prices_from_folder(tickers)
if prices_all.empty:
    st.error("Failed to load any prices from the CSVs (after exclusions).")
    st.stop()

# Require full data (no NaNs) in both formation and test windows
rets = prices_all.pct_change()
fwin = rets.loc[formation_start:formation_end]
twin = rets.loc[TEST_START:TEST_END]
keep = fwin.notna().all() & twin.notna().all()
rets = rets.loc[:, keep.index[keep]]

if rets.shape[1] < 50:
    st.warning(f"Only {rets.shape[1]} tickers have complete data in both windows. Results may be noisy.")

formation = fwin.apply(compute_cum)
test      = twin.apply(compute_cum)

# Partition by grouping (robust)
top, bot, ranks = partition_groups(formation, group_mode)
if len(top) == 0 or len(bot) == 0:
    st.error("Grouping produced an empty portfolio (likely all formation returns tied). Try a different lookback.")
    st.stop()

# -------------------- TITLE --------------------
st.title("Momentum vs Reversal — S&P 500 (Point‑in‑Time) Q4’24–Q1’25")
st.caption("Data source: per‑ticker CSVs in repo. Formation ranking uses selected lookback; "
           "test window is Oct 1, 2024 → Mar 31, 2025.")

# -------------------- CUMULATIVE (with SPY) --------------------
twin2 = rets.loc[TEST_START:TEST_END]
g_top = twin2[list(top)].mean(axis=1)
g_bot = twin2[list(bot)].mean(axis=1)

curves = {}
curves["Top"]    = (1 + g_top.fillna(0)).cumprod()
curves["Bottom"] = (1 + g_bot.fillna(0)).cumprod()

# SPY benchmark (cached by date strings)
dates_index = curves["Top"].index
spy_raw = load_spy_series(
    dates_index.min().strftime("%Y-%m-%d"),
    dates_index.max().strftime("%Y-%m-%d")
)
spy_aligned = spy_raw.reindex(dates_index).ffill()
spy_rets = spy_aligned.pct_change().fillna(0)
curves["SPY"] = (1 + spy_rets).cumprod()

cum_df = pd.DataFrame(curves).dropna()
cum_df = cum_df / cum_df.iloc[0] * INITIAL_INV

fig1 = px.line(cum_df.reset_index(), x="Date", y=["Top","Bottom","SPY"],
               title=(f"Cumulative Portfolio Value — {group_mode}  "
                      f"(Formation {formation_start.date()}→{formation_end.date()})"),
               labels={"value":"Portfolio Value ($)", "variable":"Series"})
fig1.for_each_trace(lambda t: t.update(line=dict(width=3)))
st.plotly_chart(fig1, use_container_width=True)

# -------------------- LONG–SHORT & SHORT–LONG --------------------
ls = g_top - g_bot
sl = g_bot - g_top
ls_w = (1 + ls.fillna(0)).cumprod(); ls_w = ls_w / ls_w.iloc[0] * INITIAL_INV
sl_w = (1 + sl.fillna(0)).cumprod(); sl_w = sl_w / sl_w.iloc[0] * INITIAL_INV

fig_ls = px.line(x=ls_w.index, y=ls_w.values,
                 labels={"x":"Date", "y":"Value ($)"},
                 title=f"Long–Short vs Short–Long ({group_mode})")
fig_ls.update_traces(name="Long Top / Short Bottom", line=dict(width=3))
fig_ls.add_trace(go.Scatter(x=sl_w.index, y=sl_w.values,
                            name="Long Bottom / Short Top",
                            mode="lines"))
st.plotly_chart(fig_ls, use_container_width=True)

# -------------------- DECILE BAR (gradient view) --------------------
deciles = (np.ceil(ranks * 10)).astype(int).clip(upper=10)
decile_df = pd.DataFrame({"formation": formation, "test": test, "decile": deciles})
avg_future = decile_df.groupby("decile", as_index=False)["test"].mean()

fig2 = px.bar(avg_future, x="decile", y="test",
              title="Average Test Return by Formation Decile",
              labels={"test":"Avg Test Return"})
fig2.update_yaxes(tickformat=".2%")
fig2.update_traces(text=avg_future["test"].map("{:.2%}".format), textposition="outside")
st.plotly_chart(fig2, use_container_width=True)

# -------------------- SCATTER + REGRESSION + CI (robust) --------------------
valid = pd.notna(formation) & pd.notna(test)
formation_ = formation[valid]
test_      = test[valid]

if (formation_.nunique() < 2) or (len(formation_) < 5):
    st.warning("Not enough cross‑sectional variation to fit regression for this selection.")
else:
    X = sm.add_constant(formation_)
    reg = sm.OLS(test_, X, missing="drop").fit()

    # Safely get the slope regardless of its Series name
    slope_key = next((k for k in reg.params.index if k.lower() != "const"), None)
    if slope_key is None or len(reg.params) < 2:
        st.warning("Regression returned only an intercept. Skipping regression chart.")
    else:
        beta  = float(reg.params.loc[slope_key])
        tval  = float(reg.tvalues.loc[slope_key])
        pval  = float(reg.pvalues.loc[slope_key])

        xgrid = np.linspace(formation_.min(), formation_.max(), 200)
        pred  = reg.get_prediction(sm.add_constant(xgrid)).summary_frame(alpha=0.05)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=np.hstack([xgrid, xgrid[::-1]]),
            y=np.hstack([pred["mean_ci_lower"], pred["mean_ci_upper"][::-1]]),
            fill='toself', showlegend=False, fillcolor="rgba(0,0,255,0.1)",
            line=dict(width=0)
        ))
        fig3.add_trace(go.Scatter(
            x=xgrid, y=pred["mean"], mode="lines",
            name=f"Fit β={beta:.2f} (t={tval:.2f}, p={pval:.3g})",
            line=dict(color="black", dash="dash")
        ))
        fig3.add_trace(go.Scatter(
            x=formation_, y=test_, mode="markers", name="Stocks",
            marker=dict(size=6, opacity=0.7)
        ))
        fig3.update_layout(
            title=f"Cross‑Section: Test vs Formation Return (lookback={LOOKBACK_MONTHS}m)",
            xaxis_title="Formation Return",
            yaxis_title="Test Return",
            template="plotly_white"
        )
        fig3.update_xaxes(tickformat=".2%")
        fig3.update_yaxes(tickformat=".2%")
        st.plotly_chart(fig3, use_container_width=True)

# -------------------- SUMMARY TABLE --------------------
ann_vol_top, sharpe_top, mdd_top = port_stats(g_top.dropna())
ann_vol_bot, sharpe_bot, mdd_bot = port_stats(g_bot.dropna())
ann_vol_ls,  sharpe_ls,  mdd_ls  = port_stats(ls.dropna())

summary = pd.DataFrame([
    ["Top",    compute_cum(g_top), ann_vol_top, sharpe_top, mdd_top],
    ["Bottom", compute_cum(g_bot), ann_vol_bot, sharpe_bot, mdd_bot],
    ["Long–Short", compute_cum(ls), ann_vol_ls, sharpe_ls, mdd_ls]
], columns=["Portfolio", "Cum Return", "Ann Vol", "Sharpe", "Max DD"]).set_index("Portfolio")

st.subheader("Summary (Test Window)")
st.dataframe(summary.style.format({
    "Cum Return":"{:.2%}", "Ann Vol":"{:.2%}", "Sharpe":"{:.2f}", "Max DD":"{:.2%}"
}))
st.caption("β>0 & significant → momentum; β<0 & significant → reversal; else → inconclusive.")
