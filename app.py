# app.py (Option 2 – light, professional theme)
import os
import io
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# -------------------- THEME --------------------
st.set_page_config(page_title="SPX Momentum vs Reversal (Q4'24–Q1'25)", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f6fa; color: #111; }
    section[data-testid="stSidebar"] { background-color: #ffffff; }
    h1, h2, h3, h4, h5 { color: #111; }
    </style>
    """,
    unsafe_allow_html=True
)

# Chart background color (white)
PLOT_BG = "#ffffff"

# Brand line colors
COLOR_TOP = "#1f77b4"     # blue
COLOR_BOTTOM = "#d62728"  # red
COLOR_SPY = "#111111"     # black/dark gray

# -------------------- CONFIG --------------------
PRICES_DIR = "Necessary_CSVs"   # per-ticker CSVs with columns: Date, AdjClose (or Close)
TEST_START = pd.Timestamp("2024-10-01")
TEST_END   = pd.Timestamp("2025-03-31")
INITIAL_INV = 10000

# -------------------- LOADERS --------------------
@st.cache_data(show_spinner=False)
def list_available_tickers():
    files = [f for f in os.listdir(PRICES_DIR) if f.endswith(".csv")]
    return sorted({os.path.splitext(f)[0] for f in files})

@st.cache_data(show_spinner=False)
def load_prices_from_folder(tickers):
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
    ranks = formation.rank(pct=True)
    if mode == "Decile (Top 10% vs Bottom 10%)":
        top = ranks[ranks >= 0.9].index
        bot = ranks[ranks <= 0.1].index
    elif mode == "Quartile (Top 25% vs Bottom 25%)":
        top = ranks[ranks >= 0.75].index
        bot = ranks[ranks <= 0.25].index
    else:  # Half
        med = formation.median()
        top = formation[formation >= med].index
        bot = formation[formation <  med].index
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

def _make_deciles(ranks: pd.Series) -> pd.Series:
    return (np.ceil(ranks * 10)).astype(int).clip(upper=10)

def _cumcurve(rets: pd.Series) -> pd.Series:
    return (1 + rets.fillna(0)).cumprod()

def preview(df: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    return df.head(n) if len(df) > n else df

# ---- BUY-AND-HOLD HELPERS (no daily rebalancing) ----
def buyhold_value(prices: pd.DataFrame, notional: float) -> pd.Series:
    """
    Equal-weight buy & hold portfolio value from raw prices (no rebalancing).
    prices: DataFrame indexed by date, columns=tickers (aligned, non-empty)
    notional: starting dollars invested in this portfolio
    """
    if prices.empty or prices.shape[1] == 0:
        return pd.Series(dtype=float)
    p0 = prices.iloc[0]
    shares = (notional / prices.shape[1]) / p0
    return (prices @ shares)

def short_value(prices: pd.DataFrame, notional: float) -> pd.Series:
    """
    Short-leg equity value if you short equal-weight at t0 and hold.
    Short proceeds are kept as cash (0% carry).
    Equity_t = notional + Σ shares * (p0 - p_t)
    """
    if prices.empty or prices.shape[1] == 0:
        return pd.Series(dtype=float)
    p0 = prices.iloc[0]
    shares = (notional / prices.shape[1]) / p0  # shares shorted
    pnl = (shares * (p0 - prices)).sum(axis=1)
    return notional + pnl

def series_returns_from_value(V: pd.Series) -> pd.Series:
    return V.pct_change().fillna(0)

# -------------------- SIDEBAR (Run + Audit) --------------------
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
    audit_on = st.checkbox("Audit mode: show intermediate tables", value=False)
    offer_downloads = st.checkbox("Include ZIP of CSVs", value=True)
    run = st.form_submit_button("Run")

if not run:
    st.title("Momentum vs Reversal — S&P 500 (Point-in-Time) Q4’24–Q1’25")
    st.info("Select your lookback & grouping in the sidebar, then click **Run**.")
    st.stop()

formation_end   = TEST_START - pd.Timedelta(days=1)
formation_start = TEST_START - pd.DateOffset(months=LOOKBACK_MONTHS)
st.sidebar.write(f"Formation: **{formation_start.date()} → {formation_end.date()}**")
st.sidebar.write(f"Test: **{TEST_START.date()} → {TEST_END.date()}**")

# -------------------- LOAD & FILTER DATA (robust) --------------------
all_files = list_available_tickers()
if not all_files:
    st.error(f"No CSVs found in `{PRICES_DIR}/`.")
    st.stop()

EXCLUDE = {"SPY"}
tickers = [t for t in all_files if t not in EXCLUDE]

prices_all = load_prices_from_folder(tickers)
if prices_all.empty:
    st.error("Failed to load any prices from the CSVs.")
    st.stop()

rets_all = prices_all.pct_change()
fwin_all = rets_all.loc[formation_start:formation_end]
twin_all = rets_all.loc[TEST_START:TEST_END]

if fwin_all.empty or twin_all.empty:
    st.error("No trading days in the selected formation or test window.")
    st.stop()

keep = fwin_all.notna().all() & twin_all.notna().all()
cols = keep.index[keep]

rets = rets_all.loc[:, cols]

fwin = rets.loc[formation_start:formation_end, cols]
twin = rets.loc[TEST_START:TEST_END, cols]

px_form_all = prices_all.loc[fwin.index, cols]
px_test_all = prices_all.loc[twin.index, cols]

px_test_all = px_test_all.dropna(axis=1, how="any")    # should already be clean from your keep-filter
px_test_all = px_test_all.loc[:, (px_test_all.iloc[0] > 0)]

formation_bh = (px_form_all.iloc[-1] / px_form_all.iloc[0] - 1).rename("formation_bh")
test_bh      = (px_test_all.iloc[-1] / px_test_all.iloc[0] - 1).rename("test_bh")


if rets.shape[1] < 50:
    st.warning(f"Only {rets.shape[1]} tickers have complete data in both windows. Results may be noisy.")

formation = formation_bh
test      = test_bh

top, bot, ranks = partition_groups(formation, group_mode)
if len(top) == 0 or len(bot) == 0:
    st.error("Grouping produced an empty portfolio. Try a different lookback.")
    st.stop()

# -------------------- TITLE --------------------
st.title("Momentum vs Reversal — S&P 500 (Point-in-Time) Q4’24–Q1’25")
st.caption("Data source: per-ticker CSVs in repo. Formation ranking uses selected lookback; "
           "test window is Oct 1, 2024 → Mar 31, 2025.")

# -------------------- BUILD TEST-WINDOW PRICES FOR GROUPS (with anchor day) --------------------
# Find the anchor day = last trading day before TEST_START, so the first included return is anchor->TEST_START
idx = prices_all.index.searchsorted(TEST_START)
anchor_date = prices_all.index[max(0, idx - 1)]

# Extended test window: start from anchor day, then trim to TEST_START for plotting/metrics
px_test_all_ext = prices_all.loc[anchor_date:TEST_END, cols]

# Decide group membership first
avail = set(px_test_all_ext.columns)
top_use = [t for t in list(top) if t in avail]
bot_use = [t for t in list(bot) if t in avail]
if len(top_use) == 0 or len(bot_use) == 0:
    st.error("Selected groups are empty after the coverage filter.")
    st.stop()

px_top_ext = px_test_all_ext[top_use]
px_bot_ext = px_test_all_ext[bot_use]

# -------------------- CUMULATIVE (Top/Bottom/SPY) — BUY & HOLD with anchor day --------------------
# Compute shares using anchor day prices (no daily rebalancing)
p0_top = px_top_ext.iloc[0]
p0_bot = px_bot_ext.iloc[0]
shares_top_long = (INITIAL_INV / len(p0_top)) / p0_top
shares_bot_long = (INITIAL_INV / len(p0_bot)) / p0_bot

# Long-only value paths (start at anchor day), then trim to TEST_START
V_top_ext = px_top_ext @ shares_top_long
V_bot_ext = px_bot_ext @ shares_bot_long

V_top = V_top_ext.loc[TEST_START:]
V_bot = V_bot_ext.loc[TEST_START:]

# SPY with the same anchor logic
spy_raw = load_spy_series(anchor_date.strftime("%Y-%m-%d"), TEST_END.strftime("%Y-%m-%d"))
spy_px_ext = spy_raw.reindex(px_test_all_ext.index).ffill()
V_spy_ext = (spy_px_ext / spy_px_ext.iloc[0]) * INITIAL_INV
V_spy = V_spy_ext.loc[TEST_START:]

# Data for the main cumulative chart
cum_df = pd.DataFrame({"Top": V_top, "Bottom": V_bot, "SPY": V_spy}).dropna()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cum_df.index, y=cum_df["Top"],    name="Top",
                          line=dict(width=3, color=COLOR_TOP)))
fig1.add_trace(go.Scatter(x=cum_df.index, y=cum_df["Bottom"], name="Bottom",
                          line=dict(width=3, color=COLOR_BOTTOM)))
fig1.add_trace(go.Scatter(x=cum_df.index, y=cum_df["SPY"],    name="SPY",
                          line=dict(width=2, color=COLOR_SPY, dash="dot")))
fig1.update_layout(
    title=(f"Cumulative Portfolio Value — Buy & Hold  {group_mode}  "
           f"(Formation {formation_start.date()}→{formation_end.date()})"),
    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
)
fig1.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig1, use_container_width=True)

# -------------------- LONG–SHORT & SHORT–LONG — BUY & HOLD (anchor day) --------------------
half = INITIAL_INV / 2.0

# Long Top (half)
shares_top_long_half = (half / len(p0_top)) / p0_top
V_long_top_ext = px_top_ext @ shares_top_long_half

# Short Bottom (half): equity value = initial cash + Σ shares*(p0 - p_t)
shares_bot_short_half = (half / len(p0_bot)) / p0_bot
pnl_short_bot_ext = (shares_bot_short_half * (p0_bot - px_bot_ext)).sum(axis=1)
V_short_bot_ext = half + pnl_short_bot_ext

# LS value path from anchor, then trim
V_ls = (V_long_top_ext + V_short_bot_ext).loc[TEST_START:]

# Long Bottom (half)
shares_bot_long_half = (half / len(p0_bot)) / p0_bot
V_long_bot_ext = px_bot_ext @ shares_bot_long_half

# Short Top (half)
shares_top_short_half = (half / len(p0_top)) / p0_top
pnl_short_top_ext = (shares_top_short_half * (p0_top - px_top_ext)).sum(axis=1)
V_short_top_ext = half + pnl_short_top_ext

# SL value path trimmed
V_sl = (V_long_bot_ext + V_short_top_ext).loc[TEST_START:]

fig_ls = go.Figure()
fig_ls.add_trace(go.Scatter(x=V_ls.index, y=V_ls.values, name="Long Top / Short Bottom",
                            line=dict(width=3, color=COLOR_TOP)))
fig_ls.add_trace(go.Scatter(x=V_sl.index, y=V_sl.values, name="Long Bottom / Short Top",
                            line=dict(width=3, color=COLOR_BOTTOM)))
fig_ls.update_layout(
    title=f"Long–Short vs Short–Long — Buy & Hold ({group_mode})",
    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
)
fig_ls.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig_ls, use_container_width=True)

# -------------------- DECILE BAR (compounded from buy&hold value paths) --------------------
deciles_full = _make_deciles(ranks)  # 1..10 per ticker

decile_rows = []
decile_cum_paths = {}

for d, members_idx in deciles_full.groupby(deciles_full).groups.items():
    members = [t for t in members_idx if t in px_test_all.columns]
    if not members:
        continue
    V_dec = buyhold_value(px_test_all[members], INITIAL_INV)
    total_ret = V_dec.iloc[-1] / V_dec.iloc[0] - 1.0
    decile_rows.append({"decile": int(d), "test_total_return": float(total_ret)})
    decile_cum_paths[int(d)] = (V_dec / V_dec.iloc[0]).rename(f"decile_{int(d)}_wealth")

avg_future = pd.DataFrame(decile_rows).sort_values("decile")

fig2 = px.bar(
    avg_future, x="decile", y="test_total_return",
    title="Compounded Test Return by Formation Decile (Buy & Hold)",
    labels={"test_total_return": "Total Return"},
    template="plotly_white"
)
fig2.update_yaxes(tickformat=".2%")
ymin, ymax = float(avg_future["test_total_return"].min()), float(avg_future["test_total_return"].max())
pad = max(0.01, 0.12 * (ymax - ymin))
fig2.update_yaxes(range=[ymin - pad, ymax + pad])
fig2.update_traces(text=avg_future["test_total_return"].map("{:.2%}".format),
                   textposition="outside", cliponaxis=False)
fig2.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, margin=dict(t=80, b=60, l=70, r=40))
st.plotly_chart(fig2, use_container_width=True)

# -------------------- SCATTER + REGRESSION + CI (selected groups only) --------------------
if "Decile" in group_mode:
    top_name, bot_name = "Top 10%", "Bottom 10%"
elif "Quartile" in group_mode:
    top_name, bot_name = "Top 25%", "Bottom 25%"
else:
    top_name, bot_name = "Top 50%", "Bottom 50%"

sel_idx = top_use + bot_use
formation_sel = formation.loc[sel_idx]
test_sel      = test.loc[sel_idx]
group_flag = pd.Series(index=sel_idx, dtype=object)
group_flag.loc[top_use] = top_name
group_flag.loc[bot_use] = bot_name

df_sel = pd.DataFrame({
    "Ticker": sel_idx,
    "formation": formation_sel.values,
    "test": test_sel.values,
    "group": group_flag.values
}).dropna()

reg = None
if df_sel.empty or df_sel["formation"].nunique() < 2:
    st.warning("Not enough variation in the selected groups to plot regression.")
else:
    X = pd.DataFrame({"const": 1.0, "formation": df_sel["formation"].values},
                     index=df_sel["Ticker"])
    y = pd.Series(df_sel["test"].values, index=df_sel["Ticker"])
    reg = sm.OLS(y, X, missing="drop").fit()

    beta = float(reg.params["formation"])
    tval = float(reg.tvalues["formation"])
    pval = float(reg.pvalues["formation"])

    xgrid = np.linspace(df_sel["formation"].min(), df_sel["formation"].max(), 200)
    Xg = pd.DataFrame({"const": 1.0, "formation": xgrid})
    pred = reg.get_prediction(Xg).summary_frame(alpha=0.05)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=np.hstack([xgrid, xgrid[::-1]]),
        y=np.hstack([pred["mean_ci_lower"], pred["mean_ci_upper"][::-1]]),
        fill='toself', showlegend=False, fillcolor="rgba(31, 119, 180, 0.12)", line=dict(width=0)
    ))
    fig3.add_trace(go.Scatter(
        x=xgrid, y=pred["mean"], mode="lines",
        name=f"Fit β={beta:.2f} (t={tval:.2f}, p={pval:.3g})",
        line=dict(color="#111", dash="dash"),
        hovertemplate='Formation: %{x:.2%}<br>Ŷ: %{y:.2%}<extra></extra>'
    ))
    for label, color in [(top_name, COLOR_TOP), (bot_name, COLOR_BOTTOM)]:
        d = df_sel[df_sel["group"] == label]
        fig3.add_trace(go.Scatter(
            x=d["formation"], y=d["test"], mode="markers", name=label,
            marker=dict(size=7, opacity=0.8, color=color),
            customdata=np.stack([d["Ticker"].values], axis=-1),
            hovertemplate=("Ticker: %{customdata[0]}<br>"
                           "Formation (In-Sample) Return: %{x:.2%}<br>"
                           "Test (Out-Of-Sample) Return: %{y:.2%}<extra></extra>")
        ))
    fig3.update_layout(
        title=f"Cross-Section (Selected Groups Only): {top_name} vs {bot_name}",
        xaxis_title="Formation Return", yaxis_title="Test Return",
        template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
    )
    fig3.update_xaxes(tickformat=".2%")
    fig3.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- SUMMARY TABLE — BUY & HOLD --------------------
ret_top = series_returns_from_value(V_top)
ret_bot = series_returns_from_value(V_bot)
ret_ls  = series_returns_from_value(V_ls)
ret_sl  = series_returns_from_value(V_sl)

def _stats_from_rets(s):
    ann_vol, sharpe, mdd = port_stats(s.dropna())
    cum = (1 + s).prod() - 1
    return cum, ann_vol, sharpe, mdd

cum_top, av_top, sh_top, dd_top = _stats_from_rets(ret_top)
cum_bot, av_bot, sh_bot, dd_bot = _stats_from_rets(ret_bot)
cum_ls,  av_ls,  sh_ls,  dd_ls  = _stats_from_rets(ret_ls)
cum_sl,  av_sl,  sh_sl,  dd_sl  = _stats_from_rets(ret_sl)

summary = pd.DataFrame([
    ["Top (Buy&Hold)",         cum_top, av_top, sh_top, dd_top],
    ["Bottom (Buy&Hold)",      cum_bot, av_bot, sh_bot, dd_bot],
    ["Long–Short (Buy&Hold)",  cum_ls,  av_ls,  sh_ls,  dd_ls ],
    ["Short–Long (Buy&Hold)",  cum_sl,  av_sl,  sh_sl,  dd_sl]
], columns=["Portfolio", "Cumulative Return", "Annualized Vol", "Sharpe", "Max Drawdown"]).set_index("Portfolio")

st.subheader("Summary (Test Window) — Buy & Hold")
st.dataframe(summary.style.format({
    "Cumulative Return":"{:.2%}", "Annualized Vol":"{:.2%}", "Sharpe":"{:.2f}", "Max Drawdown":"{:.2%}"
}))

# -------------------- AUDIT MODE (dates/prices + group splits + ZIP) --------------------
if audit_on:
    audit = {}

    # ---- Window boundary dates (intended vs used) + boundary prices ----
    used_formation_start = fwin.index.min() if not fwin.empty else pd.NaT
    used_formation_end   = fwin.index.max() if not fwin.empty else pd.NaT
    used_test_start      = twin.index.min() if not twin.empty else pd.NaT
    used_test_end        = twin.index.max() if not twin.empty else pd.NaT

    intended_formation_start = (TEST_START - pd.DateOffset(months=LOOKBACK_MONTHS)).normalize()
    intended_formation_end   = (TEST_START - pd.Timedelta(days=1)).normalize()
    intended_test_start      = TEST_START.normalize()
    intended_test_end        = TEST_END.normalize()

    window_boundaries = pd.DataFrame({
        "anchor": ["formation_start","formation_end","test_start","test_end"],
        "intended_calendar_date": [
            intended_formation_start, intended_formation_end,
            intended_test_start, intended_test_end
        ],
        "used_trading_date": [
            used_formation_start, used_formation_end,
            used_test_start, used_test_end
        ]
    })

    anchors = ["formation_start","formation_end","test_start","test_end"]
    anchor_dates = [used_formation_start, used_formation_end, used_test_start, used_test_end]
    boundary_prices = prices_all[cols].reindex(anchor_dates)
    boundary_prices.index = anchors

    boundary_prices_long = (
        boundary_prices.reset_index(names="anchor")
                       .melt(id_vars="anchor", var_name="ticker", value_name="price")
    )

    with st.expander(f"window_boundaries  —  shape {window_boundaries.shape}"):
        st.dataframe(window_boundaries)
    with st.expander(f"boundary_prices (anchors × tickers)  —  shape {boundary_prices.shape}"):
        st.dataframe(boundary_prices)
    with st.expander(f"boundary_prices_long (tidy)  —  shape {boundary_prices_long.shape}"):
        st.dataframe(boundary_prices_long)

    audit["window_boundaries"] = window_boundaries
    audit["boundary_prices_wide"] = boundary_prices
    audit["boundary_prices_long"] = boundary_prices_long

    # ---- Group membership + per-group prices/returns (formation & test) ----
    group_label = pd.Series("Neither", index=cols, dtype=object)
    group_label.loc[top_use] = "Top"
    group_label.loc[bot_use] = "Bottom"

    group_membership = pd.DataFrame({
        "ticker": cols,
        "rank_pct": ranks.reindex(cols).values,
        "decile": _make_deciles(ranks).reindex(cols).values,
        "formation_cumret": formation.reindex(cols).values,
        "test_cumret": test.reindex(cols).values,
        "group": group_label.values
    }).sort_values(["group","rank_pct"], ascending=[True, False]).reset_index(drop=True)

    price_formation_all = prices_all.loc[fwin.index, cols]
    price_test_all      = prices_all.loc[twin.index, cols]

    top_prices_formation = price_formation_all[top_use]
    bot_prices_formation = price_formation_all[bot_use]
    top_prices_test      = price_test_all[top_use]
    bot_prices_test      = price_test_all[bot_use]

    ret_formation_all = fwin
    ret_test_all      = twin
    top_returns_formation = ret_formation_all[top_use]
    bot_returns_formation = ret_formation_all[bot_use]
    top_returns_test      = ret_test_all[top_use]
    bot_returns_test      = ret_test_all[bot_use]

    with st.expander(f"group_membership  —  shape {group_membership.shape}"):
        st.dataframe(preview(group_membership, n=2000))
    with st.expander(f"top_prices_formation  —  shape {top_prices_formation.shape}"):
        st.dataframe(preview(top_prices_formation))
    with st.expander(f"bottom_prices_formation  —  shape {bot_prices_formation.shape}"):
        st.dataframe(preview(bot_prices_formation))
    with st.expander(f"top_prices_test  —  shape {top_prices_test.shape}"):
        st.dataframe(preview(top_prices_test))
    with st.expander(f"bottom_prices_test  —  shape {bot_prices_test.shape}"):
        st.dataframe(preview(bot_prices_test))

    p0_top = px_top_ext.iloc[0] if not px_top_ext.empty else pd.Series(dtype=float)
    p0_bot = px_bot_ext.iloc[0] if not px_bot_ext.empty else pd.Series(dtype=float)
    shares_top_long = ((INITIAL_INV) / max(1, len(p0_top))) / p0_top if not p0_top.empty else pd.Series(dtype=float)
    shares_bot_long = ((INITIAL_INV) / max(1, len(p0_bot))) / p0_bot if not p0_bot.empty else pd.Series(dtype=float)

    short_half = INITIAL_INV / 2.0
    shares_bot_short = (short_half / max(1, len(p0_bot))) / p0_bot if not p0_bot.empty else pd.Series(dtype=float)
    shares_top_short = (short_half / max(1, len(p0_top))) / p0_top if not p0_top.empty else pd.Series(dtype=float)

    audit["group_membership"]          = group_membership
    audit["buyhold_top_value"]         = V_top.to_frame(name="V_top")
    audit["buyhold_bottom_value"]      = V_bot.to_frame(name="V_bottom")
    audit["longshort_value"]           = V_ls.to_frame(name="V_LS")
    audit["shortlong_value"]           = V_sl.to_frame(name="V_SL")
    audit["buyhold_top_initial_shares"] = shares_top_long.rename("shares_long_per_ticker").to_frame()
    audit["buyhold_bottom_initial_shares"] = shares_bot_long.rename("shares_long_per_ticker").to_frame()
    audit["short_bot_initial_shares"]  = shares_bot_short.rename("shares_short_per_ticker").to_frame()
    audit["short_top_initial_shares"]  = shares_top_short.rename("shares_short_per_ticker").to_frame()

    audit["top_prices_formation"]      = top_prices_formation
    audit["bottom_prices_formation"]   = bot_prices_formation
    audit["top_prices_test"]           = top_prices_test
    audit["bottom_prices_test"]        = bot_prices_test
    audit["top_returns_formation"]     = top_returns_formation
    audit["bottom_returns_formation"]  = bot_returns_formation
    audit["top_returns_test"]          = top_returns_test
    audit["bottom_returns_test"]       = bot_returns_test

    # Include final summary + (optional) OLS summary
    audit["summary_metrics"] = summary
    if reg is not None:
        ols_text = reg.summary().as_text()
    else:
        ols_text = None

    # ZIP everything
    if offer_downloads:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, df in audit.items():
                if isinstance(df, pd.DataFrame):
                    fname = name.replace(" ", "_").replace("/", "-") + ".csv"
                    zf.writestr(fname, df.to_csv(index=True))
            # Also stash the per-decile cum wealth paths
            if decile_cum_paths:
                decile_df = pd.DataFrame(decile_cum_paths).sort_index()
                zf.writestr("decile_wealth_paths.csv", decile_df.to_csv(index=True))
            if ols_text is not None:
                zf.writestr("ols_summary.txt", ols_text)
        st.download_button(
            "⬇️ Download all audit CSVs (ZIP)",
            data=buf.getvalue(),
            file_name="audit_exports.zip",
            mime="application/zip"
        )
