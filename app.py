# app.py — S&P 500 Constituents: Momentum vs Mean Reversion (lean)
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf

# ================ UI / THEME ================
st.set_page_config(page_title="S&P 500 Constituents: Momentum vs Mean Reversion", layout="wide")
st.markdown("""
<style>
  .stApp { background:#f5f6fa; color:#111; }
  section[data-testid="stSidebar"] { background:#fff; }
  h1, h2, h3, h4, h5 { color:#111; }
</style>
""", unsafe_allow_html=True)

PLOT_BG = "#ffffff"
COLOR_TOP, COLOR_BOT, COLOR_SPY = "#1f77b4", "#d62728", "#111111"

# ================ CONFIG ================
PRICES_DIR  = "Necessary_CSVs"  # CSVs: Date, AdjClose (or Close)
TEST_START  = pd.Timestamp("2024-10-01")
TEST_END    = pd.Timestamp("2025-03-31")
INITIAL_INV = 10_000

# ================ LOADERS ================
@st.cache_data(show_spinner=False)
def list_available_tickers() -> list[str]:
    files = [f for f in os.listdir(PRICES_DIR) if f.endswith(".csv")]
    return sorted({os.path.splitext(f)[0] for f in files})

@st.cache_data(show_spinner=False)
def load_prices_from_folder(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for t in tickers:
        path = os.path.join(PRICES_DIR, f"{t}.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path, parse_dates=["Date"])
        col = "AdjClose" if "AdjClose" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None: continue
        frames.append(df[["Date", col]].rename(columns={col: t}).set_index("Date"))
    if not frames: return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "Date"
    return out

@st.cache_data(show_spinner=False)
def load_spy_series(start_date: str, end_date: str) -> pd.Series:
    local = os.path.join(PRICES_DIR, "SPY.csv")
    start = pd.to_datetime(start_date) - pd.Timedelta(days=5)
    end   = pd.to_datetime(end_date)   + pd.Timedelta(days=5)
    if os.path.exists(local):
        df = pd.read_csv(local, parse_dates=["Date"])
        col = "AdjClose" if "AdjClose" in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None: raise ValueError("SPY.csv must have 'AdjClose' or 'Close'.")
        return df.set_index("Date")[col].rename("SPY").loc[start:end]
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False, threads=False)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].rename("SPY")

# ================ UTILS ================
def partition_groups(formation: pd.Series, mode: str):
    ranks = formation.rank(pct=True)
    m = (mode or "").lower()
    if "10%" in m or "decile" in m:
        top = ranks[ranks >= 0.90].index; bot = ranks[ranks <= 0.10].index
    elif "25%" in m or "quartile" in m:
        top = ranks[ranks >= 0.75].index; bot = ranks[ranks <= 0.25].index
    else:
        med = formation.median()
        top = formation[formation >= med].index; bot = formation[formation < med].index
        if len(top) == 0 or len(bot) == 0:
            r2 = formation.rank(pct=True, method="average")
            top = r2[r2 >= 0.5].index; bot = r2[r2 < 0.5].index
    return list(top), list(bot), ranks

def make_deciles(ranks: pd.Series) -> pd.Series:
    return (np.ceil(ranks * 10)).astype(int).clip(upper=10)

def buyhold_value(prices: pd.DataFrame, notional: float) -> pd.Series:
    if prices.empty: return pd.Series(dtype=float)
    p0 = prices.iloc[0]; shares = (notional / prices.shape[1]) / p0
    return prices @ shares

def port_stats(series: pd.Series):
    series = series.dropna()
    if series.empty: return np.nan, np.nan, np.nan
    std = series.std()
    if std == 0 or np.isnan(std): return np.nan, np.nan, np.nan
    ann_vol = std * np.sqrt(252); sharpe = (series.mean() / std) * np.sqrt(252)
    wealth = (1 + series).cumprod()
    mdd = (wealth / wealth.cummax() - 1).min()
    return ann_vol, sharpe, mdd

# Fixed-shares helpers
def _equal_weight_shares(p0: pd.Series, notional: float) -> pd.Series:
    return (notional / len(p0)) / p0

def _long_leg_value(px_ext: pd.DataFrame, p0: pd.Series, notional: float) -> pd.Series:
    return px_ext @ _equal_weight_shares(p0, notional)

def _short_leg_value(px_ext: pd.DataFrame, p0: pd.Series, notional: float) -> pd.Series:
    sh = _equal_weight_shares(p0, notional)
    pnl = (sh * (p0 - px_ext)).sum(axis=1)
    return notional + pnl

# ================ SIDEBAR ================
with st.sidebar.form("controls"):
    st.header("Parameters")
    choice = st.selectbox("Formation lookback", ["1 month", "3 months", "6 months", "12 months"], index=1)
    LOOKBACK_MONTHS = {"1 month":1, "3 months":3, "6 months":6, "12 months":12}[choice]
    group_mode = st.selectbox("Grouping", ["Top 10% vs Bottom 10%","Top 25% vs Bottom 25%","Top 50% vs Bottom 50%"], index=0)
    run = st.form_submit_button("Run")

if not run:
    st.title("S&P 500 Constituents: Momentum vs Mean Reversion")
    st.info("Pick your lookback & grouping, then click **Run**.")
    st.stop()

formation_end   = TEST_START - pd.Timedelta(days=1)
formation_start = TEST_START - pd.DateOffset(months=LOOKBACK_MONTHS)
st.sidebar.write(f"In-sample: **{formation_start.date()} → {formation_end.date()}**")
st.sidebar.write(f"Out-of-sample: **{TEST_START.date()} → {TEST_END.date()}**")

# ================ LOAD & FILTER ================
all_files = list_available_tickers()
if not all_files: st.error(f"No CSVs in `{PRICES_DIR}/`."); st.stop()

tickers = [t for t in all_files if t != "SPY"]
prices_all = load_prices_from_folder(tickers)
if prices_all.empty: st.error("Failed to load any prices from CSVs."); st.stop()

rets_all = prices_all.pct_change()
fwin_all = rets_all.loc[formation_start:formation_end]
twin_all = rets_all.loc[TEST_START:TEST_END]
if fwin_all.empty or twin_all.empty: st.error("No trading days in the formation or test window."); st.stop()

keep = fwin_all.notna().all() & twin_all.notna().all()
cols = keep.index[keep]
if len(cols) == 0: st.error("No tickers with complete data in both windows."); st.stop()
if len(cols) < 50: st.warning(f"Only {len(cols)} tickers have complete data. Results may be noisy.")

# Formation/Test returns (buy & hold over the window)
px_form_all = prices_all.loc[fwin_all.index, cols]
px_test_all = prices_all.loc[twin_all.index, cols].dropna(axis=1, how="any")
px_test_all = px_test_all.loc[:, (px_test_all.iloc[0] > 0)]
formation = (px_form_all.iloc[-1] / px_form_all.iloc[0] - 1)
test      = (px_test_all.iloc[-1] / px_test_all.iloc[0] - 1)

top, bot, ranks = partition_groups(formation, group_mode)

# ================ TITLE ================
st.title("S&P 500 Constituents: Momentum vs Mean Reversion")
st.caption("Formation (in-sample) uses your lookback; test (out-of-sample) is Oct 1, 2024 → Mar 28, 2025.")
st.caption("Created by Dylan Sturdevant for Morgan Stanley exercise")

# ================ ANCHOR & GROUP PRICES ================
idx = prices_all.index.searchsorted(TEST_START)
anchor_date = prices_all.index[max(0, idx - 1)]
px_test_all_ext = prices_all.loc[anchor_date:TEST_END, cols]

top_use = [t for t in top if t in px_test_all_ext.columns]
bot_use = [t for t in bot if t in px_test_all_ext.columns]
if not top_use or not bot_use: st.error("Selected groups are empty after coverage filter."); st.stop()

px_top_ext = px_test_all_ext[top_use]
px_bot_ext = px_test_all_ext[bot_use]
p0_top, p0_bot = px_top_ext.iloc[0], px_bot_ext.iloc[0]

# ================ BUY & HOLD (normalized to $10k) ================
V_top_ext = buyhold_value(px_top_ext, INITIAL_INV); V_top_ext = V_top_ext / V_top_ext.iloc[0] * INITIAL_INV
V_bot_ext = buyhold_value(px_bot_ext, INITIAL_INV); V_bot_ext = V_bot_ext / V_bot_ext.iloc[0] * INITIAL_INV
V_top = V_top_ext.loc[TEST_START:]; V_bot = V_bot_ext.loc[TEST_START:]

spy_raw = load_spy_series(anchor_date.strftime("%Y-%m-%d"), TEST_END.strftime("%Y-%m-%d"))
spy_px_ext = spy_raw.reindex(px_test_all_ext.index).ffill()
V_spy_ext = (spy_px_ext / spy_px_ext.iloc[0]) * INITIAL_INV
V_spy = V_spy_ext.loc[TEST_START:]

# ================ LONG–SHORT / SHORT–LONG (fixed shares, no rebal) ================
half = INITIAL_INV / 2.0
V_ls_ext = _long_leg_value(px_top_ext, p0_top, half) + _short_leg_value(px_bot_ext, p0_bot, half)
V_sl_ext = _long_leg_value(px_bot_ext, p0_bot, half) + _short_leg_value(px_top_ext, p0_top, half)
V_ls_ext = V_ls_ext / V_ls_ext.iloc[0] * INITIAL_INV
V_sl_ext = V_sl_ext / V_sl_ext.iloc[0] * INITIAL_INV
V_ls, V_sl = V_ls_ext.loc[TEST_START:], V_sl_ext.loc[TEST_START:]

# ================ CHARTS ================
# B&H Top/Bottom/SPY
cum_df = pd.DataFrame({"Top": V_top, "Bottom": V_bot, "SPY": V_spy}).dropna(how="all")
fig1 = go.Figure()
for name, color in [("Top", COLOR_TOP), ("Bottom", COLOR_BOT), ("SPY", COLOR_SPY)]:
    fig1.add_trace(go.Scatter(
        x=cum_df.index, y=cum_df[name], name=name,
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{name}: $%{{y:,.2f}}<extra></extra>",
        line=dict(width=3 if name!="SPY" else 2, color=color, dash="dot" if name=="SPY" else None)
    ))
fig1.update_layout(
    title=f"Cumulative Portfolio Value: {group_mode} (in-sample {formation_start.date()} → {formation_end.date()})",
    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
    hovermode="x unified", hoverlabel=dict(namelength=-1)
)
fig1.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig1, use_container_width=True)

# LS vs SL
fig_ls = go.Figure()
fig_ls.add_trace(go.Scatter(x=V_ls.index, y=V_ls.values, name="Long Top / Short Bottom (Momentum)",
                            hovertemplate="%{x|%Y-%m-%d}<br>LS: $%{y:,.2f}<extra></extra>",
                            line=dict(width=3, color=COLOR_TOP)))
fig_ls.add_trace(go.Scatter(x=V_sl.index, y=V_sl.values, name="Long Bottom / Short Top (Mean Reversion)",
                            hovertemplate="%{x|%Y-%m-%d}<br>SL: $%{y:,.2f}<extra></extra>",
                            line=dict(width=3, color=COLOR_BOT)))
fig_ls.update_layout(
    title=f"Long–Short vs Short–Long: {group_mode} (in-sample {formation_start.date()} → {formation_end.date()})",
    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
    hovermode="x unified"
)
fig_ls.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig_ls, use_container_width=True)

# Decile bar (OOS)
deciles_full = make_deciles(ranks)
rows = []
for d, members_idx in deciles_full.groupby(deciles_full).groups.items():
    members = [t for t in members_idx if t in px_test_all_ext.columns]
    if not members: continue
    v_ext = buyhold_value(px_test_all_ext[members], INITIAL_INV)
    v_ext = v_ext / v_ext.iloc[0] * INITIAL_INV
    rows.append({"decile": int(d), "test_total_return": float(v_ext.iloc[-1] / INITIAL_INV - 1.0)})
avg_future = pd.DataFrame(rows).sort_values("decile")
fig2 = px.bar(avg_future, x="decile", y="test_total_return",
              title="Compounded (out-of-sample) Return by Decile", labels={"test_total_return":"Total Return"},
              template="plotly_white")
fig2.update_yaxes(tickformat=".2%")
if not avg_future.empty:
    ymin, ymax = float(avg_future["test_total_return"].min()), float(avg_future["test_total_return"].max())
    pad = max(0.01, 0.12 * (ymax - ymin))
    fig2.update_yaxes(range=[ymin - pad, ymax + pad])
    fig2.update_traces(text=avg_future["test_total_return"].map("{:.2%}".format),
                       textposition="outside", cliponaxis=False)
fig2.update_layout(plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG, margin=dict(t=80, b=60, l=70, r=40))
st.plotly_chart(fig2, use_container_width=True)

# ================ CROSS-SECTION REGRESSION ================
gm = (group_mode or "").lower()
if "10%" in gm or "decile" in gm:  top_name, bot_name = "Top 10%", "Bottom 10%"
elif "25%" in gm or "quartile" in gm: top_name, bot_name = "Top 25%", "Bottom 25%"
else: top_name, bot_name = "Top 50%", "Bottom 50%"

sel_idx = top_use + bot_use
df_sel = pd.DataFrame({
    "Ticker": sel_idx,
    "formation": formation.reindex(sel_idx).values,
    "test":      test.reindex(sel_idx).values,
    "group":     ([top_name]*len(top_use)) + ([bot_name]*len(bot_use))
}).dropna()

if df_sel.empty or df_sel["formation"].nunique() < 2:
    st.warning("Not enough variation to run the regression.")
else:
    X = pd.DataFrame({"const": 1.0, "formation": df_sel["formation"].values}, index=df_sel["Ticker"])
    y = pd.Series(df_sel["test"].values, index=df_sel["Ticker"])
    reg = sm.OLS(y, X, missing="drop").fit()

    beta = float(reg.params["formation"])
    tval = float(reg.tvalues["formation"])
    pval = float(reg.pvalues["formation"])

    xgrid = np.linspace(df_sel["formation"].min(), df_sel["formation"].max(), 200)
    pred  = reg.get_prediction(pd.DataFrame({"const": 1.0, "formation": xgrid})).summary_frame(alpha=0.05)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=np.hstack([xgrid, xgrid[::-1]]),
        y=np.hstack([pred["mean_ci_lower"], pred["mean_ci_upper"][::-1]]),
        fill='toself', showlegend=False, fillcolor="rgba(31,119,180,0.12)", line=dict(width=0),
        hoverinfo="skip"
    ))
    fig3.add_trace(go.Scatter(
        x=xgrid, y=pred["mean"], mode="lines",
        name=f"Predicted test (OOS) — β={beta:.2f} (t={tval:.2f}, p={pval:.3g})",
        line=dict(color="#111", dash="dash"),
        hovertemplate=("Formation (in-sample) return: %{x:.2%}<br>"
                       "Predicted test (out-of-sample) return: %{y:.2%}<extra></extra>")
    ))
    for label, color in [(top_name, COLOR_TOP), (bot_name, COLOR_BOT)]:
        d = df_sel[df_sel["group"] == label]
        fig3.add_trace(go.Scatter(
            x=d["formation"], y=d["test"], mode="markers", name=label,
            marker=dict(size=9, opacity=0.95, color=color),
            customdata=np.stack([d["Ticker"].values], axis=-1),
            hovertemplate=("Ticker: %{customdata[0]}<br>"
                           "Formation (in-sample) return: %{x:.2%}<br>"
                           "Test (out-of-sample) return: %{y:.2%}<extra></extra>")
        ))
    fig3.update_layout(
        title=f"Cross-Section: {top_name} vs {bot_name}",
        xaxis_title="Formation Return", yaxis_title="Test Return",
        template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
    )
    fig3.update_xaxes(tickformat=".2%"); fig3.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig3, use_container_width=True)

# ================ SUMMARY ================
def _stats_from_series(V_ext: pd.Series, V: pd.Series):
    ret = V.pct_change().fillna(0)
    av, sh, mdd = port_stats(ret)
    cum_anchor = V_ext.iloc[-1] / INITIAL_INV - 1.0
    return cum_anchor, av, sh, mdd

cum_top, av_top, sh_top, dd_top = _stats_from_series(V_top_ext, V_top)
cum_bot, av_bot, sh_bot, dd_bot = _stats_from_series(V_bot_ext, V_bot)
cum_ls,  av_ls,  sh_ls,  dd_ls  = _stats_from_series(V_ls_ext,  V_ls)
cum_sl,  av_sl,  sh_sl,  dd_sl  = _stats_from_series(V_sl_ext,  V_sl)

summary = pd.DataFrame([
    ["Top",        cum_top, av_top, sh_top, dd_top],
    ["Bottom",     cum_bot, av_bot, sh_bot, dd_bot],
    ["Long–Short", cum_ls,  av_ls,  sh_ls,  dd_ls],
    ["Short–Long", cum_sl,  av_sl,  sh_sl,  dd_sl]
], columns=["Portfolio", "Cumulative Return", "Annualized Vol", "Sharpe", "Max Drawdown"]).set_index("Portfolio")

st.subheader("Summary (Out-of-Sample Window)")
st.dataframe(summary.style.format({
    "Cumulative Return": "{:.2%}",
    "Annualized Vol":    "{:.2%}",
    "Sharpe":            "{:.2f}",
    "Max Drawdown":      "{:.2%}"
}))
