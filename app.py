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
    /* Page background + default text */
    .stApp {
        background-color: #f5f6fa;  /* light gray */
        color: #111;                /* dark text */
    }
    /* Sidebar white */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
    /* Make headings slightly darker */
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
    w = (1 + rets.fillna(0)).cumprod()
    return w / w.iloc[0]

def preview(df: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    return df.head(n) if len(df) > n else df

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
fwin = rets.loc[formation_start:formation_end]
twin = rets.loc[TEST_START:TEST_END]

if rets.shape[1] < 50:
    st.warning(f"Only {rets.shape[1]} tickers have complete data in both windows. Results may be noisy.")

formation = fwin.apply(compute_cum)
test      = twin.apply(compute_cum)

top, bot, ranks = partition_groups(formation, group_mode)
if len(top) == 0 or len(bot) == 0:
    st.error("Grouping produced an empty portfolio. Try a different lookback.")
    st.stop()

# -------------------- TITLE --------------------
st.title("Momentum vs Reversal — S&P 500 (Point-in-Time) Q4’24–Q1’25")
st.caption("Data source: per-ticker CSVs in repo. Formation ranking uses selected lookback; "
           "test window is Oct 1, 2024 → Mar 31, 2025.")

# -------------------- CUMULATIVE (Top/Bottom/SPY) --------------------
twin2 = rets.loc[TEST_START:TEST_END]
avail = set(twin2.columns)
top_use = [t for t in list(top) if t in avail]
bot_use = [t for t in list(bot) if t in avail]
if len(top_use) == 0 or len(bot_use) == 0:
    st.error("Selected groups are empty after the coverage filter.")
    st.stop()

g_top = twin2[top_use].mean(axis=1)
g_bot = twin2[bot_use].mean(axis=1)

curves = {}
curves["Top"]    = (1 + g_top.fillna(0)).cumprod()
curves["Bottom"] = (1 + g_bot.fillna(0)).cumprod()

dates_index = curves["Top"].index
spy_raw = load_spy_series(dates_index.min().strftime("%Y-%m-%d"),
                          dates_index.max().strftime("%Y-%m-%d"))
spy_aligned = spy_raw.reindex(dates_index).ffill()
spy_rets = spy_aligned.pct_change().fillna(0)
curves["SPY"] = (1 + spy_rets).cumprod()

cum_df = pd.DataFrame(curves).dropna()
cum_df = cum_df / cum_df.iloc[0] * INITIAL_INV

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=cum_df.index, y=cum_df["Top"],    name="Top",
                          line=dict(width=3, color=COLOR_TOP),
                          hovertemplate='Date=%{x|%b %d, %Y}<br>Value ($)=%{y:$,.2f}<extra></extra>'))
fig1.add_trace(go.Scatter(x=cum_df.index, y=cum_df["Bottom"], name="Bottom",
                          line=dict(width=3, color=COLOR_BOTTOM),
                          hovertemplate='Date=%{x|%b %d, %Y}<br>Value ($)=%{y:$,.2f}<extra></extra>'))
fig1.add_trace(go.Scatter(x=cum_df.index, y=cum_df["SPY"],    name="SPY",
                          line=dict(width=2, color=COLOR_SPY, dash="dot"),
                          hovertemplate='Date=%{x|%b %d, %Y}<br>Value ($)=%{y:$,.2f}<extra></extra>'))
fig1.update_layout(
    title=(f"Cumulative Portfolio Value — {group_mode}  "
           f"(Formation {formation_start.date()}→{formation_end.date()})"),
    xaxis_title="Date", yaxis_title="Portfolio Value ($)",
    template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
)
fig1.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig1, use_container_width=True)

# -------------------- LONG–SHORT & SHORT–LONG --------------------
ls = g_top - g_bot
sl = g_bot - g_top
ls_w = (1 + ls.fillna(0)).cumprod(); ls_w = ls_w / ls_w.iloc[0] * INITIAL_INV
sl_w = (1 + sl.fillna(0)).cumprod(); sl_w = sl_w / sl_w.iloc[0] * INITIAL_INV

fig_ls = go.Figure()
fig_ls.add_trace(go.Scatter(x=ls_w.index, y=ls_w.values, mode="lines",
                            name="Long Top / Short Bottom",
                            line=dict(width=3, color=COLOR_TOP),
                            hovertemplate='Date=%{x|%b %d, %Y}<br>Value ($)=%{y:$,.2f}<extra></extra>'))
fig_ls.add_trace(go.Scatter(x=sl_w.index, y=sl_w.values, mode="lines",
                            name="Long Bottom / Short Top",
                            line=dict(width=3, color=COLOR_BOTTOM),
                            hovertemplate='Date=%{x|%b %d, %Y}<br>Value ($)=%{y:$,.2f}<extra></extra>'))
fig_ls.update_layout(
    title=f"Long–Short vs Short–Long ({group_mode})",
    xaxis_title="Date", yaxis_title="Value ($)",
    template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
)
fig_ls.update_yaxes(tickprefix="$", separatethousands=True)
st.plotly_chart(fig_ls, use_container_width=True)

# -------------------- DECILE BAR --------------------
deciles = _make_deciles(ranks)
decile_df = pd.DataFrame({"formation": formation, "test": test, "decile": deciles})
avg_future = decile_df.groupby("decile", as_index=False)["test"].mean()

fig2 = px.bar(avg_future, x="decile", y="test",
              title="Average Test Return by Formation Decile",
              labels={"test":"Avg Test Return"}, template="plotly_white")
fig2.update_yaxes(tickformat=".2%")
ymin, ymax = float(avg_future["test"].min()), float(avg_future["test"].max())
pad = max(0.01, 0.12 * (ymax - ymin))
fig2.update_yaxes(range=[ymin - pad, ymax + pad])
fig2.update_traces(text=avg_future["test"].map("{:.2%}".format),
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
    # Group points
    for label, color in [(top_name, COLOR_TOP), (bot_name, COLOR_BOTTOM)]:
        d = df_sel[df_sel["group"] == label]
        fig3.add_trace(go.Scatter(
            x=d["formation"], y=d["test"], mode="markers", name=label,
            marker=dict(size=7, opacity=0.8, color=color),
            customdata=np.stack([d["Ticker"].values], axis=-1),
            hovertemplate=(
                "Ticker: %{customdata[0]}<br>"
                "Formation (In-Sample) Return: %{x:.2%}<br>"
                "Test (Out-of-Sample) Return: %{y:.2%}<extra></extra>"
            )
        ))
    fig3.update_layout(
        title=f"Cross-Section (Selected Groups Only): {top_name} vs {bot_name}",
        xaxis_title="Formation Return", yaxis_title="Test Return",
        template="plotly_white", plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG
    )
    fig3.update_xaxes(tickformat=".2%")
    fig3.update_yaxes(tickformat=".2%")
    st.plotly_chart(fig3, use_container_width=True)

# -------------------- SUMMARY TABLE --------------------
def _stats(s):
    ann_vol, sharpe, mdd = port_stats(s.dropna())
    return compute_cum(s), ann_vol, sharpe, mdd

ls = g_top - g_bot
sl = g_bot - g_top
cum_top, av_top, sh_top, dd_top   = _stats(g_top)
cum_bot, av_bot, sh_bot, dd_bot   = _stats(g_bot)
cum_ls,  av_ls,  sh_ls,  dd_ls    = _stats(ls)
cum_sl,  av_sl,  sh_sl,  dd_sl    = _stats(sl)

summary = pd.DataFrame([
    ["Top",         cum_top, av_top, sh_top, dd_top],
    ["Bottom",      cum_bot, av_bot, sh_bot, dd_bot],
    ["Long–Short",  cum_ls,  av_ls,  sh_ls,  dd_ls ],
    ["Short–Long",  cum_sl,  av_sl,  sh_sl,  dd_sl]
], columns=["Portfolio", "Cumulative Return", "Annualized Vol", "Sharpe", "Max Drawdown"]).set_index("Portfolio")

st.subheader("Summary (Test Window)")
st.dataframe(summary.style.format({
    "Cumulative Return":"{:.2%}", "Annualized Vol":"{:.2%}", "Sharpe":"{:.2f}", "Max Drawdown":"{:.2%}"
}))
st.caption("β>0 & significant (p<0.05) → momentum; β<0 & significant → reversal; else → inconclusive.")

# -------------------- AUDIT MODE (all intermediates + ZIP) --------------------
if audit_on:
    audit = {}

    # 1) Sources & shapes
    audit["prices_all (loaded)"] = prices_all
    audit["rets_all (pct_change)"] = rets_all
    audit["fwin (formation rets)"] = fwin
    audit["twin (test rets)"] = twin

    # 2) Universe filters
    keep_mask = fwin_all.notna().all() & twin_all.notna().all()
    audit["keep_mask (complete both windows)"] = keep_mask.to_frame("keep")
    audit["kept_cols (index)"] = pd.DataFrame({"ticker": cols})

    # 3) Formation stats & ranks
    audit["formation_cumret"] = formation.sort_values(ascending=False).to_frame("formation_cumret")
    audit["ranks_0to1"] = ranks.to_frame("rank_pct")
    audit["deciles_1to10"] = _make_deciles(ranks).to_frame("decile")

    # 4) Test stats
    audit["test_cumret"] = test.sort_values(ascending=False).to_frame("test_cumret")

    # 5) Group membership
    top_flag = pd.Series(False, index=cols); top_flag.loc[list(top)] = True
    bot_flag = pd.Series(False, index=cols); bot_flag.loc[list(bot)] = True
    audit["group_flags"] = pd.DataFrame({"top_grp": top_flag, "bot_grp": bot_flag})

    # 6) In-test daily portfolio series
    audit["g_top_daily_rets"] = g_top.to_frame("g_top")
    audit["g_bot_daily_rets"] = g_bot.to_frame("g_bot")
    audit["ls_daily_rets"] = (g_top - g_bot).to_frame("long_short")
    audit["sl_daily_rets"] = (g_bot - g_top).to_frame("short_long")

    audit["g_top_cum_curve"] = _cumcurve(g_top).to_frame("g_top_curve")
    audit["g_bot_cum_curve"] = _cumcurve(g_bot).to_frame("g_bot_curve")
    audit["ls_cum_curve"]    = _cumcurve(g_top - g_bot).to_frame("ls_curve")
    audit["sl_cum_curve"]    = _cumcurve(g_bot - g_top).to_frame("sl_curve")

    # 7) SPY aligned + returns used
    audit["spy_aligned_px"] = spy_aligned.to_frame("SPY_px")
    audit["spy_daily_rets"] = spy_rets.to_frame("SPY_ret")

    # 8) Regression inputs/outputs (selected groups only)
    audit["regression_inputs"] = pd.DataFrame({
        "formation": df_sel.set_index("Ticker")["formation"] if not df_sel.empty else pd.Series(dtype=float),
        "test": df_sel.set_index("Ticker")["test"] if not df_sel.empty else pd.Series(dtype=float),
        "group": df_sel.set_index("Ticker")["group"] if not df_sel.empty else pd.Series(dtype=object),
    })

    st.subheader("Audit Tables")
    for name, df in audit.items():
        with st.expander(f"{name}  —  shape {getattr(df, 'shape', None)}"):
            st.dataframe(preview(df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)))

    if reg is not None:
        st.subheader("OLS Summary (text)")
        st.code(reg.summary().as_text())

    if offer_downloads:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, df in audit.items():
                fname = name.replace(" ", "_").replace("/", "-") + ".csv"
                tmp = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
                zf.writestr(fname, tmp.to_csv(index=True))
            zf.writestr("summary_metrics.csv", summary.to_csv(index=True))
            if reg is not None:
                zf.writestr("ols_summary.txt", reg.summary().as_text())
        st.download_button(
            "⬇️ Download all audit CSVs (ZIP)",
            data=buf.getvalue(),
            file_name="audit_exports.zip",
            mime="application/zip"
        )
