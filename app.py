from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="SaleScope",
    page_icon=str(Path(__file__).resolve().parent / "favicon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def find_file(filename: str, folders: list[Path]) -> Path:
    for folder in folders:
        path = folder / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}")


MODEL_PATH = find_file(
    "best_model.joblib",
    [ROOT / "models", ROOT, Path("/content/models"), Path("/content")],
)
TRAIN_PATH = find_file(
    "train.csv",
    [ROOT / "data", ROOT, Path("/content/data"), Path("/content")],
)
TEST_PATH = find_file(
    "test.csv",
    [ROOT / "data", ROOT, Path("/content/data"), Path("/content")],
)

MODEL_NAME = "Random Forest"
VALIDATION_R2 = 0.878
VALIDATION_RMSE = 18.319
VALIDATION_MAE = 15.254

BG = "#F5EFE3"
SURFACE = "#FFF9F0"
TEXT = "#2B1B16"
MUTED = "#6B5043"
BROWN = "#4A2C20"
ACCENT = "#8A5A3B"
BORDER = "#DCCBBC"
SOFT = "#E8D8C3"


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
        :root {{
            --bg: {BG};
            --surface: {SURFACE};
            --text: {TEXT};
            --muted: {MUTED};
            --brown: {BROWN};
            --accent: {ACCENT};
            --border: {BORDER};
            --soft: {SOFT};
        }}

        .stApp {{
            background:
                radial-gradient(circle at 5% 12%, rgba(138,90,59,.10) 0 78px, transparent 79px),
                radial-gradient(circle at 96% 82%, rgba(74,44,32,.08) 0 105px, transparent 106px),
                radial-gradient(circle at 88% 9%, rgba(232,216,195,.65) 0 54px, transparent 55px),
                repeating-linear-gradient(135deg, rgba(74,44,32,.018) 0 1px, transparent 1px 14px),
                var(--bg);
            color: var(--text);
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .28;
            background-image:
                radial-gradient(rgba(43,27,22,.10) .7px, transparent .7px),
                radial-gradient(rgba(43,27,22,.06) .6px, transparent .6px);
            background-position: 0 0, 7px 7px;
            background-size: 14px 14px;
            mix-blend-mode: multiply;
        }}

        #MainMenu, footer {{ visibility: hidden; }}
        header {{ background: transparent !important; }}

        .block-container {{
            max-width: 1180px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}

        .brand {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: .9rem;
        }}

        .brand-title {{
            font-size: 1.9rem;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -.045em;
            color: var(--brown);
        }}

        .brand-subtitle {{
            margin-top: .28rem;
            color: var(--muted);
            font-size: .83rem;
        }}

        .brand-mark {{
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: 1.5px solid var(--brown);
            display: grid;
            place-items: center;
            color: var(--brown);
            font-weight: 800;
            background: rgba(255,249,240,.65);
        }}

        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {{
            border-radius: 18px;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(255,249,240,.84);
            border: 1px solid rgba(74,44,32,.14) !important;
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(74,44,32,.06);
        }}


        .links {{ display: flex; gap: .5rem; }}
        .link-btn {{
            display: inline-flex; align-items: center; gap: .45rem;
            padding: .42rem .8rem; border-radius: 999px;
            border: 1.5px solid var(--brown); color: var(--brown) !important;
            background: rgba(255,249,240,.7); font-size: .78rem; font-weight: 750;
            text-decoration: none !important;
            transition: transform .18s ease, background .18s ease, color .18s ease, box-shadow .18s ease;
        }}
        .link-btn:hover {{
            background: var(--brown); color: #FFF9F0 !important;
            transform: translateY(-2px); box-shadow: 0 6px 14px rgba(74,44,32,.22);
        }}

        @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
        @keyframes pop {{ 0% {{ transform: scale(.85); opacity: 0; }} 60% {{ transform: scale(1.04); }} 100% {{ transform: scale(1); opacity: 1; }} }}
        @keyframes floaty {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-4px); }} }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ animation: fadeUp .6s ease both; transition: transform .2s ease, box-shadow .2s ease; }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{ transform: translateY(-2px); box-shadow: 0 14px 32px rgba(74,44,32,.11); }}
        .brand-title {{ animation: fadeUp .5s ease both; }}
        .stitched {{ animation: fadeUp .6s .1s ease both; }}
        .chip {{ animation: pop .5s .3s ease both; }}
        .gauge {{ margin-top: 1rem; }}
        .gauge-track {{ position: relative; height: 10px; border-radius: 99px; background: var(--soft); overflow: hidden; }}
        .gauge-fill {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, #B48768, var(--brown)); transform-origin: left; animation: grow 1s .2s cubic-bezier(.2,.8,.2,1) both; }}
        @keyframes grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
        .gauge-scale {{ display: flex; justify-content: space-between; font-size: .66rem; color: var(--muted); margin-top: .3rem; }}
        .trend-badge {{ display: inline-block; animation: floaty 2.4s ease-in-out infinite; }}

        .panel {{
            background: rgba(255,249,240,.84);
            border: 1px solid rgba(74,44,32,.14);
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(74,44,32,.06);
            padding: 1rem 1.05rem;
        }}

        .panel-label {{
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: .66rem;
            font-weight: 800;
            margin-bottom: .55rem;
        }}

        .stitched {{
            position: relative;
            border: 1.5px dashed var(--brown);
            border-radius: 16px;
            background: rgba(232,216,195,.42);
            padding: .62rem .85rem;
            margin: .85rem 0 1rem;
        }}

        .stitched::after {{
            content: "";
            position: absolute;
            inset: 4px;
            border: 1px dashed rgba(74,44,32,.22);
            border-radius: 12px;
            pointer-events: none;
        }}

        .stitched-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1.2fr .9fr .9fr .9fr;
            gap: .35rem;
            align-items: center;
        }}

        .stitched-label {{
            font-size: .64rem;
            text-transform: uppercase;
            letter-spacing: .1em;
            color: var(--muted);
            font-weight: 800;
        }}

        .stitched-value {{
            margin-top: .1rem;
            font-size: .92rem;
            font-weight: 800;
            color: var(--brown);
        }}

        .result {{
            padding: .35rem 0 .15rem;
        }}

        .result-number {{
            font-size: 3.35rem;
            line-height: .95;
            letter-spacing: -.05em;
            font-weight: 850;
            color: var(--brown);
        }}

        .result-unit {{
            color: var(--muted);
            font-size: .82rem;
            margin-top: .4rem;
        }}

        .result-meta {{
            display: flex;
            gap: .45rem;
            flex-wrap: wrap;
            margin-top: .8rem;
        }}

        .chip {{
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: .28rem .52rem;
            color: var(--brown);
            background: rgba(255,255,255,.44);
            font-size: .72rem;
            font-weight: 700;
        }}

        .insight {{
            color: var(--muted);
            font-size: .75rem;
            line-height: 1.45;
            margin-top: .8rem;
        }}

        div[data-baseweb="select"] > div {{
            border-color: var(--border) !important;
            background: rgba(255,249,240,.9) !important;
            border-radius: 11px !important;
        }}

        label, .stSelectbox label {{
            color: var(--brown) !important;
            font-size: .74rem !important;
            font-weight: 750 !important;
        }}

        .stDownloadButton button {{
            width: 100%;
            border-radius: 10px;
            border: 1px solid var(--brown);
            background: var(--brown);
            color: #FFFFFF;
            font-weight: 750;
        }}

        .stDownloadButton button:hover {{
            border-color: var(--accent);
            background: var(--accent);
            color: #FFFFFF;
        }}

        .tiny-note {{
            color: var(--muted);
            font-size: .68rem;
            margin-top: .25rem;
        }}

        .section-title {{
            color: var(--brown);
            font-size: .9rem;
            font-weight: 800;
            letter-spacing: -.01em;
            margin-bottom: .45rem;
        }}

        @media (max-width: 800px) {{
            .stitched-grid {{ grid-template-columns: 1fr 1fr; }}
            .result-number {{ font-size: 2.8rem; }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data and model
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    train["store"] = train["store"].astype(str)
    train["item"] = train["item"].astype(str)
    test["store"] = test["store"].astype(str)
    test["item"] = test["item"].astype(str)

    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["quarter"] = train["date"].dt.quarter

    monthly = (
        train.groupby(["year", "month", "quarter", "store", "item"], as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "monthly_demand"})
    )

    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
    )

    start_year = monthly["year"].min()
    monthly["time_index"] = (
        (monthly["year"] - start_year) * 12 + monthly["month"]
    )
    monthly = monthly.sort_values(["date", "store", "item"]).reset_index(drop=True)

    test["year"] = test["date"].dt.year
    test["month"] = test["date"].dt.month
    test["quarter"] = test["date"].dt.quarter
    test["time_index"] = (
        (test["year"] - start_year) * 12 + test["month"]
    )

    # Any month through 2027 can be forecast: the model only needs year/month/time_index.
    # ponytail: Random Forest can't extrapolate trend past the training range, so far-out months flatten.
    grid = pd.MultiIndex.from_product(
        [pd.date_range("2018-01-01", "2027-12-01", freq="MS"), sorted(train["store"].unique()), sorted(train["item"].unique())],
        names=["date", "store", "item"],
    ).to_frame(index=False)
    grid["year"] = grid["date"].dt.year
    grid["month"] = grid["date"].dt.month
    grid["quarter"] = grid["date"].dt.quarter
    grid["time_index"] = (grid["year"] - start_year) * 12 + grid["month"]
    future = grid[["date", "year", "month", "quarter", "time_index", "store", "item"]]

    return train, test, monthly, future


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


train_raw, test_raw, monthly_df, future_df = load_data()
model = load_model()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand">
        <div>
            <div class="brand-title">SaleScope</div>
            <div class="brand-subtitle">Monthly store & product demand forecasting</div>
        </div>
        <div class="links">
            <a class="link-btn" href="https://github.com/varadisthedev/SaleScope" target="_blank" rel="noopener">
                <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
                GitHub
            </a>
            <a class="link-btn" href="https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data" target="_blank" rel="noopener">
                <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.285.18.046.149.034.255-.036.315l-6.555 6.344 6.836 8.507c.095.104.117.208.07.358"/></svg>
                Kaggle dataset
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Forecast controls
# -----------------------------------------------------------------------------
st.markdown('<div class="panel-label">Forecast</div>', unsafe_allow_html=True)
control_cols = st.columns([1, 1, 1.2])

stores = sorted(monthly_df["store"].unique(), key=lambda x: int(x))
items = sorted(monthly_df["item"].unique(), key=lambda x: int(x))
months = sorted(future_df["date"].dt.to_period("M").astype(str).unique())

with control_cols[0]:
    selected_store = st.selectbox("Store", stores, format_func=lambda x: f"Store {x}")
with control_cols[1]:
    selected_item = st.selectbox("Product", items, format_func=lambda x: f"Item {x}")
with control_cols[2]:
    selected_month = st.selectbox("Forecast month", months)


# -----------------------------------------------------------------------------
# Best model strip
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="stitched">
        <div class="stitched-grid">
            <div>
                <div class="stitched-label">Best model</div>
                <div class="stitched-value">{MODEL_NAME}</div>
            </div>
            <div>
                <div class="stitched-label">Validation R²</div>
                <div class="stitched-value">{VALIDATION_R2:.1%}</div>
            </div>
            <div>
                <div class="stitched-label">RMSE</div>
                <div class="stitched-value">{VALIDATION_RMSE:.2f} units</div>
            </div>
            <div>
                <div class="stitched-label">MAE</div>
                <div class="stitched-value">{VALIDATION_MAE:.2f} units</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Selected forecast
# -----------------------------------------------------------------------------
selected_future = future_df[
    (future_df["store"] == selected_store)
    & (future_df["item"] == selected_item)
    & (future_df["date"].dt.to_period("M").astype(str) == selected_month)
].copy()

prediction = float(model.predict(selected_future[[
    "year", "month", "quarter", "time_index", "store", "item"
]])[0])
prediction = max(0, prediction)
predicted_units = int(round(prediction))

history = monthly_df[
    (monthly_df["store"] == selected_store) & (monthly_df["item"] == selected_item)
].sort_values("date").copy()

future_selected = future_df[
    (future_df["store"] == selected_store) & (future_df["item"] == selected_item)
].copy()
future_selected["predicted_monthly_demand"] = np.maximum(
    0,
    model.predict(future_selected[[
        "year", "month", "quarter", "time_index", "store", "item"
    ]])
)

previous_value = None
prior_pred = future_selected[future_selected["date"] < selected_future["date"].iloc[0]]
prior = history[history["date"] < selected_future["date"].iloc[0]]
if not prior_pred.empty:
    previous_value = float(prior_pred.iloc[-1]["predicted_monthly_demand"])
elif not prior.empty:
    previous_value = float(prior.iloc[-1]["monthly_demand"])

change_text = "—"
if previous_value is not None and previous_value != 0:
    change = (predicted_units - previous_value) / previous_value
    change_text = f"{change:+.1%}"


left, right = st.columns([0.84, 1.55], gap="large")

with left, st.container(border=True):
    st.markdown('<div class="panel-label">Selected forecast</div>', unsafe_allow_html=True)
    uid = f"{selected_store}_{selected_item}_{selected_month}".replace("-", "_")
    h_min, h_avg, h_max = history["monthly_demand"].min(), history["monthly_demand"].mean(), history["monthly_demand"].max()
    span = max(h_max - h_min, 1)
    fill = min(max((predicted_units - h_min) / span, 0.03), 1.0)
    arrow = "▲" if change_text.startswith("+") else ("▼" if change_text.startswith("-") else "•")
    st.markdown(
        f"""
        <style>
            @property --n {{ syntax: "<integer>"; initial-value: 0; inherits: false; }}
            @keyframes count_{uid} {{ from {{ --n: 0; }} to {{ --n: {predicted_units}; }} }}
            .result-number.anim_{uid} {{ animation: count_{uid} 1s ease-out both; counter-reset: num var(--n); }}
            .result-number.anim_{uid}::after {{ content: counter(num); }}
            .gauge-fill.anim_{uid} {{ width: {fill*100:.1f}%; }}
        </style>
        <div class="result">
            <div class="result-number anim_{uid}"></div>
            <div class="result-unit">predicted units · {selected_month} · Store {selected_store} · Item {selected_item}</div>
            <div class="result-meta">
                <span class="chip"><span class="trend-badge">{arrow}</span>&nbsp;vs previous: {change_text}</span>
                <span class="chip">monthly demand</span>
            </div>
            <div class="gauge">
                <div class="gauge-track"><div class="gauge-fill anim_{uid}"></div></div>
                <div class="gauge-scale"><span>Low {h_min:,.0f}</span><span>Avg {h_avg:,.0f}</span><span>Peak {h_max:,.0f}</span></div>
            </div>
            <div class="insight">
                Where this forecast sits within the store-product pair's historical monthly range.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("_last_sel") != uid:
        if "_last_sel" in st.session_state:
            st.toast(f"New forecast: {predicted_units:,} units", icon="📈")
        st.session_state["_last_sel"] = uid

with right, st.container(border=True):
    st.markdown('<div class="section-title">Demand trend</div>', unsafe_allow_html=True)

    chart_history = history[["date", "monthly_demand"]].copy()
    chart_future = future_selected[
        future_selected["date"] <= selected_future["date"].iloc[0]
    ][["date", "predicted_monthly_demand"]].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_history["date"],
            y=chart_history["monthly_demand"],
            mode="lines+markers",
            name="Historical",
            line=dict(color=BROWN, width=2.5),
            marker=dict(size=5, color=BROWN),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_future["date"],
            y=chart_future["predicted_monthly_demand"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=ACCENT, width=2.5, dash="dash"),
            marker=dict(size=6, color=ACCENT),
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        xaxis=dict(showgrid=False, title=None, tickformat="%b %Y"),
        yaxis=dict(showgrid=True, gridcolor="rgba(74,44,32,.10)", title=None),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# -----------------------------------------------------------------------------
# Month overview
# -----------------------------------------------------------------------------
month_forecast = future_df[
    future_df["date"].dt.to_period("M").astype(str) == selected_month
].copy()
month_forecast["predicted_units"] = np.maximum(
    0,
    model.predict(month_forecast[[
        "year", "month", "quarter", "time_index", "store", "item"
    ]]),
)
month_forecast["predicted_units"] = month_forecast["predicted_units"].round().astype(int)

store_totals = (
    month_forecast.groupby("store", as_index=False)["predicted_units"]
    .sum()
    .sort_values("predicted_units", ascending=False)
)
item_totals = (
    month_forecast.groupby("item", as_index=False)["predicted_units"]
    .sum()
    .sort_values("predicted_units", ascending=False)
)

top_items = item_totals["item"].head(10).tolist()
heatmap = (
    month_forecast[month_forecast["item"].isin(top_items)]
    .pivot(index="item", columns="store", values="predicted_units")
    .reindex(top_items)
    .sort_index(axis=1, key=lambda c: c.astype(int))
)

bottom_left, bottom_right = st.columns([1.05, 1.45], gap="large")

with bottom_left, st.container(border=True):
    st.markdown(f'<div class="section-title">Demand map · top 10 products · {selected_month}</div>', unsafe_allow_html=True)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap.values,
            x=[f"S{x}" for x in heatmap.columns],
            y=[f"Item {x}" for x in heatmap.index],
            text=heatmap.values,
            texttemplate="%{text}",
            textfont=dict(size=10),
            xgap=2,
            ygap=2,
            colorscale=[[0, "#F0E3D4"], [0.5, "#B48768"], [1, BROWN]],
            hovertemplate="%{y} · %{x}<br>%{z:,.0f} units<extra></extra>",
            showscale=False,
        )
    )
    fig_heat.update_layout(
        height=300,
        margin=dict(l=4, r=4, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(showgrid=False, side="top", fixedrange=True),
        yaxis=dict(showgrid=False, autorange="reversed", fixedrange=True),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    top_store = store_totals.iloc[0]
    top_item = item_totals.iloc[0]
    st.markdown(
        f'<div class="tiny-note">Highest store demand: Store {top_store["store"]} · {int(top_store["predicted_units"]):,} units &nbsp;|&nbsp; Highest item demand: Item {top_item["item"]} · {int(top_item["predicted_units"]):,} units</div>',
        unsafe_allow_html=True,
    )

with bottom_right, st.container(border=True):
    st.markdown(f'<div class="section-title">Top store-product forecasts · {selected_month}</div>', unsafe_allow_html=True)

    display_df = (
        month_forecast[["store", "item", "predicted_units"]]
        .sort_values("predicted_units", ascending=False)
        .head(10)
        .rename(columns={
            "store": "Store",
            "item": "Product",
            "predicted_units": "Predicted units",
        })
    )
    display_df["Store"] = display_df["Store"].map(lambda x: f"Store {x}")
    display_df["Product"] = display_df["Product"].map(lambda x: f"Item {x}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "Predicted units": st.column_config.NumberColumn(format="%d"),
        },
    )

    download_df = month_forecast[["date", "store", "item", "predicted_units"]].copy()
    download_df["date"] = download_df["date"].dt.strftime("%Y-%m-%d")
    download_df = download_df.rename(columns={
        "date": "Date",
        "store": "Store",
        "item": "Product",
        "predicted_units": "Predicted_Units",
    })

    st.download_button(
        "Download month forecast",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name=f"salescope_{selected_month}.csv",
        mime="text/csv",
    )


st.markdown(
    '<div class="tiny-note" style="text-align:center; margin-top:1rem;">SaleScope · Random Forest regression · R² is used as the validation model-fit score.</div>',
    unsafe_allow_html=True,
)
