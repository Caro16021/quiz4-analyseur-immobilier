import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

try:
    from google import genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


load_dotenv()


DATA_FILE = "kc_house_data.csv"
LOGO_FILE = Path("assets/king-county-logo.svg")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CHART_COLORS = {
    "forest": "#0F5E67",
    "teal": "#1CC7C1",
    "sage": "#97D700",
    "sand": "#F4F7EE",
    "amber": "#B8E65A",
    "coral": "#C75C4D",
    "ink": "#114146",
}
NEIGHBORHOOD_CMAP = LinearSegmentedColormap.from_list(
    "turquoise_apple",
    ["#0F5E67", "#1CC7C1", "#97D700"],
)


st.set_page_config(
    page_title="Analyseur immobilier - Comte de King",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["zipcode"] = df["zipcode"].astype(str)
    df["floors"] = pd.to_numeric(df["floors"], errors="coerce")

    sale_year = df["date"].dt.year
    df["price_per_sqft"] = df["price"] / df["sqft_living"].replace(0, np.nan)
    df["age"] = sale_year - df["yr_built"]
    df["is_renovated"] = df["yr_renovated"].fillna(0).gt(0)
    df["has_basement"] = df["sqft_basement"].fillna(0).gt(0)
    df["neighborhood_avg_house_age"] = df.groupby("zipcode")["age"].transform("mean")

    return df


def fmt_money(value: float) -> str:
    return f"{value:,.0f} $".replace(",", " ")


def fmt_signed_money(value: float) -> str:
    return f"{value:+,.0f} $".replace(",", " ")


def fmt_pct(value: float) -> str:
    return f"{value:.1f} %"


def get_secret(name: str) -> Optional[str]:
    env_value = os.getenv(name)
    if env_value:
        return env_value

    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_data
def load_logo_svg(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(151, 215, 0, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(28, 199, 193, 0.2), transparent 24%),
                linear-gradient(180deg, #f4f7ee 0%, #eef6ef 45%, #f7fbf3 100%);
            color: #114146;
        }
        .main .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #114146 0%, #0f5e67 46%, #1cc7c1 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"] * {
            color: #f5efe2;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
            background: rgba(255, 248, 240, 0.08);
            border-radius: 14px;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] > div > div:first-child {
            background: rgba(255, 255, 255, 0.18) !important;
            border-radius: 999px !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] > div > div:nth-child(2) {
            background: #97D700 !important;
            border-radius: 999px !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] [style*="background-color: rgb(255, 75, 75)"] {
            background-color: #97D700 !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] [style*="background: rgb(255, 75, 75)"] {
            background: #97D700 !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] [style*="rgb(255, 75, 75)"] {
            border-color: #97D700 !important;
            background-color: #97D700 !important;
        }
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stSlider * {
            accent-color: #97D700 !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] div[role="slider"] {
            background-color: #97D700 !important;
            box-shadow: 0 0 0 0.22rem rgba(151, 215, 0, 0.28) !important;
            border-color: #eef6ef !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] div[role="slider"] > div {
            color: #f5efe2 !important;
        }
        section[data-testid="stSidebar"] div.stSlider div[data-baseweb="slider"] svg * {
            fill: #97D700 !important;
            stroke: #97D700 !important;
        }
        .hero-card {
            background:
                linear-gradient(135deg, rgba(17, 65, 70, 0.97) 0%, rgba(15, 94, 103, 0.94) 54%, rgba(151, 215, 0, 0.88) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 28px;
            padding: 2rem 2.2rem;
            color: #fff7ed;
            box-shadow: 0 20px 45px rgba(15, 94, 103, 0.16);
            margin-bottom: 1.4rem;
        }
        .hero-layout {
            display: grid;
            grid-template-columns: 118px minmax(0, 1fr);
            gap: 1.2rem;
            align-items: start;
        }
        .hero-mark {
            background: rgba(255, 251, 243, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 26px;
            padding: 0.8rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }
        .hero-mark svg {
            display: block;
            width: 100%;
            height: auto;
        }
        .hero-copy-group {
            min-width: 0;
        }
        .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.76rem;
            opacity: 0.8;
            margin-bottom: 0.55rem;
        }
        .hero-subline {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.75rem;
            background: rgba(255, 251, 243, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 999px;
            font-size: 0.86rem;
            margin-bottom: 0.9rem;
        }
        .hero-title {
            font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
            font-size: clamp(2rem, 4vw, 3.5rem);
            line-height: 1.05;
            margin: 0 0 0.6rem 0;
        }
        .hero-copy {
            max-width: 46rem;
            font-size: 1rem;
            line-height: 1.65;
            color: rgba(255, 247, 237, 0.88);
            margin-bottom: 1rem;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.2rem;
        }
        .hero-chip {
            background: rgba(255, 250, 242, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            backdrop-filter: blur(4px);
        }
        .hero-chip-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.72;
        }
        .hero-chip-value {
            display: block;
            margin-top: 0.3rem;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .sidebar-brand {
            display: grid;
            grid-template-columns: 52px minmax(0, 1fr);
            gap: 0.8rem;
            align-items: center;
            margin: 0.2rem 0 1rem 0;
            padding: 0.8rem 0.85rem;
            background: rgba(255, 249, 241, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
        }
        .sidebar-brand svg {
            width: 100%;
            height: auto;
            display: block;
        }
        .sidebar-brand-title {
            font-weight: 700;
            font-size: 0.98rem;
            line-height: 1.1;
        }
        .sidebar-brand-copy {
            font-size: 0.78rem;
            opacity: 0.8;
            margin-top: 0.15rem;
        }
        .section-lead {
            background: rgba(255, 253, 249, 0.78);
            border: 1px solid rgba(15, 94, 103, 0.08);
            border-radius: 20px;
            padding: 1rem 1.15rem;
            margin: 0.2rem 0 1rem 0;
            box-shadow: 0 12px 24px rgba(15, 94, 103, 0.05);
        }
        .section-lead strong {
            color: #0f5e67;
        }
        .property-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin: 0.25rem 0 1rem 0;
        }
        .property-badge {
            background: rgba(151, 215, 0, 0.14);
            color: #0f5e67;
            border: 1px solid rgba(151, 215, 0, 0.22);
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.92rem;
            font-weight: 600;
        }
        [data-testid="stMetric"] {
            background: rgba(255, 253, 249, 0.92);
            border: 1px solid rgba(15, 94, 103, 0.08);
            border-radius: 20px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 14px 26px rgba(15, 94, 103, 0.06);
        }
        [data-testid="stMetricLabel"] {
            color: #537979;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #114146;
        }
        .stButton > button {
            background: linear-gradient(135deg, #0f5e67 0%, #1cc7c1 100%);
            color: #fff9f0;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 10px 22px rgba(15, 94, 103, 0.18);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #0b4c53 0%, #17aea8 100%);
            color: #fffdf7;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.7rem;
            margin-bottom: 0.85rem;
        }
        div[data-baseweb="tab-list"] button {
            border-radius: 999px;
            background: rgba(255, 253, 249, 0.65);
            border: 1px solid rgba(15, 94, 103, 0.08);
            padding: 0.65rem 1rem;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: #0f5e67;
            color: #fff7ed;
        }
        div[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"] {
            border-radius: 18px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(15, 94, 103, 0.08);
            box-shadow: 0 12px 24px rgba(15, 94, 103, 0.05);
            background: rgba(255, 253, 249, 0.88);
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(15, 94, 103, 0.08);
            background: rgba(255, 253, 249, 0.85);
        }
        h1, h2, h3 {
            color: #114146;
            letter-spacing: -0.02em;
        }
        @media (max-width: 900px) {
            .hero-layout {
                grid-template-columns: 1fr;
            }
            .hero-mark {
                max-width: 120px;
            }
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(data: pd.DataFrame) -> None:
    llm_engine = "Gemini" if get_secret("GEMINI_API_KEY") else "OpenAI" if get_secret("OPENAI_API_KEY") else "LLM non configure"
    logo_svg = load_logo_svg(str(LOGO_FILE))
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-layout">
                <div class="hero-mark">{logo_svg}</div>
                <div class="hero-copy-group">
                    <div class="hero-kicker">King County Market Intelligence</div>
                    <div class="hero-subline">Seattle Region • Visualisation, comparables et notes d'analyste</div>
                    <div class="hero-title">Analyseur immobilier pour l'exploration du marche et l'evaluation d'actifs</div>
                    <div class="hero-copy">
                        Explorez les transactions du comte de King, identifiez les poches de valeur
                        et comparez rapidement une propriete a son marche local avec une lecture
                        visuelle plus proche d'une note d'analyste.
                    </div>
                </div>
            </div>
            <div class="hero-grid">
                <div class="hero-chip">
                    <span class="hero-chip-label">Transactions disponibles</span>
                    <span class="hero-chip-value">{len(data):,} ventes</span>
                </div>
                <div class="hero-chip">
                    <span class="hero-chip-label">Couverture geographique</span>
                    <span class="hero-chip-value">{data['zipcode'].nunique()} codes postaux</span>
                </div>
                <div class="hero-chip">
                    <span class="hero-chip-label">Moteur d'analyse</span>
                    <span class="hero-chip-value">{llm_engine}</span>
                </div>
            </div>
        </div>
        """.replace(",", " "),
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    logo_svg = load_logo_svg(str(LOGO_FILE))
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div>{logo_svg}</div>
            <div>
                <div class="sidebar-brand-title">King County Analyzer</div>
                <div class="sidebar-brand-copy">Tableau de bord immobilier interactif</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_lead(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="section-lead">
            <strong>{title}</strong><br>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_neighborhood_profile(df: pd.DataFrame, zipcode: str) -> dict:
    neighborhood = df[df["zipcode"] == zipcode].copy()
    return {
        "median_price": neighborhood["price"].median(),
        "median_price_per_sqft": neighborhood["price_per_sqft"].median(),
        "renovation_rate": neighborhood["is_renovated"].mean() * 100,
        "avg_house_age": neighborhood["age"].mean(),
        "transaction_count": len(neighborhood),
    }


def render_neighborhood_profile(df: pd.DataFrame, zipcode: str) -> None:
    profile = get_neighborhood_profile(df, zipcode)
    render_section_lead(
        "Profil du quartier",
        "Ce bloc synthétise le niveau de prix local et la profondeur du marché autour de la propriété sélectionnée.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix median quartier", fmt_money(profile["median_price"]))
    c2.metric("Prix median / pi2", fmt_money(profile["median_price_per_sqft"]))
    c3.metric("% renovees", fmt_pct(profile["renovation_rate"]))
    c4.metric("Transactions quartier", f"{int(profile['transaction_count']):,}".replace(",", " "))


def render_verdict_banner(status: str, gap_value: float, gap_pct: float) -> None:
    if gap_pct <= -5:
        badge_color = "#3FAF5F"
        verdict = "Decote attractive"
        explanation = "La propriete se positionne sous la moyenne des comparables, ce qui peut signaler une opportunite d'achat."
    elif gap_pct >= 5:
        badge_color = "#C75C4D"
        verdict = "Surcote visible"
        explanation = "La propriete se situe au-dessus des comparables et exige une lecture plus prudente du potentiel d'investissement."
    else:
        badge_color = "#1CC7C1"
        verdict = "Prix proche du marche"
        explanation = "Le positionnement est relativement aligne sur les ventes comparables du secteur."

    st.markdown(
        f"""
        <div style="
            background: rgba(255, 253, 249, 0.92);
            border: 1px solid rgba(15, 94, 103, 0.08);
            border-left: 8px solid {badge_color};
            border-radius: 22px;
            padding: 1rem 1.15rem;
            margin: 0.45rem 0 1rem 0;
            box-shadow: 0 12px 24px rgba(15, 94, 103, 0.05);
        ">
            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; color: #537979;">Verdict investissement</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #114146; margin-top: 0.15rem;">{verdict}</div>
            <div style="margin-top: 0.35rem; color: #315f63;">
                Ecart observe : <strong>{fmt_signed_money(gap_value)}</strong> ({gap_pct:+.1f}%).
                {explanation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_positioning_gauge(gap_pct: float) -> None:
    clipped_gap = max(min(gap_pct, 20), -20)
    left_fill = max(0.0, (20 + clipped_gap) / 40 * 100)
    st.markdown(
        f"""
        <div style="margin: 0.2rem 0 1rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#5b7387; margin-bottom:0.35rem;">
                <span>Decote</span>
                <span>Prix juste</span>
                <span>Surcote</span>
            </div>
            <div style="
                position: relative;
                height: 16px;
                border-radius: 999px;
                background: linear-gradient(90deg, #3FAF5F 0%, #9EE86C 48%, #1CC7C1 100%);
                box-shadow: inset 0 1px 3px rgba(15, 94, 103, 0.12);
            ">
                <div style="
                    position: absolute;
                    left: calc({left_fill}% - 12px);
                    top: -6px;
                    width: 24px;
                    height: 28px;
                    border-radius: 14px;
                    background: #0f5e67;
                    border: 3px solid #fff7ed;
                    box-shadow: 0 8px 18px rgba(15, 94, 103, 0.18);
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_property_badges(property_row: pd.Series) -> None:
    badges = [
        f"Code postal {property_row['zipcode']}",
        f"Prix / pi2 {fmt_money(property_row['price_per_sqft'])}",
        f"Condition {int(property_row['condition'])}/5",
        "Renovee" if property_row["is_renovated"] else "Non renovee",
        "Sous-sol present" if property_row["has_basement"] else "Sans sous-sol",
        "Front de mer" if property_row["waterfront"] == 1 else "Sans front de mer",
    ]
    chips = "".join(f'<span class="property-badge">{badge}</span>' for badge in badges)
    st.markdown(f'<div class="property-badges">{chips}</div>', unsafe_allow_html=True)


def style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12, fontsize=13, fontweight="bold", color=CHART_COLORS["ink"])
    ax.set_xlabel(xlabel, color=CHART_COLORS["ink"])
    ax.set_ylabel(ylabel, color=CHART_COLORS["ink"])
    ax.grid(axis="y", color="#d9e1db", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#395852")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9fb4ab")
    ax.spines["bottom"].set_color("#9fb4ab")


def build_market_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtres du marche")
    st.sidebar.caption("Affinez un segment de marche, puis lisez les indicateurs et graphiques mis a jour en temps reel.")

    price_range = st.sidebar.slider(
        "Fourchette de prix ($)",
        min_value=int(df["price"].min()),
        max_value=int(df["price"].max()),
        value=(int(df["price"].quantile(0.01)), int(df["price"].quantile(0.99))),
        step=10000,
    )
    bedroom_range = st.sidebar.slider(
        "Nombre de chambres",
        min_value=int(df["bedrooms"].min()),
        max_value=int(df["bedrooms"].max()),
        value=(int(df["bedrooms"].min()), int(df["bedrooms"].max())),
        step=1,
    )
    zipcode_values = st.sidebar.multiselect(
        "Code postal",
        options=sorted(df["zipcode"].unique().tolist()),
        default=[],
    )
    neighborhood_age_range = st.sidebar.slider(
        "Age moyen des maisons du quartier (ans)",
        min_value=float(df["neighborhood_avg_house_age"].min()),
        max_value=float(df["neighborhood_avg_house_age"].max()),
        value=(
            float(df["neighborhood_avg_house_age"].min()),
            float(df["neighborhood_avg_house_age"].max()),
        ),
        step=1.0,
    )
    year_range = st.sidebar.slider(
        "Annee de construction",
        min_value=int(df["yr_built"].min()),
        max_value=int(df["yr_built"].max()),
        value=(int(df["yr_built"].min()), int(df["yr_built"].max())),
        step=1,
    )
    waterfront_only = st.sidebar.checkbox("Front de mer uniquement", value=False)

    filtered = df[
        df["price"].between(*price_range)
        & df["bedrooms"].between(*bedroom_range)
        & df["neighborhood_avg_house_age"].between(*neighborhood_age_range)
        & df["yr_built"].between(*year_range)
    ].copy()

    if zipcode_values:
        filtered = filtered[filtered["zipcode"].isin(zipcode_values)]

    if waterfront_only:
        filtered = filtered[filtered["waterfront"] == 1]

    return filtered


def plot_price_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fffdf9")
    ax.set_facecolor("#fffdf9")
    ax.hist(df["price"], bins=30, color=CHART_COLORS["teal"], edgecolor="#fffaf2", alpha=0.95)
    ax.axvline(df["price"].median(), color=CHART_COLORS["amber"], linewidth=2.2, linestyle="--")
    style_axes(ax, "Distribution des prix", "Prix de vente ($)", "Nombre de proprietes")
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    return fig


def plot_price_vs_sqft(df: pd.DataFrame):
    sample = df.sample(min(len(df), 2500), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#fffdf9")
    ax.set_facecolor("#fffdf9")
    scatter = ax.scatter(
        sample["sqft_living"],
        sample["price"],
        c=sample["neighborhood_avg_house_age"],
        cmap=NEIGHBORHOOD_CMAP,
        alpha=0.72,
        edgecolors="white",
        linewidths=0.25,
    )
    style_axes(ax, "Prix vs superficie habitable", "Superficie habitable (pi2)", "Prix ($)")
    fig.colorbar(scatter, ax=ax, label="Age moyen du quartier")
    ax.ticklabel_format(style="plain", axis="both")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = [
        "price",
        "price_per_sqft",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "neighborhood_avg_house_age",
        "condition",
        "age",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#fffdf9")
    ax.set_facecolor("#fffdf9")
    im = ax.imshow(corr.values, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_title("Matrice de correlation", loc="left", pad=12, fontsize=13, fontweight="bold", color=CHART_COLORS["ink"])
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_avg_price_by_zipcode(df: pd.DataFrame):
    by_zipcode = (
        df.groupby("zipcode", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="#fffdf9")
    ax.set_facecolor("#fffdf9")
    ax.bar(by_zipcode["zipcode"], by_zipcode["price"], color=CHART_COLORS["amber"])
    style_axes(ax, "Prix moyen par code postal (Top 10)", "Code postal", "Prix moyen ($)")
    ax.ticklabel_format(style="plain", axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def call_llm(prompt: str) -> str:
    gemini_api_key = get_secret("GEMINI_API_KEY")
    if gemini_api_key:
        if genai is None:
            return "La dependance `google-genai` n'est pas installee dans l'environnement Python."

        try:
            client = genai.Client(api_key=gemini_api_key)
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as exc:
            return f"Echec de l'appel Gemini : {exc}"

    openai_api_key = get_secret("OPENAI_API_KEY")
    if openai_api_key:
        if OpenAI is None:
            return "La dependance `openai` n'est pas installee dans l'environnement Python."

        try:
            client = OpenAI(api_key=openai_api_key)
            response = client.responses.create(
                model=DEFAULT_OPENAI_MODEL,
                input=prompt,
            )
            return response.output_text.strip()
        except Exception as exc:
            return f"Echec de l'appel OpenAI : {exc}"

    return (
        "Configuration manquante : ajoutez `GEMINI_API_KEY` ou `OPENAI_API_KEY` "
        "dans le fichier `.env` ou dans les secrets Streamlit pour activer les resumes LLM."
    )


def build_market_prompt(df: pd.DataFrame) -> str:
    neighborhood_age_by_zipcode = (
        df.groupby("zipcode")["neighborhood_avg_house_age"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .round(1)
        .to_dict()
    )
    pct_waterfront = df["waterfront"].mean() * 100

    return f"""
Tu es un analyste immobilier senior. Voici les statistiques d'un segment
du marche immobilier du comte de King (Seattle) :

- Nombre de proprietes : {len(df)}
- Prix moyen : {df["price"].mean():,.0f} $
- Prix median : {df["price"].median():,.0f} $
- Prix min / max : {df["price"].min():,.0f} $ / {df["price"].max():,.0f} $
- Prix moyen par pi2 : {df["price_per_sqft"].mean():,.0f} $
- Age moyen des maisons : {df["age"].mean():.1f} ans
- Age moyen des maisons du quartier : {df["neighborhood_avg_house_age"].mean():.1f} ans
- % maisons renovees : {df["is_renovated"].mean() * 100:.1f}%
- % maisons avec sous-sol : {df["has_basement"].mean() * 100:.1f}%
- Quartiers les plus anciens (zipcode) : {neighborhood_age_by_zipcode}
- % front de mer : {pct_waterfront:.1f}%

Redige un resume executif de ce segment en 3-4 paragraphes.
Identifie les tendances cles et les opportunites d'investissement.
"""


def render_market_tab(df: pd.DataFrame) -> None:
    st.subheader("Exploration du marche")
    render_section_lead(
        "Lecture du segment filtre",
        "Utilisez la barre laterale pour isoler une zone du marche. Les KPI, graphiques et resume LLM se recalculent automatiquement sur le segment selectionne.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N proprietes", f"{len(df):,}".replace(",", " "))
    c2.metric("Prix moyen", fmt_money(df["price"].mean()))
    c3.metric("Prix median", fmt_money(df["price"].median()))
    c4.metric("Prix moyen / pi2", fmt_money(df["price_per_sqft"].mean()))

    summary_cols = st.columns(3)
    summary_cols[0].metric("% renovees", fmt_pct(df["is_renovated"].mean() * 100))
    summary_cols[1].metric("% avec sous-sol", fmt_pct(df["has_basement"].mean() * 100))
    summary_cols[2].metric("% front de mer", fmt_pct(df["waterfront"].mean() * 100))

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_price_histogram(df), clear_figure=True)
        st.pyplot(plot_correlation_heatmap(df), clear_figure=True)
    with right:
        st.pyplot(plot_price_vs_sqft(df), clear_figure=True)
        st.pyplot(plot_avg_price_by_zipcode(df), clear_figure=True)

    if st.button("Generer un resume du marche", key="market_summary"):
        with st.spinner("Generation du resume en cours..."):
            summary = call_llm(build_market_prompt(df))
        st.markdown("### Resume du marche")
        st.write(summary)


def select_property(df: pd.DataFrame) -> Optional[pd.Series]:
    st.subheader("Selection de la propriete")
    render_section_lead(
        "Selection guidee",
        "Choisissez un code postal, puis un profil de chambres pour concentrer l'analyse sur une maison precise et la comparer a ses ventes de reference.",
    )

    zipcodes = sorted(df["zipcode"].unique().tolist())
    selected_zipcode = st.selectbox("1. Code postal", options=zipcodes, key="property_zip")

    zipcode_df = df[df["zipcode"] == selected_zipcode].copy()
    bedroom_options = sorted(zipcode_df["bedrooms"].dropna().astype(int).unique().tolist())
    selected_bedrooms = st.selectbox("2. Chambres", options=bedroom_options, key="property_bedrooms")

    candidates = zipcode_df[zipcode_df["bedrooms"] == selected_bedrooms].copy()
    candidates = candidates.sort_values("price", ascending=False)
    label_map = {
        row["id"]: (
            f"ID {row['id']} | {fmt_money(row['price'])} | "
            f"{int(row['sqft_living'])} pi2 | quartier {row['neighborhood_avg_house_age']:.1f} ans"
        )
        for _, row in candidates.iterrows()
    }

    selected_id = st.selectbox(
        "3. Propriete",
        options=candidates["id"].tolist(),
        format_func=lambda property_id: label_map[property_id],
        key="property_id",
    )

    selected = candidates[candidates["id"] == selected_id]
    if selected.empty:
        return None

    return selected.iloc[0]


def find_comparables(df: pd.DataFrame, property_row: pd.Series) -> pd.DataFrame:
    low_sqft = property_row["sqft_living"] * 0.8
    high_sqft = property_row["sqft_living"] * 1.2

    comps = df[
        (df["id"] != property_row["id"])
        & (df["zipcode"] == property_row["zipcode"])
        & (df["bedrooms"] == property_row["bedrooms"])
        & (df["sqft_living"].between(low_sqft, high_sqft))
    ].copy()

    comps["distance_score"] = (
        (comps["bathrooms"] - property_row["bathrooms"]).abs()
        + (comps["sqft_living"] - property_row["sqft_living"]).abs() / max(property_row["sqft_living"], 1)
        + (comps["grade"] - property_row["grade"]).abs() * 0.5
        + (comps["condition"] - property_row["condition"]).abs() * 0.5
    )

    comps = comps.sort_values(["distance_score", "date"], ascending=[True, False])

    if len(comps) >= 5:
        comps["comp_scope"] = "Strict"
        return comps.head(10)

    relaxed = df[
        (df["id"] != property_row["id"])
        & (df["zipcode"] == property_row["zipcode"])
        & (df["bedrooms"] == property_row["bedrooms"])
    ].copy()

    relaxed["distance_score"] = (
        (relaxed["bathrooms"] - property_row["bathrooms"]).abs()
        + (relaxed["sqft_living"] - property_row["sqft_living"]).abs() / max(property_row["sqft_living"], 1)
        + (relaxed["grade"] - property_row["grade"]).abs() * 0.5
        + (relaxed["condition"] - property_row["condition"]).abs() * 0.5
    )
    relaxed["comp_scope"] = np.where(
        relaxed["sqft_living"].between(low_sqft, high_sqft), "Strict", "Elargi"
    )

    return relaxed.sort_values(["distance_score", "date"], ascending=[True, False]).head(10)


def render_property_profile(property_row: pd.Series) -> None:
    st.subheader("Fiche descriptive")
    render_property_badges(property_row)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix", fmt_money(property_row["price"]))
    c2.metric("Prix / pi2", fmt_money(property_row["price_per_sqft"]))
    c3.metric("Chambres / bains", f"{int(property_row['bedrooms'])} / {property_row['bathrooms']}")
    c4.metric("Superficie", f"{int(property_row['sqft_living']):,} pi2".replace(",", " "))

    details = pd.DataFrame(
        [
            ("ID", property_row["id"]),
            ("Date de vente", property_row["date"].strftime("%Y-%m-%d") if pd.notna(property_row["date"]) else "N/A"),
            ("Code postal", property_row["zipcode"]),
            ("Terrain", f"{int(property_row['sqft_lot']):,} pi2".replace(",", " ")),
            ("Age moyen des maisons du quartier", f"{property_row['neighborhood_avg_house_age']:.1f} ans"),
            ("Condition", int(property_row["condition"])),
            ("Annee de construction", int(property_row["yr_built"])),
            ("Age lors de la vente", f"{int(property_row['age'])} ans"),
            ("Renovee", "Oui" if property_row["is_renovated"] else "Non"),
            ("Sous-sol", "Oui" if property_row["has_basement"] else "Non"),
            ("Front de mer", "Oui" if property_row["waterfront"] == 1 else "Non"),
            ("Vue", int(property_row["view"])),
        ],
        columns=["Attribut", "Valeur"],
    )
    st.dataframe(details, use_container_width=True, hide_index=True)


def plot_comparables(property_row: pd.Series, comps: pd.DataFrame):
    plot_df = comps[["id", "price"]].copy()
    plot_df["label"] = plot_df["id"].astype(str)
    property_plot = pd.DataFrame(
        [{"id": property_row["id"], "price": property_row["price"], "label": "Propriete analysee"}]
    )
    plot_df = pd.concat([property_plot, plot_df], ignore_index=True)

    colors = [CHART_COLORS["coral"]] + [CHART_COLORS["sage"]] * (len(plot_df) - 1)

    fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#fffdf9")
    ax.set_facecolor("#fffdf9")
    ax.bar(plot_df["label"], plot_df["price"], color=colors)
    ax.axhline(comps["price"].mean(), color=CHART_COLORS["amber"], linestyle="--", linewidth=2, alpha=0.95)
    style_axes(ax, "Comparaison des prix avec les comparables", "Proprietes", "Prix ($)")
    ax.ticklabel_format(style="plain", axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.annotate(
        "Sujet",
        xy=(0, property_row["price"]),
        xytext=(0, property_row["price"] * 1.03),
        ha="center",
        color=CHART_COLORS["coral"],
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def build_property_prompt(property_row: pd.Series, comps: pd.DataFrame, mean_comp_price: float, gap_value: float, gap_pct: float, status: str) -> str:
    return f"""
Tu es un analyste immobilier senior. Evalue cette propriete pour un investisseur :

PROPRIETE ANALYSEE :
- Prix : {property_row["price"]:,.0f} $
- Chambres : {int(property_row["bedrooms"])} | Salles de bain : {property_row["bathrooms"]}
- Superficie : {int(property_row["sqft_living"])} pi2 | Terrain : {int(property_row["sqft_lot"])} pi2
- Age moyen des maisons du quartier : {property_row["neighborhood_avg_house_age"]:.1f} ans | Condition : {int(property_row["condition"])}/5
- Annee de construction : {int(property_row["yr_built"])} | Renovee : {"Oui" if property_row["is_renovated"] else "Non"}
- Front de mer : {"Oui" if property_row["waterfront"] == 1 else "Non"} | Vue : {int(property_row["view"])}/4

ANALYSE COMPARATIVE :
- Nombre de comparables trouves : {len(comps)}
- Prix moyen des comparables : {mean_comp_price:,.0f} $
- Prix median des comparables : {comps["price"].median():,.0f} $
- Ecart vs comparables : {gap_value:+,.0f} $ ({gap_pct:+.1f}%)
- Statut : {status}

Redige une recommandation d'investissement en 3-4 paragraphes.
Inclus : evaluation du prix, forces et faiblesses, verdict final
(Acheter / A surveiller / Eviter) avec justification.
"""


def render_property_tab(df: pd.DataFrame) -> None:
    property_row = select_property(df)
    if property_row is None:
        st.warning("Impossible de charger la propriete selectionnee.")
        return

    render_property_profile(property_row)
    render_neighborhood_profile(df, property_row["zipcode"])

    st.subheader("Recherche de comparables")
    comps = find_comparables(df, property_row)
    if comps.empty:
        st.warning("Aucun comparable n'a ete trouve avec les criteres demandes.")
        return

    mean_comp_price = comps["price"].mean()
    gap_value = property_row["price"] - mean_comp_price
    gap_pct = (gap_value / mean_comp_price * 100) if mean_comp_price else 0.0
    status = "Surcote" if gap_value > 0 else "Decote"

    c1, c2, c3 = st.columns(3)
    c1.metric("Prix moyen des comparables", fmt_money(mean_comp_price))
    c2.metric("Ecart en $", fmt_signed_money(gap_value))
    c3.metric("Ecart en %", fmt_pct(gap_pct))
    render_verdict_banner(status, gap_value, gap_pct)
    render_positioning_gauge(gap_pct)
    if "Elargi" in comps["comp_scope"].values:
        st.caption(
            "Note : moins de 5 comparables stricts etaient disponibles. "
            "La liste a ete elargie aux maisons du meme code postal et du meme nombre de chambres."
        )

    display_cols = [
        "id",
        "date",
        "price",
        "price_per_sqft",
        "sqft_living",
        "bathrooms",
        "neighborhood_avg_house_age",
        "condition",
        "distance_score",
        "comp_scope",
    ]
    table = comps[display_cols].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table = table.rename(
        columns={
            "price": "Prix",
            "price_per_sqft": "Prix / pi2",
            "sqft_living": "Superficie",
            "bathrooms": "Salles de bain",
            "neighborhood_avg_house_age": "Age moyen quartier",
            "condition": "Condition",
            "distance_score": "Score de proximite",
            "comp_scope": "Portee comp",
            "date": "Date",
            "id": "ID",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.pyplot(plot_comparables(property_row, comps), clear_figure=True)

    if st.button("Generer une recommandation", key="property_recommendation"):
        with st.spinner("Generation de la recommandation en cours..."):
            recommendation = call_llm(
                build_property_prompt(property_row, comps, mean_comp_price, gap_value, gap_pct, status)
            )
        st.markdown("### Recommandation d'investissement")
        st.write(recommendation)


def main() -> None:
    apply_custom_theme()

    try:
        data = load_data(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Le fichier {DATA_FILE} est introuvable dans le dossier du projet.")
        st.stop()

    render_sidebar_brand()
    render_hero(data)

    filtered_market = build_market_filters(data)

    if filtered_market.empty:
        st.warning("Aucune propriete ne correspond aux filtres actuels.")
        st.stop()

    with st.expander("Apercu de la preparation des donnees"):
        st.write(
            "Colonnes preparees : `price_per_sqft`, `age`, `is_renovated`, `has_basement`."
        )
        st.dataframe(
            filtered_market[
                [
                    "id",
                    "date",
                    "price",
                    "sqft_living",
                    "price_per_sqft",
                    "age",
                    "is_renovated",
                    "has_basement",
                ]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )

    tab1, tab2 = st.tabs(["Exploration du marche", "Analyse d'une propriete"])

    with tab1:
        render_market_tab(filtered_market)

    with tab2:
        render_property_tab(data)


if __name__ == "__main__":
    main()
