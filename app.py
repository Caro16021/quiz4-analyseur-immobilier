import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

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
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


st.set_page_config(
    page_title="Analyseur immobilier - Comte de King",
    layout="wide",
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


def build_market_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filtres du marche")

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
    grade_range = st.sidebar.slider(
        "Grade de construction",
        min_value=int(df["grade"].min()),
        max_value=int(df["grade"].max()),
        value=(int(df["grade"].min()), int(df["grade"].max())),
        step=1,
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
        & df["grade"].between(*grade_range)
        & df["yr_built"].between(*year_range)
    ].copy()

    if zipcode_values:
        filtered = filtered[filtered["zipcode"].isin(zipcode_values)]

    if waterfront_only:
        filtered = filtered[filtered["waterfront"] == 1]

    return filtered


def plot_price_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["price"], bins=30, color="#1f77b4", edgecolor="white")
    ax.set_title("Distribution des prix")
    ax.set_xlabel("Prix de vente ($)")
    ax.set_ylabel("Nombre de proprietes")
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    return fig


def plot_price_vs_sqft(df: pd.DataFrame):
    sample = df.sample(min(len(df), 2500), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    scatter = ax.scatter(
        sample["sqft_living"],
        sample["price"],
        c=sample["grade"],
        cmap="viridis",
        alpha=0.7,
    )
    ax.set_title("Prix vs superficie habitable")
    ax.set_xlabel("Superficie habitable (pi2)")
    ax.set_ylabel("Prix ($)")
    fig.colorbar(scatter, ax=ax, label="Grade")
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
        "grade",
        "condition",
        "age",
    ]
    corr = df[numeric_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Matrice de correlation")
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

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_zipcode["zipcode"], by_zipcode["price"], color="#ff7f0e")
    ax.set_title("Prix moyen par code postal (Top 10)")
    ax.set_xlabel("Code postal")
    ax.set_ylabel("Prix moyen ($)")
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
    grade_distribution = (
        df["grade"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
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
- % maisons renovees : {df["is_renovated"].mean() * 100:.1f}%
- % maisons avec sous-sol : {df["has_basement"].mean() * 100:.1f}%
- Repartition par grade : {grade_distribution}
- % front de mer : {pct_waterfront:.1f}%

Redige un resume executif de ce segment en 3-4 paragraphes.
Identifie les tendances cles et les opportunites d'investissement.
"""


def render_market_tab(df: pd.DataFrame) -> None:
    st.subheader("Exploration du marche")

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
            f"{int(row['sqft_living'])} pi2 | grade {int(row['grade'])}"
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
            ("Grade", int(property_row["grade"])),
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

    colors = ["#d62728"] + ["#1f77b4"] * (len(plot_df) - 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(plot_df["label"], plot_df["price"], color=colors)
    ax.set_title("Comparaison des prix avec les comparables")
    ax.set_xlabel("Proprietes")
    ax.set_ylabel("Prix ($)")
    ax.ticklabel_format(style="plain", axis="y")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.annotate(
        "Sujet",
        xy=(0, property_row["price"]),
        xytext=(0, property_row["price"] * 1.03),
        ha="center",
        color="#d62728",
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
- Grade : {int(property_row["grade"])}/13 | Condition : {int(property_row["condition"])}/5
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
    st.info(f"Statut de la propriete : {status} par rapport a son marche local.")
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
        "grade",
        "condition",
        "distance_score",
        "comp_scope",
    ]
    table = comps[display_cols].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
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
    st.title("Analyseur immobilier - Comte de King")
    st.write(
        "Application interactive pour explorer le marche immobilier du comte de King "
        "et evaluer une propriete individuelle a partir de comparables."
    )

    try:
        data = load_data(DATA_FILE)
    except FileNotFoundError:
        st.error(f"Le fichier {DATA_FILE} est introuvable dans le dossier du projet.")
        st.stop()

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
