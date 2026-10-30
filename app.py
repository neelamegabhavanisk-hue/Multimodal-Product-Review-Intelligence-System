from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.review_intelligence import (
    build_top_mentions,
    compute_summary,
    generate_recommendation,
    load_reviews,
)

DATA_PATH = Path(__file__).parent / "data" / "sample_reviews.csv"

st.set_page_config(
    page_title="Multimodal Product Review Intelligence System",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f5f7ff 0%, #eef4ff 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 700;
        }
        .stApp {
            color: #14213d;
        }
        .card {
            border-radius: 16px;
            padding: 18px;
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(128, 143, 180, 0.25);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_demo_data():
    return load_reviews(str(DATA_PATH))


def render_review_cards(df):
    if df.empty:
        st.info("No reviews available for this selection.")
        return

    for _, row in df.iterrows():
        sentiment = row.get("sentiment_label", "Neutral")
        color = "#22c55e" if sentiment == "Positive" else "#f59e0b" if sentiment == "Neutral" else "#ef4444"
        st.markdown(
            f"""
            <div class="card" style="margin-bottom: 16px;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
                    <div>
                        <div style="font-size:1.05rem; font-weight:700;">{row.get('product_name', 'Product')}</div>
                        <div style="font-size:0.8rem; color:#475467; margin-top:4px;">{row.get('category', 'Uncategorized')} • Rating: {row.get('rating', 0)}/5</div>
                    </div>
                    <span style="background:{color}; color:white; padding:7px 12px; border-radius:999px; font-size:0.8rem; font-weight:700;">
                        {sentiment}
                    </span>
                </div>
                <p style="margin-top:14px; margin-bottom: 8px; line-height:1.6; font-size:0.96rem;">{row.get('review_text', '')}</p>
                <div style="font-size:0.8rem; color:#475467;">Aspects: {', '.join(row.get('aspects', [])) or 'None'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if row.get("image_url"):
            st.image(row["image_url"], width=220, caption="Review image")


def main():
    st.title("📊 Multimodal Product Review Intelligence System")
    st.caption("Analyze real customer feedback, identify issue drivers, and monitor sentiment trends across products.")

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload review data", type=["csv", "xlsx"])
        selected_category = st.selectbox("Category filter", ["All", "Electronics", "Home", "Accessories", "Uncategorized"])
        show_demo = st.toggle("Use demo data", value=(uploaded_file is None))

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Failed to read uploaded file: {exc}")
            st.stop()
    elif show_demo:
        df = get_demo_data()
    else:
        df = get_demo_data()

    if "review_text" not in df.columns:
        for candidate in ["comment", "review", "feedback"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "review_text"})
                break

    if "review_text" not in df.columns:
        st.warning("The uploaded file must include a review text column such as 'review_text' or 'comment'.")
        st.stop()

    if selected_category != "All":
        df = df[df["category"].astype(str).str.contains(selected_category, case=False, na=False)]

    df = df.copy()
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0)
    df["product_name"] = df.get("product_name", "Unknown Product").fillna("Unknown Product")
    df["category"] = df.get("category", "Uncategorized").fillna("Uncategorized")
    df["image_url"] = df.get("image_url", "").fillna("")
    df["review_text"] = df["review_text"].fillna("")

    enriched = load_reviews(dataframe=df)
    summary = compute_summary(enriched)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", summary["total_reviews"])
    col2.metric("Avg Rating", f"{summary['average_rating']}/5")
    col3.metric("Positive Rate", f"{summary['positive_rate']}%")
    col4.metric("Top Aspect", summary["top_aspect"])

    sentiment_counts = summary["sentiment_counts"]
    sentiment_df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Count": [
                sentiment_counts.get("Positive", 0),
                sentiment_counts.get("Neutral", 0),
                sentiment_counts.get("Negative", 0),
            ],
        }
    )

    overview, reviews, raw_data = st.tabs(["Overview", "Reviews", "Raw Data"])

    with overview:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
            fig = px.bar(
                sentiment_df,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map={"Positive": "#34d399", "Neutral": "#fbbf24", "Negative": "#f87171"},
                text="Count",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            st.markdown('<div class="section-title">Top Product Aspects</div>', unsafe_allow_html=True)
            aspect_tuples = build_top_mentions(enriched, limit=6)
            if aspect_tuples:
                aspect_df = pd.DataFrame(aspect_tuples, columns=["Aspect", "Mentions"])
                fig2 = px.bar(
                    aspect_df,
                    x="Mentions",
                    y="Aspect",
                    orientation="h",
                    color="Aspect",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No aspects detected in the current review set.")

        st.markdown('<div class="section-title">Executive Insight</div>', unsafe_allow_html=True)
        st.info(generate_recommendation(enriched))

    with reviews:
        render_review_cards(enriched)

    with raw_data:
        st.dataframe(
            enriched[["product_name", "category", "rating", "review_text", "sentiment_label", "sentiment_score", "aspects"]],
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
