import pandas as pd

from src.review_intelligence import analyze_review_sentiment, load_reviews


def test_analyze_review_sentiment_positive_and_negative():
    assert analyze_review_sentiment("Battery life is excellent and the screen is very bright.") > 0
    assert analyze_review_sentiment("Terrible sound quality and battery drains too fast.") < 0


def test_load_reviews_accepts_comment_column_and_enriches_data():
    df = pd.DataFrame([
        {"product_name": "Demo", "category": "Electronics", "rating": 5, "comment": "Great camera and excellent battery life."},
        {"product_name": "Demo2", "category": "Accessories", "rating": 1, "comment": "Very poor quality and slow performance."},
    ])

    result = load_reviews(dataframe=df)
    assert "review_text" in result.columns
    assert result["sentiment_label"].isin(["Positive", "Negative"]).all()
