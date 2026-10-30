import re
from collections import Counter
from typing import List

import pandas as pd

POSITIVE_WORDS = {
    "excellent": 2.5,
    "amazing": 2.5,
    "love": 2.4,
    "great": 2.2,
    "good": 1.7,
    "fast": 1.6,
    "smooth": 1.6,
    "clear": 1.6,
    "vibrant": 1.8,
    "comfortable": 1.8,
    "premium": 1.8,
    "durable": 1.8,
    "easy": 1.5,
    "best": 2.4,
    "beautiful": 2.0,
    "helpful": 1.7,
    "reliable": 1.9,
    "powerful": 1.8,
    "sharp": 1.8,
    "bright": 1.6,
    "value": 1.5,
    "perfect": 2.6,
    "affordable": 1.6,
    "lightweight": 1.5,
    "nice": 1.4,
    "warm": 1.3,
    "stylish": 1.5,
    "better": 1.7,
    "worth": 1.8,
    "easy": 1.5,
    "quiet": 1.4,
    "reliable": 1.9,
    "strong": 1.5,
}

NEGATIVE_WORDS = {
    "bad": 2.5,
    "poor": 2.7,
    "slow": 2.2,
    "cheap": 2.3,
    "broken": 3.0,
    "laggy": 2.6,
    "weak": 2.0,
    "disappointing": 2.8,
    "terrible": 3.2,
    "hard": 1.7,
    "issue": 2.1,
    "problem": 2.2,
    "delay": 2.0,
    "faulty": 3.0,
    "noisy": 2.0,
    "blurry": 2.1,
    "overpriced": 2.3,
    "uncomfortable": 2.4,
    "heavy": 1.4,
    "awful": 3.1,
    "dull": 2.1,
    "drain": 2.2,
    "expensive": 1.7,
    "weak": 2.0,
    "fragile": 2.7,
    "noisy": 2.1,
    "average": 1.2,
    "lack": 1.8,
    "unable": 1.9,
}

INTENSIFIERS = {"very": 1.6, "really": 1.5, "extremely": 1.8, "super": 1.7, "so": 1.4, "quite": 1.3, "highly": 1.5}
NEGATIONS = {"not": True, "no": True, "never": True, "hardly": True, "without": True}

ASPECT_KEYWORDS = {
    "battery": ["battery", "charge", "charging", "lasts", "drain", "power"],
    "camera": ["camera", "photo", "video", "picture", "lens", "zoom"],
    "display": ["screen", "display", "pixel", "brightness", "lcd", "monitor"],
    "sound": ["sound", "audio", "speaker", "music", "voice", "bass"],
    "performance": ["speed", "performance", "lag", "fast", "smooth", "processor", "laggy"],
    "design": ["design", "look", "style", "build", "material", "fit", "appearance"],
    "price": ["price", "value", "cost", "cheap", "expensive", "worth"],
    "comfort": ["comfortable", "ergonomic", "fit", "soft", "handy", "strap", "wear"],
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze_review_sentiment(text: str) -> float:
    norm = normalize_text(text)
    if not norm:
        return 0.0

    tokens = norm.split()
    score = 0.0
    prev_word = None

    for idx, word in enumerate(tokens):
        if word in NEGATIONS:
            prev_word = word
            continue

        if word in INTENSIFIERS:
            prev_word = word
            continue

        if word in POSITIVE_WORDS:
            weight = POSITIVE_WORDS[word]
            if prev_word in NEGATIONS:
                weight *= -1.0
            if prev_word in INTENSIFIERS:
                weight *= INTENSIFIERS[prev_word]
            score += weight
        elif word in NEGATIVE_WORDS:
            weight = NEGATIVE_WORDS[word]
            if prev_word in NEGATIONS:
                weight *= -1.0
            if prev_word in INTENSIFIERS:
                weight *= INTENSIFIERS[prev_word]
            score -= weight

        prev_word = word

    return round(score, 2)


def sentiment_score(text: str) -> int:
    return int(round(analyze_review_sentiment(text)))


def get_sentiment_label(score: int) -> str:
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def extract_aspects(text: str) -> List[str]:
    norm = normalize_text(text)
    mentioned = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", norm) for word in keywords):
            mentioned.append(aspect)
    return mentioned


def load_reviews(csv_path: str | None = None, dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    if dataframe is not None:
        df = dataframe.copy()
    else:
        if csv_path is None:
            raise ValueError("Either a CSV path or a DataFrame is required.")
        df = pd.read_csv(csv_path)

    if df.empty:
        return df

    if "review_text" not in df.columns:
        for candidate in ["comment", "review", "feedback", "review_text"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "review_text"})
                break

    if "review_text" not in df.columns:
        df["review_text"] = ""

    if "rating" not in df.columns:
        df["rating"] = 0

    if "product_name" not in df.columns:
        df["product_name"] = "Unknown Product"

    if "category" not in df.columns:
        df["category"] = "Uncategorized"

    if "image_url" not in df.columns:
        df["image_url"] = ""

    df["review_text"] = df["review_text"].fillna("")
    df["sentiment_score"] = df["review_text"].apply(sentiment_score)
    df["sentiment_label"] = df["sentiment_score"].apply(get_sentiment_label)
    df["aspects"] = df["review_text"].apply(extract_aspects)
    return df


def compute_summary(df: pd.DataFrame):
    if df.empty:
        return {
            "total_reviews": 0,
            "average_rating": 0,
            "positive_rate": 0,
            "top_aspect": "None",
            "sentiment_counts": {"Positive": 0, "Negative": 0, "Neutral": 0},
        }

    total_reviews = len(df)
    average_rating = df["rating"].mean() if "rating" in df.columns else 0
    positive_rate = (df["sentiment_label"].eq("Positive").mean() * 100) if "sentiment_label" in df.columns else 0

    aspect_counter = Counter()
    for aspects in df.get("aspects", pd.Series([], dtype="object")):
        for aspect in aspects:
            aspect_counter[aspect] += 1

    top_aspect = aspect_counter.most_common(1)[0][0] if aspect_counter else "None"
    sentiment_counts = df["sentiment_label"].value_counts().to_dict()

    return {
        "total_reviews": total_reviews,
        "average_rating": round(float(average_rating), 2),
        "positive_rate": round(float(positive_rate), 1),
        "top_aspect": top_aspect,
        "sentiment_counts": sentiment_counts,
    }


def build_top_mentions(df: pd.DataFrame, limit: int = 5):
    all_aspects = []
    for aspects in df.get("aspects", pd.Series([], dtype="object")):
        all_aspects.extend(aspects)
    counter = Counter(all_aspects)
    return [(aspect, count) for aspect, count in counter.most_common(limit)]


def generate_recommendation(df: pd.DataFrame) -> str:
    summary = compute_summary(df)
    if summary["total_reviews"] == 0:
        return "Upload review data to begin the analysis."

    negative = summary["sentiment_counts"].get("Negative", 0)
    top_aspect = summary["top_aspect"]
    if negative and top_aspect != "None":
        return (
            f"The strongest issue area is {top_aspect}. Focus on product improvements there to reduce "
            f"negative sentiment and boost customer trust."
        )
    if summary["positive_rate"] > 65:
        return "Customer sentiment is strong; continue investing in the features that drive repeat satisfaction and positive advocacy."
    return "Customer sentiment is balanced; monitor the most-mentioned concerns and keep quality improvements visible to buyers."
