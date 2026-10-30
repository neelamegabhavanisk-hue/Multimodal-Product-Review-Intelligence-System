# Multimodal Product Review Intelligence System

A Streamlit dashboard to process product reviews, extract sentiment and product aspects, and surface actionable insights from text and image-driven customer feedback.

## Features

- Upload a CSV file with product reviews
- Automatically score sentiment using a lightweight keyword-based engine
- Extract common aspects such as battery, camera, display, sound, and price
- View summary KPIs and charts for product performance
- Display review cards with optional product images
- Use built-in demo data when no file is uploaded

## Project structure

- `app.py`: main Streamlit dashboard
- `src/review_intelligence.py`: review analysis logic
- `data/sample_reviews.csv`: demo dataset
- `requirements.txt`: Python dependencies

## Run locally

1. Open a terminal in this folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Expected CSV columns

The app works with a CSV containing one or more of these columns:

- `product_name`
- `category`
- `rating`
- `review_text`
- `image_url`
- `features`

If `review_text` is missing, the app falls back to `comment` or `review` when available.
