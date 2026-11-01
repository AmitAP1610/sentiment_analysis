"""
Interactive Web Application for 3-Class Sentiment Analysis & Theme Extraction
"""

import joblib
import re

# Clean text function
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load trained models
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def analyze_review(review: str):
    if not review.strip():
        return "Please enter a product review.", "0%", "N/A"

    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    # Get probability mapping
    classes = model.classes_
    prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
    confidence = max(probabilities) * 100

    emoji_map = {
        "Positive": "Positive 😊",
        "Neutral": "Neutral 😐",
        "Negative": "Negative ☹️"
    }

    display_sentiment = emoji_map.get(prediction, prediction)
    
    breakdown = (
        f"Positive: {prob_dict.get('Positive', 0)*100:.1f}%\n"
        f"Neutral:  {prob_dict.get('Neutral', 0)*100:.1f}%\n"
        f"Negative: {prob_dict.get('Negative', 0)*100:.1f}%"
    )

    return display_sentiment, f"{confidence:.1f}%", breakdown

if __name__ == "__main__":
    try:
        import gradio as gr
        demo = gr.Interface(
            fn=analyze_review,
            inputs=gr.Textbox(
                lines=4,
                placeholder="Enter customer review text here (e.g., 'Battery lasts long, but buttons feel cheap')...",
                label="Customer Product Review"
            ),
            outputs=[
                gr.Textbox(label="Predicted Sentiment (3-Class)"),
                gr.Textbox(label="Model Confidence"),
                gr.Textbox(label="Class Probabilities Breakdown")
            ],
            title="🛒 Product Review Sentiment Analyzer (Positive / Neutral / Negative)",
            description="Classifies customer reviews into 3 sentiment classes using TF-IDF and Logistic Regression.",
            examples=[
                ["The battery life is phenomenal and easily lasts two full days. Highly recommended!"],
                ["It does the job as expected, but nothing extraordinary. Build is acceptable."],
                ["Broke down after just three days of normal usage. Terrible build quality."]
            ]
        )
        print("Launching Gradio UI...")
        demo.launch()
    except ImportError:
        print("Gradio is not installed yet. Running CLI mode instead:")
        while True:
            user_input = input("\nEnter a review (or 'q' to quit): ")
            if user_input.strip().lower() == 'q':
                break
            sent, conf, breakdown = analyze_review(user_input)
            print(f"Sentiment: {sent}")
            print(f"Confidence: {conf}")
            print(f"Breakdown:\n{breakdown}")

