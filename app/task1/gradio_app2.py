import gradio as gr

def predict_sentiment(text):
    if not text:
        return "Please enter some text."
    return "Positive 😊" if "good" in text.lower() or "love" in text.lower() else "Negative 😞"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter a sentence"),
    outputs=gr.Textbox(label="Sentiment"),
    title="🧠 Sentiment Analyzer (Gradio)",
    description="Type a sentence and see whether it's positive or negative."
)

demo.launch()
