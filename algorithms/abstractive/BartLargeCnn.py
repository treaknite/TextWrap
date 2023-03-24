from transformers import pipeline

def bartSummary(input_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(input_text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    return summary