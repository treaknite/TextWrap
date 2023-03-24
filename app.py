# Imports
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.summarization.summarizer import summarize
import matplotlib.pyplot as plt
import io
import base64

# Import algorithms
from ./algorithms.abstractive import *
from ./algorithms.extractive import *
from ./algorithms.textWrapAlgorithm import * 


# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")


# get input text from client then summarise text and generate summary and summary
@app.route("/summariser", methods=['GET', 'POST'])
def summariser():
    if request.method == "GET":
        return render_template('webApp.html')

    if request.method == "POST":
        # Get user input from html form
        text = request.form['initial-corpus']
        algorithm = request.form['algorithm']

        # Summarize text based on selected algorithm
        # Using TextWrap Algorithm
        if algorithm == 'textWrap':
            summary = runner(text)

        # For extractive based approach algorithm
        if algorithm == 'gensim':
            summary = summarize(text, ratio=0.2)
        elif algorithm == 'textrank':
            summary = extract_top_sentences(text, num_sentences=3)
        elif algorithm == 'lexrank':
            summary = lexrank_summarize(text, n=3)
        elif algorithm == 'sumbasic':
            summary = sumbasic_summarize(text, length=3)
        elif algorithm == 'frequencyMethod':
            summary = summaryGenerator(text)
        

        # For abstractive based approach
        if algorithm == 'bartLargeCnn':
            summary = bartLargeCnn(text)
        elif algorithm == '':
            pass
        elif algorithm == '':
            pass
            

        # Generate summary report
        word_counts = len(text.split())
        summary_counts = len(summary.split())
        summary_ratio = summary_counts / word_counts
        fig = plt.figure()
        plt.bar(['Original Text', 'Summary'], [word_counts, summary_counts])
        plt.title('Word Counts')
        plt.xlabel('Text')
        plt.ylabel('Word Count')
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Render summary template with summary and summary report
        return render_template('summary.html', summary=summary, plot_url=plot_url)


if __name__ == "__main__":
    app.run(threaded=False)
