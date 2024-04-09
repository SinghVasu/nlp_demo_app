import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from docx import Document
import pdfplumber
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import altair as alt
import heapq

# Setting up the page configuration and theme
st.set_page_config(page_title="NLP Analysis Apps", layout="wide")

def app1():
    st.title("NLP Showcase")
    nlp = spacy.load("en_core_web_sm")

    with st.expander("Choose input mode"):
        text_input_mode = st.radio("Select", ["Text Area", "Upload Document"])

    user_input = ""
    if text_input_mode == "Text Area":
        user_input = st.text_area("Enter text (up to 5000 characters):", placeholder="Type or paste your text here...")
    else:
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            st.success("File successfully uploaded!")
            if uploaded_file.type == "application/pdf":
                user_input = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                user_input = read_docx(uploaded_file)
            else:
                user_input = uploaded_file.getvalue().decode("utf-8")

    if st.button("Analyze") and user_input:
        with st.spinner('Processing text...'):
            processed_text = preprocess_text(user_input, nlp)
            sentiment = analyze_sentiment(processed_text)
            generate_wordcloud(processed_text)
            display_word_frequencies(processed_text)
            display_named_entities(user_input, nlp)
            display_pos_tags(user_input, nlp)
            display_tfidf_scores(processed_text)

def app2():
    st.title("Customer Feedback Analysis")
    nlp = spacy.load("en_core_web_sm")

    with st.expander("Choose input mode"):
        text_input_mode = st.radio("Select", ["Text Area", "Upload Document"])

    user_input = ""
    if text_input_mode == "Text Area":
        user_input = st.text_area("Enter customer feedback (up to 5000 characters):", placeholder="Type or paste customer feedback here...")
    else:
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            st.success("File successfully uploaded!")
            if uploaded_file.type == "application/pdf":
                user_input = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                user_input = read_docx(uploaded_file)
            else:
                user_input = uploaded_file.getvalue().decode("utf-8")

    if st.button("Analyze") and user_input:
        with st.spinner('Processing customer feedback...'):
            processed_text = preprocess_text(user_input, nlp)
            sentiment = analyze_sentiment(processed_text)
            generate_wordcloud(processed_text)
            display_feedback_summary(user_input, nlp)

# Helper functions used in app1 and app2 for NLP and visualization tasks
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "".join([page.extract_text() + '\n' for page in pdf.pages])
    return text

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess_text(text, nlp):
    text = text.lower()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.like_num])

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def generate_wordcloud(text):
    wordcloud = WordCloud(width=500, height=300, background_color='white').generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def display_word_frequencies(text):
    word_freq = pd.Series(text.split()).value_counts().reset_index()
    word_freq.columns = ['Word', 'Frequency']
    st.bar_chart(word_freq.set_index('Word'))

def display_named_entities(text, nlp):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents if not ent.text.isspace()]
    st.dataframe(pd.DataFrame(entities, columns=["Entity", "Label"]))

def display_pos_tags(text, nlp):
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc if not token.is_punct and not token.is_space]
    st.dataframe(pd.DataFrame(pos_tags, columns=['Token', 'POS']))

def display_tfidf_scores(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    st.dataframe(df.T.sort_values(by=0, ascending=False).head(10))

def display_feedback_summary(text, nlp):
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    doc = nlp(text)
    for word in doc:
        if word.text.lower() not in stop_words:
            word_frequencies[word.text.lower()] = word_frequencies.get(word.text.lower(), 0) + 1
    maximum_freq = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_freq
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sent.text for sent in summary_sentences])
    st.write(summary)

# Add pages to your app
pages = {
    "NLP Showcase": app1,
    "Customer Feedback Analysis": app2
}

st.sidebar.title('Apps')
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page()
