import gradio as gr
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pymorphy3
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

def preprocess(text):
    pattern = re.compile("[^а-яА-Яa-zA-Z0-9\-.,;]+")
    sw = stopwords.words("russian")
    morph = pymorphy3.MorphAnalyzer()
    text = text.lower()
    text = pattern.sub(" ", text)
    tokens = simple_preprocess(text, deacc=True)
    tokens = [
        morph.parse(token)[0].normal_form
        for token in tokens
        if token not in sw and len(token) > 2
    ]
    return tokens

def get_d2v_vector(text):
    doc = preprocess(text)
    return d2v.infer_vector(doc)


def find_top_matches(input_text, cv_texts, cv_embs, top_n=5):
    sims = cosine_similarity([get_d2v_vector(input_text)], cv_embs)[0]
    scores = [(sim, cv_text) for cv_text, sim in zip(cv_texts, sims)]
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_n]

with open('d2v.pkl', 'rb') as f:
    d2v = pickle.load(f)

with open('cv_sample.pkl', 'rb') as f:
    cv_texts = pickle.load(f)

with open('embs.pkl', 'rb') as f:
    cv_embs = pickle.load(f)


with gr.Blocks() as app:
    gr.Markdown("## Поиск резюме")
    with gr.Row():
        input_text = gr.Textbox(label="Текст вакансии")
    with gr.Row():
        button = gr.Button("Найти топ-5 подходящих резюме")
    output = gr.Dataframe(label="Топ-5 резюме")

    button.click(fn=lambda x: find_top_matches(x, cv_texts, cv_embs), inputs=input_text, outputs=output)

app.launch()