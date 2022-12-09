from transformers import pipeline
from flask import Flask, request
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

data = pd.read_csv('traveldocument.csv')


embeddings = model.encode(data["Question"])


def old_version():
    p = pipeline('question-answering')



    #p(question="what can I do?", context=context)

    @app.route("/ask", methods=['GET'])
    def ask():
        return p(question=request.args.get("q"), context=context)['answer']


app = Flask(__name__)

@app.route("/")
def main():
    with open("index.html") as fil:
        page = fil.read()
    return page


@app.route("/ask", methods=['GET'])
def ask():
    question=request.args.get("q")
    encodedq = model.encode([question])
    distances = np.linalg.norm(embeddings - encodedq, axis=1)
    closest = np.argsort(distances)[0]
    return data["Answer"][closest]

