from transformers import pipeline
from flask import Flask, request, send_file
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

data = pd.read_csv('traveldocument.csv')
questions = [sent_tokenize(q) for q in data['Question']]
answerids = [id for id, qlist in enumerate(questions) for item in qlist]
questions = [item for qlist in questions for item in qlist]
answers = data['Answer'].tolist()
alltext = questions + answers

embeddings = model.encode(alltext)
types = [0]*len(questions) + [1]*len(answers)


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

@app.route('/image/traveldoca.png')
def serve_image():
    return send_file('image/traveldoca.png', mimetype='image/png')


@app.route("/ask", methods=['GET'])
def ask():
    question=request.args.get("q")
    encodedq = model.encode([question])
    distances = np.linalg.norm(embeddings - encodedq, axis=1)
    closest = np.argsort(distances)[0]
    if (types[closest]==0): # Question
        return answers[answerids[closest]]
    else:
        return alltext[closest]

