from transformers import pipeline
from flask import Flask, request
p = pipeline('question-answering')

with open('content.txt') as fil:
    context = fil.read()

#p(question="what can I do?", context=context)




app = Flask(__name__)

@app.route("/")
def main():
    with open("index.html") as fil:
        page = fil.read()
    return page

@app.route("/ask", methods=['GET'])
def ask():
    return p(question=request.args.get("q"), context=context)['answer']