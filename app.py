import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROP_API_KEY")
model_name = os.getenv("MODEL_NAME")

from langchain_groq import ChatGroq

llm = ChatGroq(temperature=0.5,groq_api_key=groq_api_key,
               model_name=model_name)


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings()

vectordb1 = FAISS.load_local("faiss_indx_db", embeddings, allow_dangerous_deserialization=True)

retriever = vectordb1.as_retriever(search_kwargs={"k": 2})

from langchain.prompts import PromptTemplate

prompt_template = """

### INSTRUCTION: REMOVE INTRODUCTION

Use the following pieces of context to answer the question at the end.
Provide your answer in Portuguese.
If you don't know the answer, just say that to look for an assistant, don't try to make up an answer.
If user say any greeting, reply cordially, and don´t try to make up an answer.

CONTEXT: {context}

QUESTION: {question}

ANSWER:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs = {"prompt": PROMPT}
)



from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = qa_chain(msg)
    print("Response : ", response["result"])
    return str(response["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug= True)
