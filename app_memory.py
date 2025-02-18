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

prompt_template = (
    "Provide your answer in Portuguese."
    "Just go straight to the point and don't take too long to answer the question."
    "If you don't know the answer, just say that to look for an assistant, don't try to make up an answer."
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)

PROMPT = PromptTemplate.from_template(prompt_template)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

# Create the multipurpose chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=retriever, 
    condense_question_prompt=PROMPT,
    return_source_documents=True
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
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug= True)
