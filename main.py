import os
import uuid
from flask import Flask, request, render_template, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv  # Import dotenv to load .env file

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Together
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # Use secret key from .env file
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["pdf"]
        if uploaded_file.filename.endswith(".pdf"):
            # Save file
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(filepath)

            # Load and process PDF
            loader = PyPDFLoader(filepath)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)

            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embedding_model)

            llm = Together(
                model="mistralai/Mistral-7B-Instruct-v0.1",
                temperature=0.3,
                max_tokens=512
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )

            question_prompt = "Generate a meaningful question from this document that can be answered from the content."
            result = qa_chain.invoke(question_prompt)
            generated_question = result["result"]

            session["filepath"] = filepath
            session["question"] = generated_question

            return render_template("index.html", question=generated_question, stage="answer")

    return render_template("index.html", question=None, stage="upload")


@app.route("/check", methods=["POST"])
def check_answer():
    user_answer = request.form["answer"]
    question = session.get("question")
    filepath = session.get("filepath")

    loader = PyPDFLoader(filepath)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        temperature=0.3,
        max_tokens=512
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    model_answer = qa_chain.invoke(question)["result"]

    # Compare user vs model answer
    model_emb = semantic_model.encode(model_answer, convert_to_tensor=True)
    user_emb = semantic_model.encode(user_answer, convert_to_tensor=True)
    similarity = util.cos_sim(model_emb, user_emb).item()
    similarity = round(similarity, 2)
    correct = similarity >= 0.75
    feedback = "Correct!" if correct else "Not quite. Here's a better answer."

    return render_template(
        "index.html",
        question=question,
        stage="result",
        user_answer=user_answer,
        model_answer=model_answer,
        similarity=similarity,
        feedback=feedback
    )


if __name__ == "__main__":
    app.run(debug=True)
