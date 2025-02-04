import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Initialize Groq Client
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Dummy Groq Model names (Replace with actual Groq models when needed)
groq_models = ["gemma2-9b-it", "llama3-70b-8192", "mixtral-8x7b-32768"]

# Embedding setup (using OpenAI for embeddings)
openai_embeddings = OpenAIEmbeddings()

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# Data ingestion from uploaded PDFs
def data_ingestion(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, chunk_overlap=1000
            )
            docs.extend(text_splitter.split_documents(documents))
        finally:
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
    return docs


# Save FAISS Vector Store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, openai_embeddings)
    vectorstore_faiss.save_local("faiss_index")


# Generate Response with error handling
def get_response_llm(llm, vectorstore_faiss, query):
    try:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore_faiss.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )
        answer = qa({"query": query})
        result = answer.get("result", "No response available")
        return result
    except Exception as e:
        st.error(f"Error: {e}")
        return (
            "Sorry, there was an error while generating the response. Please try again."
        )


# Streamlit Application
def main():
    st.set_page_config(page_title="Chat with PDF using Groq", layout="wide")

    # Custom styling for centering the title and adding space below it
    st.markdown(
        """
        <style>
            .css-1d391kg {font-size: 2em; text-align: center;}
            .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; }
            .stTextInput>div>div>input { text-align: left; }
            .stApp { padding-top: 20px; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Center the title with spacing below
    st.markdown(
        "<h1 style='text-align: center;'>🔍 Chat with PDF using Groq and OpenAI Embeddings</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='height: 30px;'></div>", unsafe_allow_html=True
    )  # Adding space below the title

    # Create layout for the left sidebar and main area
    col1, col2 = st.columns([1, 3])  # Sidebar for upload, main area in center

    # Left Sidebar for file upload and model selection
    with col1:
        st.subheader("📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)!")

        st.subheader("⚙️ Select Model")
        initial_model = st.selectbox("Choose the Groq Model", groq_models)

    # Main Area (Central area) to display question and response from the selected model
    with col2:
        st.subheader("Ask a Question ❓ ")

        # Create columns for input box and button
        question_col, button_col = st.columns([3, 1])

        with question_col:
            user_question = st.text_input(
                "Type your question based on the uploaded PDFs",
                label_visibility="collapsed",
            )

        with button_col:
            # Align "Go" button to the right
            go_button = st.button("Go 🚀")

        if uploaded_files and user_question and go_button:
            with st.spinner("Processing... Please wait..."):
                docs = data_ingestion(uploaded_files)
                st.progress(25)

                # Save vector store using OpenAI embeddings
                get_vector_store(docs)
                st.progress(50)

                # Load FAISS index
                faiss_index = FAISS.load_local(
                    "faiss_index",
                    openai_embeddings,
                    allow_dangerous_deserialization=True,
                )
                st.progress(75)

                # Initialize the selected Groq model
                llm_groq = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model_name=initial_model,  # Use selected Groq model
                )

                # Get response from the selected Groq model
                groq_output = get_response_llm(llm_groq, faiss_index, user_question)
                st.progress(100)

                # Display model output
                st.subheader(f"📝 Response from {initial_model}")
                st.write(groq_output)


if __name__ == "__main__":
    main()
