import os
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def load_documents():
    root_folder = "/Users/mohithsai/Artificial Intelligance/Llama/dataset"
    loader = DirectoryLoader(
        path=root_folder,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        loader_kwargs={"extract_images": True},
        recursive=True
    )
    documents = loader.load()
    return documents

def create_retriever(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " "]
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2',
        model_kwargs={"device": "cpu"}
    )

    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")

    return db.as_retriever(search_type='mmr', search_kwargs={"k": 3})

def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2',
        model_kwargs={"device": "cpu"}
    )

    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).as_retriever(
        search_type='mmr', search_kwargs={"k": 3}
    )

def build_retrieval_qa_chain(retriever, conversation_history):

    history_context = "\n".join(
        [f"User: {entry['question']}\nSynergyMate: {entry['response']}" for entry in conversation_history]
    )

    system_prompt = (
        "You are a highly knowledgeable and empathetic medical assistant. "
        "Your primary goal is to provide information and guidance on medical conditions, symptoms, and treatments. "
        "Respond with clear, concise, and fact-based medical advice, emphasizing the importance of consulting a healthcare professional for personalized care.\n\n"
        "Conversation History:\n" + history_context + "\n\nContext: {context}"
    )

    model = OllamaLLM(
    model='llama3.1',
    temperature=0.7,
    max_tokens=512,
    top_p=0.85,
)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    stuff_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever,
        stuff_chain
    )
    return retrieval_chain

def main():
    st.title("Health Chatbot")
    st.write("Ask your medical questions below:")

    if "history" not in st.session_state:
        st.session_state["history"] = []


    if os.path.exists("faiss_index"):
        retriever = load_faiss_index()
    else:
        retriever = create_retriever(load_documents())

    chain = build_retrieval_qa_chain(retriever, st.session_state["history"])

    user_question = st.text_input("Type your medical question here:")

    if user_question:
        try:
            response = chain.invoke({"input": user_question})

            st.session_state["history"].append({"question": user_question, "response": response.get("answer", "No response available")})
            
            for chat in st.session_state["history"]:
                st.write(f"**You:** {chat['question']}")
                st.write(f"**SynergyMate:** {chat['response']}")
            
            if 'context' in response:
                st.write("Source Documents:")
                for doc in response['context']:
                    st.write(f" - Source: {doc.metadata['source']}, Page: {doc.metadata.get('page', 'N/A')}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
