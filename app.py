import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

# Path to the FAISS database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Function to load the vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Function to set a custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Function to load the Groq LLM
def load_llm():
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=1000
    )
    return llm

# Function to initialize conversation memory
def get_conversation_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Main application
def main():
    # Sidebar
    st.sidebar.title("🤖 MediBot Assistant")
    st.sidebar.markdown(
        """
        MediBot is your AI-powered medical assistant.
        
        ### Features:
        - 🩺 Answer medical questions
        - 🔬 Provide medically accurate advice
        - 📚 Reference trusted knowledge sources
        
        **Disclaimer:** Always consult a healthcare professional for serious health concerns.
        """
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []

    # Main title
    st.title("🩺 MediBot - Your Medical Assistant")

    # Display greeting if no messages yet
    if "messages" not in st.session_state or not st.session_state.messages:
        st.info("👋 Welcome to MediBot! Ask me about your symptoms, medications, or general health questions.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        elif message["role"] == "assistant":
            st.chat_message("assistant").markdown(message["content"])

    # User input
    user_query = st.chat_input("Describe your symptoms or ask a medical question")

    if user_query:
        # Display user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the provided context to answer the user's medical questions.
        If you are unsure, respond with "I don't know" rather than making up an answer.
        Provide medically accurate and concise responses.

        Context: {context}
        Question: {question}

        Answer directly, avoiding unnecessary details or small talk.
        """

        try:
            with st.spinner("Analyzing your query and fetching a response..."):
                # Load vector store
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("⚠️ Failed to load the medical knowledge base.")
                    return

                # Load LLM
                llm = load_llm()

                # Initialize conversation memory
                memory = get_conversation_memory()

                # Create a Conversational Retrieval QA chain
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=memory,
                    chain_type="stuff",
                    combine_docs_chain_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
                )

                # Run the QA chain with the user's query
                response = qa_chain({"question": user_query})

                # Prepare the result
                result = response["answer"]

                # Update chat history in session state
                st.session_state.chat_history.append((user_query, result))

                # Display chatbot's response
                st.chat_message("assistant").markdown(f"**MediBot:** {result}")
                st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()