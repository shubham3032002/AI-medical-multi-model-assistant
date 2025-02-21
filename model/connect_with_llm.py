import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def load_llm():
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=1000
    )
    return llm

llm = load_llm()


# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the information provided in the context to answer the user's question.
If the answer is not found in the context, respond with "I don't know."
Do not make up an answer or include unnecessary details.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

# Step 3: Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Step 5: User Query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

# Output the response and source documents
print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])
