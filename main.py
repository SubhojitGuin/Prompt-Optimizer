import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import faiss
import torch
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from keyword_extractor import extract_keywords, select_sentences

load_dotenv()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
torch.classes.__path__ = []

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
vector_store = FAISS.load_local(
    folder_path="vector_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Set the title of the app
st.title("Chat Application")

# User input for the query
user_query = st.text_input("User Query:")

# Retrieve the relevant chunks
if user_query:
    # Retrieve the relevant chunks from the vector store
    relevant_chunks = vector_store.similarity_search(user_query, k=10)
    relevant_chunks = [chunk.page_content for chunk in relevant_chunks]

    for chunk in relevant_chunks:
        print(chunk)

# Collapsible sections
with st.expander("Retrieved Chunks", expanded=False):
    st.write("This section contains the retrieved chunks relevant to the user query.")
    # Example content, replace with dynamic data
    if user_query:
        chunks_str = ""
        for i, chunk in enumerate(relevant_chunks):
            chunks_str += f"- Chunk {i + 1}: {chunk}\n"
        st.write(chunks_str)
    else:
        st.write("Please enter a query to see the retrieved chunks.")

with st.expander("Keywords", expanded=False):
    st.write("This section displays the keywords extracted from the user query.")
    # Example content, replace with dynamic data
    if user_query:
        keywords = extract_keywords(user_query, relevant_chunks)
        keyword_str = ""
        for i, keyword in enumerate(keywords):
            keyword_str += f"- Keyword {i + 1}: {keyword}\n"
        st.write(keyword_str)
    else:
        st.write("Please enter a query to see the extracted keywords.")

with st.expander("Relevant Context", expanded=False):
    st.write("This section provides relevant context pertaining to the user query.")
    # Example content, replace with dynamic data
    if user_query:
        selected_sent = select_sentences(corpus=relevant_chunks, keywords=keywords)
        context_str = ""
        context_str2 = ""
        for i, sent in enumerate(selected_sent):
            context_str += f"- Sentence {i + 1}: {sent}\n"
            context_str2 += f"{sent}\n"
        st.write(context_str)
    else:
        st.write("Please enter a query to see the relevant context.")

# Final Response Section
st.subheader("Final Response")
if user_query:
    st.write("This is where the final response to the user's query will be displayed.")
    # Example content, replace with dynamic response
    final_prompt = f"""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context_str2}?\n
    Question: \n{user_query}\n

    Answer:
    """

    message = llm.invoke(final_prompt)
    st.write(message.content)

else:
    st.write("Please enter a query to see the final response.")
