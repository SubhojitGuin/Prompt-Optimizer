# Import necessary libraries
import os
import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from uuid import uuid4

# Load the environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
print("Environment variables loaded")

# Create the text splitter, embeddings, index, and vector store
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
index = faiss.IndexFlatL2(768)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
print("Text splitter, embeddings, index, and vector store created")

# Load the documents from the directory and add them to the vector store
dir = './docs'
print("Processing files in directory:", dir, "...\n")
for file in os.listdir(dir):
    if file.endswith(".pdf"):
        print("Processing file:", file, end="...")
        all_split_texts = []
        loader = PyPDFLoader(os.path.join(dir, file))
        docs = loader.load()
        split_texts = text_splitter.split_documents(docs)
        all_split_texts.extend(split_texts)

        uuids = [str(uuid4()) for _ in range(len(all_split_texts))]
        vector_store.add_documents(documents=all_split_texts, ids=uuids)
        print("  done")

# Save the vector store locally
vector_store.save_local('vector_store')
print("\nVector store saved")