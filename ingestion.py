import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

if __name__ == "__main__":
    print("Ingestion ...")
    file = PyPDFLoader(r"D:\Fariz\Tugas Kuliah\Semester 8\Langchain\RAG-Opening\pedoman-akademik-untan-2023-2024-ok-compressed_1713439991.pdf")
    files = file.load()
    
    print("Splitting ...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=400,
    )

    texts = text_splitter.split_documents(documents = files)

    print(f"Split into {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # print("Ingesting")

    # #VectorStore
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    # print("Finish")
    