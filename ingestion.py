import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from pinecone import Pinecone
from constants import INDEX_NAME


pc = Pinecone()

def ingest_docs():
    loader = ReadTheDocsLoader(
        "/Users/byronmackay/Dev/AI/udemy-lang-chain-course/documentation-helper/langchain-docs/api.python.langchain.com/en/latest/"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeLangChain.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
