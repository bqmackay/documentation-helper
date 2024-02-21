import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pinecone import Pinecone as PC
from langchain_pinecone import Pinecone
from typing import Any

from constants import INDEX_NAME

pc = PC()

def run_llm(query:str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=chat, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    # print("######################")
    # docs = docsearch.similarity_search(query)
    # for d in docs:
    #     print(d.page_content)
    #     print("######################")
    # print("######################")
    return qa({"query": query})


if __name__ == "__main__":
    # print(run_llm(query="Is the version of langchain docs 0.0.148?"))
    print(run_llm(query="What is LangChain and how to I access OpenAI by using it?"))


