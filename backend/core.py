import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from pinecone import Pinecone as PC
from langchain_pinecone import Pinecone
from typing import Any, List, Dict

from constants import INDEX_NAME

pc = PC()

def run_llm(query:str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, 
        retriever=docsearch.as_retriever(),
        return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    # print(run_llm(query="Is the version of langchain docs 0.0.148?"))
    print(run_llm(query="What is LangChain and how to I access OpenAI by using it?"))


