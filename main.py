from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
from typing import Set

st.header("LangChain Documentation Chat")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []    


def create_sources_string(sources_urls: Set[str]) -> str:
    if not sources:
        return ""
    source_list = list(sources_urls)
    source_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources):
        sources_string += f"{i+1}. {source}\n"
    return sources_string

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        print(generated_response)
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = f"{generated_response['answer']}"

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append((prompt, generated_response["answer"]))

if st.session_state["chat_answers_history"]:
    for user_query, generated_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]):
        message(user_query, is_user=True)
        message(generated_response)


