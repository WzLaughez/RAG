from backend.core import run_llm
import streamlit as st


st.header("Question Answering System Pedoman Akademik Universitas Tanjungpura")


prompt = st.text_input("Prompt:", placeholder="Masukkan pertanyaan disini")

if(
    "chat_answer_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
):
    st.session_state["chat_answer_history"] = []
    st.session_state["user_prompt_history"] = []


if prompt:
    with st.spinner("Generating Response..."):
        # Call the backend function to get the answer
        # result = get_answer(prompt)
        generated_response = run_llm(prompt, llm_model_name="llama3")
        formatted_answer = (
            f"{generated_response['answer']} \n\n {generated_response['sources']} " 
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_answer)


if st.session_state["chat_answer_history"]:
    for generated_response, user_prompt in zip(
        st.session_state["chat_answer_history"],
        st.session_state["user_prompt_history"],
    ):
        # Display the chat history
        st.chat_message("user").write(user_prompt)
        st.chat_message("assistant").write(generated_response)