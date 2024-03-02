import streamlit as st
import langchain_helper as lh
import textwrap

st.title("GPTutor 1.0")
db = lh.create_vector_db_from_word_doc("./chinese_notes.docx")

with st.sidebar: 
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label = "What is the Word bank file?",
            max_chars = 50
        )

        query = st.sidebar.text_area(
            label = "Ask me something about the video?",
            max_chars = 90,
            key="user_message"
        )

        submit_button = st.form_submit_button(label="Submit")

if db:
    response = lh.get_response_from_query(db=db, user_message=query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width = 80))