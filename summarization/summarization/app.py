import streamlit as st
import os
from utils import get_claim_ids, get_pdfs, extract_text_from_first_two_pages, load_claims_data, get_claim_details
from openai_utils import generate_summary, answer_question

# Set your OpenAI API key
# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []

if 'appended_text' not in st.session_state:
    st.session_state.appended_text = ""

if 'claims_data' not in st.session_state:
    st.session_state.claims_data = load_claims_data("Claims_sample.csv")

# Streamlit App
st.set_page_config(page_title="PDF Summary Generator", layout="wide")
st.title("PDF Summary Generator")

# Sidebar options
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ("Home", "About Our Team", "Products", "Contact Us", "Terms and Conditions")
)

if menu == "Home":
    folder_path = "claim_info"
    claim_ids = get_claim_ids(folder_path)
    selected_claim_id = st.selectbox("Select Claim ID", claim_ids)

    if selected_claim_id:
        pdfs = get_pdfs(selected_claim_id, folder_path)
        select_all = st.checkbox("Select All Documents")

        if select_all:
            selected_pdfs = pdfs
        else:
            selected_pdfs = st.multiselect("Select Documents ", pdfs)

        if selected_pdfs:
            # Ensure text extraction for selected PDFs
            if not st.session_state.appended_text:
                for pdf in selected_pdfs:
                    pdf_path = os.path.join(folder_path, selected_claim_id, pdf)
                    text = extract_text_from_first_two_pages(pdf_path)
                    st.session_state.appended_text += text + "\n"

            icd, claim_amount, created_date = get_claim_details(st.session_state.claims_data, selected_claim_id)

            # Checkbox for custom question
            custom_question = st.checkbox("Ask a Custom Question")

            if custom_question:
                st.header("Ask a Custom Question")

                for msg in st.session_state.history:
                    if msg["role"] == "user":
                        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Input box for user to ask questions
                st.markdown("<div class='input-container'>", unsafe_allow_html=True)
                question = st.text_input("Enter your question:", key="question_input")
                if st.button("Send"):
                    if st.session_state.appended_text and question:
                        answer = answer_question(st.session_state.appended_text, question, st.session_state.history)
                        st.session_state.history.append({"role": "user", "content": question})
                        st.session_state.history.append({"role": "assistant", "content": answer})
                        st.experimental_rerun()
                    else:
                        st.write("Please enter a question first.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                if st.button("Generate Summary"):
                    if st.session_state.appended_text:
                        with st.spinner(f"Generating summary..."):
                            summary = generate_summary(st.session_state.appended_text)
                            st.write("**Summary:**")
                            st.write(f"Claim ICD: {icd} \n Claim Amount: {claim_amount} \n Created Date: {created_date}\n\n{summary}")
                    else:
                        st.write("Could not extract text from the selected PDFs.")

elif menu == "About Our Team":
    st.header("About Our Team")
    st.write("Information about our team goes here.")

elif menu == "Products":
    st.header("Our Products")
    st.write("Information about our products goes here.")

elif menu == "Contact Us":
    st.header("Contact Us")
    st.write("Contact information goes here.")

elif menu == "Terms and Conditions":
    st.header("Terms and Conditions")
    st.write("Terms and conditions information goes here.")

# Footer
st.markdown("---")
st.markdown("© 2024 PDF Summary Generator. All rights reserved.")

# Custom CSS for chat bubble styling
st.markdown("""
    <style>
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        align-self: flex-start;
        max-width: 70%;
        text-align: left;
    }
    .assistant-bubble {
        background-color: #EAEAEA;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        align-self: flex-end;
        max-width: 70%;
        text-align: right;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 400px;
        overflow-y: auto;
        background-color: #F0F0F0;
        padding: 10px;
        border: 1px solid #E0E0E0;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #FFFFFF;
        padding: 10px;
        border-top: 1px solid #E0E0E0;
    }
    </style>
""", unsafe_allow_html=True)
