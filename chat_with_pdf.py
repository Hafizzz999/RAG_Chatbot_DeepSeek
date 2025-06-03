import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from utils import process_documents, get_retriever
from utils_words import process_word_documents  # Import for Word document processing
from utils_excels import process_excel_documents  # Import for Excel document processing
from utils_txt import process_txt_documents  # Import for TXT document processing
from utils_powerpoint import process_powerpoint_documents  # Import for PowerPoint document processing
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import json
import os

# Ensure the chat session directory exists
CHAT_SESSION_DIR = "./chat_session"
os.makedirs(CHAT_SESSION_DIR, exist_ok=True)

# Custom prompt template
def get_custom_prompt():
    """Define and return the custom prompt template."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an educational assistant designed to help students understand their textbooks. Follow these guidelines:\n"
"1. Answer questions using only the information from the uploaded PDFs.\n"
"2. Use simple, clear language suitable for a 10th-grade student.\n"
"3. If the answer isn't in the documents, say: 'I cannot find relevant information in the provided documents.'\n"
"4. Do not speculate, assume, or invent information.\n"
"5. Maintain a professional tone and organize responses clearly (e.g., bullet points, step-by-step explanations).\n"
"6. Encourage follow-up questions by asking if further clarification is needed.\n"
"7. Provide examples to clarify concepts when helpful.\n"
"8. Keep answers concise, focused, and exam-friendly."

        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a precise and well-structured answer based on the context above. Ensure your response is easy to understand, includes examples where necessary, and is formatted in a way that students can use it for exams. If applicable, ask if the student needs further clarification."
        )
    ])

# Initialize QA Chain
def initialize_qa_chain():
    if not st.session_state.qa_chain and st.session_state.vector_store:
        llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.3)
        retriever = get_retriever()
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": get_custom_prompt()}
        )
    return st.session_state.qa_chain

# Initialize the chatbot's memory (session states)
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "current_chat_session" not in st.session_state:
        st.session_state.current_chat_session = "default_session"

# Save the current chat session to a JSON file
def save_chat_session(session_name):
    session_file = os.path.join(CHAT_SESSION_DIR, f"{session_name}.json")
    with open(session_file, "w") as f:
        json.dump(st.session_state.messages, f)

# Load a chat session from a JSON file
def load_chat_session(session_name):
    session_file = os.path.join(CHAT_SESSION_DIR, f"{session_name}.json")
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            st.session_state.messages = json.load(f)
    else:
        st.session_state.messages = []

# Delete a chat session
def delete_chat_session(session_name):
    session_file = os.path.join(CHAT_SESSION_DIR, f"{session_name}.json")
    if os.path.exists(session_file):
        os.remove(session_file)


# Sidebar section
def display_sidebar():
    with st.sidebar:
        # New Chat Button
        if st.button("New Chat", key="new_chat_button"):
            # Save the current chat session
            save_chat_session(st.session_state.current_chat_session)
            # Create a new chat session
            st.session_state.current_chat_session = f"session_{len(os.listdir(CHAT_SESSION_DIR)) + 1}"
            st.session_state.messages = []
            st.rerun()  # Refresh the app to reflect the new session

        # Dropdown for Chat History
        with st.expander("Chat History"):
            chat_sessions = [f.split(".")[0] for f in os.listdir(CHAT_SESSION_DIR) if f.endswith(".json")]
            selected_session = st.selectbox("Select a chat session:", chat_sessions, key="chat_history_selectbox")
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button("Load Session", key="load_session_button"):
                    load_chat_session(selected_session)
                    st.session_state.current_chat_session = selected_session
                    st.rerun()  # Refresh the app to reflect the loaded session
            with col2:
                if st.button("🗑️", key="delete_session_button"):
                    delete_chat_session(selected_session)
                    st.rerun()  # Refresh the app to reflect the deletion

        # Dropdown for Instructions
        with st.expander("Instructions"):
            st.info("""
            1. Upload documents.
            2. Click 'Create Knowledge Base'.
            3. Once documents are processed, start chatting with the bot!
            """)
        
        # Dropdown for Upload PDF documents
        with st.expander("PDF Documents"):
            # Streamlit file uploader widget
            pdfs = st.file_uploader(
                "Upload PDF documents", 
                type="pdf",
                accept_multiple_files=True  # Allow multiple file uploads
            )
            
            # Action Button for user to kick off the knowledge base creation process
            # Action Button for user to kick off the knowledge base creation process
            if st.button("Create Knowledge Base", key="create_kb_pdfs"):
                if not pdfs:
                    st.warning("Please upload PDF documents first!")
                    return

                try:
                    with st.spinner("Creating knowledge base... This may take a moment."):
                        vector_store = process_documents(pdfs)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = None  # Reset QA chain when new documents are processed

                    st.success("Knowledge base created!")  # Simple success message after completion

                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")  # Show error if something goes wrong

        # Dropdown for Upload Word documents
        with st.expander("Word Documents"):
            words = st.file_uploader(
                "Upload Word documents", 
                type="docx",
                accept_multiple_files=True  # Allow multiple file uploads
            )
            if st.button("Create Knowledge Base", key="create_kb_words"):
                if not words:
                    st.warning("Please upload Word documents first!")
                    return
                try:
                    with st.spinner("Creating knowledge base... This may take a moment."):
                        vector_store = process_word_documents(words)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = None  # Reset QA chain when new documents are processed
                        st.success("Knowledge base created!")  # Simple success message after completion
                except Exception as e:
                    st.error(f"Error processing Word documents: {str(e)}")  # Show error if something goes wrong

        # Dropdown for Upload TXT documents
        with st.expander("TXT Documents"):
            txts = st.file_uploader(
                "Upload TXT documents", 
                type="txt",
                accept_multiple_files=True  # Allow multiple file uploads
            )
            if st.button("Create Knowledge Base", key="create_kb_txts"):
                if not txts:
                    st.warning("Please upload TXT documents first!")
                    return
                try:
                    with st.spinner("Creating knowledge base... This may take a moment."):
                        vector_store = process_txt_documents(txts)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = None  # Reset QA chain when new documents are processed
                        st.success("Knowledge base created from TXT documents!")  # Simple success message after completion
                except Exception as e:
                    st.error(f"Error processing TXT documents: {str(e)}")  # Show error if something goes wrong

        # Dropdown for Upload Excel documents
        with st.expander("Excel Documents"):
            excels = st.file_uploader(
                "Upload Excel documents", 
                type=["xlsx", "xls"],
                accept_multiple_files=True  # Allow multiple file uploads
            )
            if st.button("Create Knowledge Base", key="create_kb_excels"):
                if not excels:
                    st.warning("Please upload Excel documents first!")
                    return
                try:
                    with st.spinner("Creating knowledge base... This may take a moment."):
                        vector_store = process_excel_documents(excels)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = None  # Reset QA chain when new documents are processed
                        st.success("Knowledge base created from Excel documents!")  # Simple success message after completion
                except Exception as e:
                    st.error(f"Error processing Excel documents: {str(e)}")  # Show error if something goes wrong

        # Dropdown for Upload PowerPoint documents
        with st.expander("PowerPoint Documents"):
            powerpoints = st.file_uploader(
                "Upload PowerPoint documents", 
                type=["pptx", "ppt"],
                accept_multiple_files=True  # Allow multiple file uploads
            )
            if st.button("Create Knowledge Base", key="create_kb_powerpoints"):
                if not powerpoints:
                    st.warning("Please upload PowerPoint documents first!")
                    return
                try:
                    with st.spinner("Creating knowledge base... This may take a moment."):
                        vector_store = process_powerpoint_documents(powerpoints)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = None  # Reset QA chain when new documents are processed
                        st.success("Knowledge base created from PowerPoint documents!")  # Simple success message after completion
                except Exception as e:
                    st.error(f"Error processing PowerPoint documents: {str(e)}")  # Show error if something goes wrong


# Chat interface section
def chat_interface():
    st.title("Let's Chat With Documents")
    st.markdown("Your personal textbook AI chatbot powered by Deepseek-r1:1.5B")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            
            with st.spinner("Fetching information..."):
                try:
                    qa_chain = initialize_qa_chain()
                    
                    if not qa_chain:
                        full_response = "Please create a knowledge base by uploading documents first."
                    else:
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]
                except Exception as e:
                    full_response = f"Error: {str(e)}"

            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Main function
def main():
    initialize_session_state()
    display_sidebar()
    chat_interface()

if __name__ == "__main__":
    main()