import os
import tempfile
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Function to load, split, and embed data from Excel documents into Chroma vector store
def process_excel_documents(excels):
    """
    Process Excel documents through loading, splitting, and embedding.
    Returns vector store instance.
    """
    # Create temporary directory for Excel storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded Excel documents to temp directory
        excel_paths = []
        for excel in excels:
            path = os.path.join(temp_dir, excel.name)
            with open(path, "wb") as f:
                f.write(excel.getbuffer())
            excel_paths.append(path)
        
        # Load the documents
        documents = []
        for path in excel_paths:
            try:
                # Read all sheets in the Excel file
                excel_data = pd.read_excel(path, sheet_name=None)
                for sheet_name, sheet_data in excel_data.items():
                    # Convert the sheet data to text
                    text = sheet_data.to_string(index=False, header=True)
                    if text.strip():  # Only add non-empty text
                        documents.append(text)
            except Exception as e:
                print(f"Error processing Excel file '{path}': {str(e)}")
        
        # Check if there is any valid text to process
        if not documents:
            raise ValueError("No valid text found in the uploaded Excel documents.")
        
        # Split documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=150  
        )
        splits = text_splitter.split_text("\n".join(documents))
        
        # Validate splits
        if not splits:
            raise ValueError("No valid chunks generated from the Excel documents.")
        
        # Instantiate the embeddings model
        embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        
        # Create embeddings and vector store
        vector_store = Chroma.from_texts(
            texts=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store

# Initialize and returns a retriever for the vector store
def get_retriever():
    """Initialize and return the vector store retriever"""
    # Initialize the embedding model
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    try:
        # Initialize the vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        # Return the retriever with MMR (Maximum Marginal Relevance) search and k=3
        return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None