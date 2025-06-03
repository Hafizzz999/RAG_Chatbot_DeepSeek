import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document  # For handling Word documents
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Function to load, split, and embed data from Word documents into Chroma vector store
def process_word_documents(words):
    """
    Process Word documents through loading, splitting, and embedding.
    Returns vector store instance.
    """
    # Create temporary directory for Word storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded Word documents to temp directory
        word_paths = []
        for word in words:
            path = os.path.join(temp_dir, word.name)
            with open(path, "wb") as f:
                f.write(word.getbuffer())
            word_paths.append(path)
        
        # Load the documents
        documents = []
        for path in word_paths:
            try:
                doc = Document(path)
                text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
                if text:
                    documents.append(text)
                else:
                    print(f"Warning: Document '{path}' contains no readable text.")
            except Exception as e:
                print(f"Error processing document '{path}': {str(e)}")

                # Check if there is any valid text to process
        if not documents:
            raise ValueError("No valid text found in the uploaded Word documents.")
        
        # Split documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=150  
        )
        splits = text_splitter.split_text("\n".join(documents))

        # Validate splits
        if not splits:
            raise ValueError("No valid chunks generated from the Word documents.")
        
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