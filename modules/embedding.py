import numpy as np
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Import our alternative embedder
from modules.embedding_alternative import get_embedder

def parse_embedding(embedding_str):
    """Parse embedding string to numpy array with robust error handling."""
    import re
    import ast
    
    try:
        if isinstance(embedding_str, (list, np.ndarray)):
            # Already a list or array
            return np.array(embedding_str)
        
        if not isinstance(embedding_str, str):
            # Try to convert to string if possible
            try:
                embedding_str = str(embedding_str)
            except:
                return None
        
        # Clean the string
        embedding_str = embedding_str.replace('\xa0', ' ').strip()
        
        # Handle different formats
        if embedding_str.startswith('[') and embedding_str.endswith(']'):
            try:
                # Try using ast.literal_eval which is safer than eval
                return np.array(ast.literal_eval(embedding_str))
            except (SyntaxError, ValueError):
                # If that fails, try pattern matching
                pattern = r'\[(.*)\]'
                match = re.search(pattern, embedding_str)
                if match:
                    values = match.group(1).split(',')
                    return np.array([float(v.strip()) for v in values])
        
        # Split by commas or spaces if not in standard format
        try:
            values = re.split(r'[,\s]+', embedding_str.strip('[]'))
            return np.array([float(v.strip()) for v in values if v.strip()])
        except:
            pass
        
    except Exception as e:
        print(f"Error parsing embedding: {e}")
    
    return None

def create_vector_store(texts, embedding_model):
    """
    Create a FAISS vector store from a list of texts.
    
    Parameters:
        texts (list): List of text strings to embed
        embedding_model: The embedding model to use
        
    Returns:
        FAISS: The vector store containing the embedded texts
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = text_splitter.split_text("\n\n".join(texts))

    vector_store = FAISS.from_texts(split_texts, embedding_model)
    
    return vector_store

def retrieve_relevant_data(query, vector_store, embedding_model, k=5):
    """
    Retrieve relevant data from a vector store based on a query.
    
    Parameters:
        query (str): The query to search for
        vector_store: The vector store to search in
        embedding_model: The embedding model to use
        k (int): Number of results to return
        
    Returns:
        str: The formatted relevant data
    """
    query_embedding = embedding_model.embed_query(query)
    docs = vector_store.similarity_search_by_vector(query_embedding, k=k)

    if not docs:
        st.write("❌ No relevant documents retrieved!")
        return "No relevant information found."

    # Debugging: Display retrieved documents
    st.write("🔍 **Searching the doc...**")
    formatted_docs = []
    
    for i, doc in enumerate(docs):
        from modules.text_processing import format_csv_text
        structured_text = format_csv_text(doc.page_content)
        formatted_docs.append(structured_text)

    return "\n\n".join(formatted_docs) if formatted_docs else "No relevant information found."

def generate_embeddings(texts, model_name="bert-base-uncased"):
    """
    Generate embeddings for a list of texts.
    
    Parameters:
        texts (list): List of text strings to embed
        model_name (str): Name of the transformer model to use
        
    Returns:
        numpy.ndarray: The embeddings for the texts
    """
    # Use our alternative embedder instead of SentenceTransformer
    model = get_embedder(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings 
