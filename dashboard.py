from langchain.llms import Ollama
import ollama
import streamlit as st
from typing import List, Dict, Generator
from collections import Counter
import re
import numpy as np
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from modules.embedding_alternative import get_embedder
import os
import chardet
from io import StringIO
import shutil
from functools import lru_cache
import concurrent.futures
from multiprocessing import Pool, cpu_count
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modularized components
from modules.text_processing import (
    csv_to_text, 
    format_csv_text, 
    extract_failure_descriptions,
    extract_dtc_codes,
    extract_charging_issues,
    extract_past_repairs,
    extract_key_failure_events
)
from modules.visualization import (
    create_pareto_chart,
    hierarchical_clustering_interactive
)
from modules.embedding import (
    create_vector_store,
    retrieve_relevant_data,
    parse_embedding
)
from modules.ollama_integration import (
    ollama_generator,
    summarize_failures
)

# Constants
EMBEDDING_MODEL_NAME = "bert-base-uncased"

# Cache the embedding model
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Cache the Ollama model list
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ollama_models():
    try:
        response = ollama.list()
        if "models" in response and isinstance(response["models"], list):
            return [model.model for model in response["models"]]
        return []
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {e}")
        return []

# Cache text processing results
@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_text_file(file_content: str, keyword: str) -> List[str]:
    try:
        failure_descriptions = extract_failure_descriptions(file_content, keyword)
        return failure_descriptions
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return []

# Cache LLM summaries - note the _llm parameter to prevent hashing issues
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_llm_summary(failure_descriptions: List[str], _llm) -> str:
    try:
        return summarize_failures(failure_descriptions, _llm)
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return f"Error generating summary: {str(e)}"

# Cache CSV processing
@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_csv_file(csv_data, encoding=None) -> pd.DataFrame:
    try:
        # Check pandas version to determine if 'errors' parameter is supported
        pd_version = pd.__version__
        has_errors_param = int(pd_version.split('.')[0]) > 1 or (int(pd_version.split('.')[0]) == 1 and int(pd_version.split('.')[1]) >= 3)
        
        # If encoding is provided, use it; otherwise try multiple encodings
        if encoding:
            if has_errors_param:
                return pd.read_csv(csv_data, encoding=encoding, errors='replace')
            else:
                return pd.read_csv(csv_data, encoding=encoding)
        
        # Try different encodings in this specific order
        encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                if has_errors_param:
                    return pd.read_csv(csv_data, encoding=enc, errors='replace')
                else:
                    return pd.read_csv(csv_data, encoding=enc)
            except UnicodeDecodeError:
                continue
        
        # If all fail, use a fallback with error replacement if available
        if has_errors_param:
            return pd.read_csv(csv_data, encoding='iso-8859-1', errors='replace')
        else:
            return pd.read_csv(csv_data, encoding='iso-8859-1')
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return pd.DataFrame()

# Function to detect encoding with multiple attempts
def detect_encoding_robust(raw_data):
    """Detect encoding with multiple fallback mechanisms."""
    # First try chardet
    encoding_result = chardet.detect(raw_data)
    encoding_detected = encoding_result['encoding']
    confidence = encoding_result['confidence']
    
    # If chardet is confident enough, use that encoding
    if encoding_detected and confidence > 0.8:
        return encoding_detected, confidence
    
    # Try a series of common encodings
    encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            raw_data.decode(encoding)
            return encoding, 1.0  # Found a working encoding
        except UnicodeDecodeError:
            continue
    
    # If nothing works, return a safe fallback
    return 'iso-8859-1', 0.5  # Last resort fallback

# DTC Meanings Dictionary
DTC_MEANINGS = {
    "0x224F81": "Function: AC charging not available",
    "0x031B93": "Load, load function: Load error detected by load management",
    # ... other DTC codes ...
}

# UI Configuration
st.set_page_config(layout="wide")
tabs = st.tabs(["🤖 Chatbot", "📊 3-level RMA Pareto Chart"])

# Tab 1: Chatbot
with tabs[0]:
    st.title("🤖 RMA Root Cause Assistant")
    st.markdown(
        """
        ### 📌 **Understand what the chatbot summarizes**
        
        The chatbot returns the summary of the **Failure Description** and **Problem Analysis** based on the 
        <span style="color:red; font-weight:bold;">Root Cause</span> labeled in the RMA data.
        
        🔍 **Please upload a TXT file** to analyze failure modes and classify them.
        """,
        unsafe_allow_html=True
    )
    
    # User Input for Keyword
    user_keyword = st.text_input("🔍 Enter a Keyword to Search for Failures:", "Leakage")
    
    # Get cached Ollama models
    model_names = get_ollama_models()
    if not model_names:
        st.error("No Ollama models found. Ensure Ollama is running.")
        st.stop()
        
    # Model selection
    selected_model = st.selectbox("Select a model:", model_names)
    llm = Ollama(model=selected_model)
    
    # Upload TXT File
    uploaded_file = st.file_uploader("📄 Upload TXT File", type=["txt"])
    
    # Process uploaded file with caching
    if uploaded_file:
        try:
            # Read raw bytes from file
            raw_data = uploaded_file.read()
            
            # Detect encoding
            encoding_detected, confidence = detect_encoding_robust(raw_data)
            
            # Debug information
            st.info(f"Detected encoding: {encoding_detected} with {confidence:.2f} confidence")
            
            # Handle cases where encoding detection might fail
            if encoding_detected is None or confidence < 0.8:
                encoding_detected = "ISO-8859-1"  # Default fallback
                st.warning(f"Low confidence in encoding detection. Using {encoding_detected} as fallback.")
            
            # Decode using the detected encoding
            try:
                text_content = raw_data.decode(encoding_detected, errors="replace")  # Replace bad characters
                
                # Normalize text by replacing problematic characters
                text_content = text_content.replace("\xa0", " ")  # Replace non-breaking space
                text_content = text_content.replace("\x00", "")   # Remove null bytes
                text_content = re.sub(r'[^\x00-\x7F]+', ' ', text_content)  # Replace non-ASCII chars
            except UnicodeDecodeError:
                st.warning(f"Could not decode file using {encoding_detected}. Trying ISO-8859-1 instead...")
                text_content = raw_data.decode("ISO-8859-1", errors="replace")  # Force fallback encoding
                
                # Normalize text
                text_content = text_content.replace("\xa0", " ")
                text_content = text_content.replace("\x00", "")
                text_content = re.sub(r'[^\x00-\x7F]+', ' ', text_content)
            
            # Use cached text processing
            failure_descriptions = process_text_file(text_content, user_keyword)
            
            if failure_descriptions:
                # Use cached LLM summary with underscore prefix to avoid hashing
                cleaned_summary = get_llm_summary(failure_descriptions, _llm=llm)
                
                # Display results
                st.success(f"Results found for: **{user_keyword}**")
                st.subheader("📋 Extracted Descriptions:")

                if isinstance(failure_descriptions, list):
                    failure_descriptions = "\n\n".join(failure_descriptions)
                st.text_area("Extracted Descriptions:", failure_descriptions, height=200)
                
                st.subheader("📊 Summary of Failure Descriptions:")
                st.write(cleaned_summary)
            else:
                st.warning(f"No descriptions matching '{user_keyword}' were found in the file.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please try uploading the file again or check if the file is corrupted.")
            import traceback
            st.code(traceback.format_exc())

# Tab 2: 3-level RMA Pareto Chart
with tabs[1]:
    st.title("📊 3-level RMA Pareto Chart")
    st.markdown(
        """
        ### 📌 **Understanding the Clustering Process**
        
        The clusters are grouped based on **text embeddings** calculated using the `transformers` library with model **`bert-base-uncased`**.  

        - The **Failure Description** in RMA data is used for defining **Failure Mode**.
        - The **Problem Analysis** data is used for defining **Failure Item** and **Failure Symptom**.
        
        🔍 **Please upload a CSV file** to analyze RMA data in tree-based Pareto Chart.
        """,
        unsafe_allow_html=True
    )
    
    n_clusters = 20
    
    # Upload CSV file
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"], key="failure_classification_upload")
    
    if csv_file:
        try:
            # Read the file content as bytes
            csv_bytes = csv_file.read()
            
            # Detect encoding
            encoding_detected, confidence = detect_encoding_robust(csv_bytes)
            
            # Debug information
            st.info(f"Detected CSV encoding: {encoding_detected} with {confidence:.2f} confidence")
            
            # Handle cases where encoding detection might fail
            if encoding_detected is None or confidence < 0.5:
                encoding_detected = "ISO-8859-1"  # Default fallback
                st.warning(f"Low confidence in CSV encoding detection. Using {encoding_detected} as fallback.")

            # Try multiple approaches to read the CSV safely
            try:
                # First try: Direct approach with detected encoding
                from io import BytesIO
                # Check pandas version for errors parameter
                pd_version = pd.__version__
                has_errors_param = int(pd_version.split('.')[0]) > 1 or (int(pd_version.split('.')[0]) == 1 and int(pd_version.split('.')[1]) >= 3)
                
                if has_errors_param:
                    df = pd.read_csv(BytesIO(csv_bytes), encoding=encoding_detected, errors="replace")
                else:
                    df = pd.read_csv(BytesIO(csv_bytes), encoding=encoding_detected)
            except Exception as e1:
                st.warning(f"First CSV parsing attempt failed: {str(e1)}")
                try:
                    # Second try: Convert bytes to string first
                    csv_str = csv_bytes.decode(encoding_detected, errors="replace")
                    
                    # Clean problematic characters
                    csv_str = csv_str.replace("\xa0", " ")  # Replace non-breaking space
                    csv_str = csv_str.replace("\x00", "")   # Remove null bytes
                    csv_str = re.sub(r'[^\x00-\x7F]+', ' ', csv_str)  # Replace non-ASCII chars
                    
                    # Convert string content to DataFrame
                    from io import StringIO
                    csv_data = StringIO(csv_str)
                    df = pd.read_csv(csv_data)
                except Exception as e2:
                    st.warning(f"Second CSV parsing attempt failed: {str(e2)}")
                    # Third try: Fallback to using process_csv_file function
                    df = process_csv_file(csv_bytes)
            
            # Display first few rows for debugging
            st.subheader("Preview of uploaded data:")
            st.dataframe(df.head(3))
            
            # Process the CSV file
            from modules.pareto_analysis import process_csv_and_visualize
            
            # Get cached embedding model
            embedding_model = get_embedding_model()
            
            # Use multiprocessing for heavy computations
            with st.spinner("Processing data with multi-processing..."):
                try:
                    # Try to use the already parsed DataFrame if available
                    if 'df' in locals() and not df.empty:
                        # Convert back to bytes if needed by the function
                        from io import BytesIO
                        buffer = BytesIO()
                        df.to_csv(buffer, index=False)
                        buffer.seek(0)
                        processed_df, charts = process_csv_and_visualize(buffer, n_clusters, embedding_model, encoding=encoding_detected)
                    else:
                        # Fall back to original bytes with detected encoding
                        processed_df, charts = process_csv_and_visualize(csv_bytes, n_clusters, embedding_model, encoding=encoding_detected)
                    
                    # Reassign to df for consistency with the rest of the code
                    df = processed_df
                    
                    # Display Level 1 Pareto Chart
                    st.subheader("📊 Level 1: Pareto Analysis of Failure Mode")
                    st.plotly_chart(charts['level1'], use_container_width=True)
                    
                    # Second layer: Failure Item
                    if "Failure_Item_Cluster_Hierarchical_Label" not in df.columns:
                        texts_item = df["Problem_Analysis_Translated"].tolist()
                        model = get_embedder()
                        embeddings_item = model.encode(texts_item, show_progress_bar=True)
                        
                        # Hierarchical Clustering
                        condensed_distance_matrix_item = pdist(embeddings_item, metric="cosine")
                        linkage_matrix_item = linkage(condensed_distance_matrix_item, method="average")
                        cluster_labels_item = fcluster(linkage_matrix_item, t=50, criterion="maxclust")
                        
                        # Store cluster labels in the entire dataset
                        df["Failure_Item_Cluster_Hierarchical_Label"] = cluster_labels_item
                        st.success("✅ Clustering applied to the entire Problem_Analysis_Translated column.")
                    
                    # Allow user to select a cluster
                    selected_cluster_index = st.selectbox(
                        "🔍 Second layer: Failure Item", 
                        charts['cluster_counts']["Cluster_Hierarchical_Label"]
                    )
                    
                    # Get the actual cluster label from the selected index
                    selected_cluster_label = charts['cluster_counts'].loc[
                        charts['cluster_counts']["Cluster_Hierarchical_Label"] == selected_cluster_index, 
                        "Cluster_Hierarchical_Label"
                    ].values[0]
                    
                    # Filter DataFrame for the selected cluster
                    df_selected_cluster = charts['df_filtered'][charts['df_filtered']["Cluster_Hierarchical_Label"] == selected_cluster_label]
                    
                    # Apply hierarchical clustering ONLY within the selected cluster
                    if not df_selected_cluster.empty:
                        st.subheader(f"📊 Level 2: Pareto Analysis of Failure Items in {selected_cluster_label}")
                        
                        # Remove failure item clusters with count > threshold
                        item_cluster_counts = df_selected_cluster["Failure_Item_Cluster_Hierarchical_Label"].value_counts().reset_index()
                        item_cluster_counts.columns = ["Failure_Item_Cluster_Hierarchical_Label", "Item_Count"]
                        
                        item_clusters_to_remove = item_cluster_counts[item_cluster_counts["Item_Count"] > 35]["Failure_Item_Cluster_Hierarchical_Label"].tolist()
                        df_selected_cluster = df_selected_cluster[~df_selected_cluster["Failure_Item_Cluster_Hierarchical_Label"].isin(item_clusters_to_remove)]
                        
                        # Recompute cluster counts after filtering
                        if not df_selected_cluster.empty:
                            item_cluster_counts = df_selected_cluster["Failure_Item_Cluster_Hierarchical_Label"].value_counts().reset_index()
                            item_cluster_counts.columns = ["Failure_Item_Cluster_Hierarchical_Label", "Item_Count"]
                            item_cluster_counts = item_cluster_counts.sort_values(by="Item_Count", ascending=False)
                            
                            item_total_count = item_cluster_counts["Item_Count"].sum()
                            item_cluster_counts["Cumulative %"] = (item_cluster_counts["Item_Count"].cumsum() / item_total_count) * 100
                            
                            # Create Level 2 Pareto Chart
                            fig_item_cluster = go.Figure()
                            
                            # Bar Chart: Failure Count
                            fig_item_cluster.add_trace(go.Bar(
                                x=item_cluster_counts["Failure_Item_Cluster_Hierarchical_Label"],
                                y=item_cluster_counts["Item_Count"],
                                name="Failure Item Count",
                                marker=dict(color="indianred"),
                                yaxis="y",
                                textposition="outside"
                            ))
                            
                            # Line Chart: Cumulative Percentage
                            fig_item_cluster.add_trace(go.Scatter(
                                x=item_cluster_counts["Failure_Item_Cluster_Hierarchical_Label"],
                                y=round(item_cluster_counts["Cumulative %"], 2),
                                name="Cumulative %",
                                mode="lines+markers",
                                yaxis="y2",
                                marker=dict(color="black", size=6),
                                line=dict(color="black", dash="dash")
                            ))
                            
                            # Format Layout
                            fig_item_cluster.update_layout(
                                title=dict(
                                    text=f"📊 Pareto Analysis for {selected_cluster_label}",
                                    font=dict(size=18, family="Arial", color="black"),
                                    x=0.5
                                ),
                                xaxis=dict(
                                    title="Failure Item Clusters",
                                    tickangle=-30,
                                    showgrid=True,
                                    tickfont=dict(size=12, family="Arial")
                                ),
                                yaxis=dict(
                                    title="Failure Item Count",
                                    showgrid=True,
                                    tickfont=dict(size=12, family="Arial")
                                ),
                                yaxis2=dict(
                                    title="Cumulative Percentage (%)",
                                    overlaying="y",
                                    side="right",
                                    range=[0, 100],
                                    showgrid=False,
                                    tickfont=dict(size=12, family="Arial")
                                ),
                                legend=dict(
                                    x=0.8, y=0.85,
                                    font=dict(size=12, family="Arial"),
                                    bgcolor="rgba(255,255,255,0.7)"
                                ),
                                hovermode="x unified",
                                plot_bgcolor="whitesmoke",
                                margin=dict(l=80, r=80, t=80, b=120),
                                height=600
                            )
                            
                            st.plotly_chart(fig_item_cluster, use_container_width=True)
                            
                            # Third layer: Failure Symptom
                            selected_failure_item_cluster = st.selectbox(
                                "🔍 Third Layer: Failure Symptom", 
                                item_cluster_counts["Failure_Item_Cluster_Hierarchical_Label"]
                            )
                            
                            # Filter DataFrame based on the selected Failure Item Cluster
                            df_failure_symptom = df_selected_cluster[
                                df_selected_cluster["Failure_Item_Cluster_Hierarchical_Label"] == selected_failure_item_cluster
                            ]
                            
                            # Count occurrences of Failure Symptoms
                            if "Failure_Symptom_Label" in df_failure_symptom.columns:
                                failure_symptom_counts = df_failure_symptom["Failure_Symptom_Label"].value_counts().reset_index()
                                failure_symptom_counts.columns = ["Failure_Symptom_Label", "Symptom_Count"]
                                failure_symptom_counts = failure_symptom_counts.sort_values(by="Symptom_Count", ascending=False)
                                
                                # Compute cumulative percentage
                                total_count_symptom = failure_symptom_counts["Symptom_Count"].sum()
                                failure_symptom_counts["Cumulative %"] = (failure_symptom_counts["Symptom_Count"].cumsum() / total_count_symptom) * 100
                                
                                # Create Level 3 Pareto Chart
                                st.subheader(f"📊 Level 3: Pareto Analysis of Failure Symptoms in {selected_failure_item_cluster}")
                                
                                fig_symptom_cluster = go.Figure()
                                
                                # Bar Chart: Failure Count
                                fig_symptom_cluster.add_trace(go.Bar(
                                    x=failure_symptom_counts["Failure_Symptom_Label"],
                                    y=failure_symptom_counts["Symptom_Count"],
                                    name="Failure Symptom Count",
                                    marker=dict(color="lightgreen"),
                                    yaxis="y",
                                    textposition="outside"
                                ))
                                
                                # Line Chart: Cumulative Percentage
                                fig_symptom_cluster.add_trace(go.Scatter(
                                    x=failure_symptom_counts["Failure_Symptom_Label"],
                                    y=round(failure_symptom_counts["Cumulative %"], 2),
                                    name="Cumulative %",
                                    mode="lines+markers",
                                    yaxis="y2",
                                    marker=dict(color="black", size=6),
                                    line=dict(color="black", dash="dash")
                                ))
                                
                                # Format Layout
                                fig_symptom_cluster.update_layout(
                                    title=dict(
                                        text=f"📊 Pareto Analysis for {selected_failure_item_cluster}",
                                        font=dict(size=18, family="Arial", color="black"),
                                        x=0.5
                                    ),
                                    xaxis=dict(
                                        title="Failure Symptom",
                                        tickangle=-30,
                                        showgrid=True,
                                        tickfont=dict(size=16, family="Arial")
                                    ),
                                    yaxis=dict(
                                        title="Failure Symptom Count",
                                        showgrid=True,
                                        tickfont=dict(size=12, family="Arial")
                                    ),
                                    yaxis2=dict(
                                        title="Cumulative Percentage (%)",
                                        overlaying="y",
                                        side="right",
                                        range=[0, 100],
                                        showgrid=False,
                                        tickfont=dict(size=12, family="Arial")
                                    ),
                                    legend=dict(
                                        x=0.8, y=0.85,
                                        font=dict(size=12, family="Arial"),
                                        bgcolor="rgba(255,255,255,0.7)"
                                    ),
                                    hovermode="x unified",
                                    plot_bgcolor="whitesmoke",
                                    margin=dict(l=80, r=80, t=80, b=120),
                                    height=600
                                )
                                
                                st.plotly_chart(fig_symptom_cluster, use_container_width=True)
                            else:
                                st.warning("⚠️ Failure_Symptom_Label column not found in the CSV.")
                        else:
                            st.warning("⚠️ No failure items remain after filtering.")
                    else:
                        st.warning("⚠️ No data available for this cluster.")
                    
                    # Show dataframe with cluster assignments
                    st.subheader("Data with Cluster Assignments")
                    display_columns = ['Failure_Description_Translated', 'Problem_Analysis_Translated', 'Cluster_Hierarchical_Label']
                    st.dataframe(df[display_columns])
                    
                    # Download link for the clustered data
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Clustered Data as CSV",
                        data=csv,
                        file_name="clustered_data.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.info("Please check if the CSV file has the required columns or try a different encoding.")
                    import traceback
                    st.code(traceback.format_exc())
                
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
            st.info("Please check if the CSV file has the required columns or try a different encoding.")
            import traceback
            st.code(traceback.format_exc())
