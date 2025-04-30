# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from modules.embedding_alternative import get_embedder, get_embeddings
from modules.visualization import create_pareto_chart
from modules.embedding import parse_embedding
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Any
from functools import lru_cache
import logging
import os
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import re  # For text normalization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory for intermediate results
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def safe_write_file(file_path, data, mode='w'):
    """Write data to file with proper encoding and error handling."""
    try:
        if mode == 'wb':
            with open(file_path, mode) as f:
                f.write(data)
        else:
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(data)
        return True
    except Exception as e:
        logger.error(f"Error writing file {file_path}: {e}")
        return False

def safe_read_file(file_path, mode='r'):
    """Read data from file with proper encoding and error handling."""
    try:
        if mode == 'rb':
            with open(file_path, mode) as f:
                return f.read(), True
        else:
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, mode, encoding=encoding) as f:
                        return f.read(), True
                except UnicodeDecodeError:
                    continue
        logger.error(f"Failed to read file {file_path} with any encoding")
        return None, False
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None, False

def save_embeddings_cache(file_hash: str, embeddings: np.ndarray) -> None:
    """Save embeddings to cache file."""
    try:
        cache_file = os.path.join(CACHE_DIR, f"{file_hash}_embeddings.npy")
        np.save(cache_file, embeddings)
        logger.info(f"Saved embeddings to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save embeddings cache: {str(e)}")

def load_embeddings_cache(file_hash: str) -> Union[np.ndarray, None]:
    """Load embeddings from cache file if exists."""
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}_embeddings.npy")
    try:
        if os.path.exists(cache_file):
            embeddings = np.load(cache_file)
            logger.info(f"Loaded embeddings from cache: {cache_file}")
            return embeddings
    except Exception as e:
        logger.warning(f"Failed to load embeddings cache: {str(e)}")
    return None

def compute_file_hash(file_content: bytes) -> str:
    """Compute hash of file content for caching."""
    import hashlib
    return hashlib.md5(file_content).hexdigest()

@lru_cache(maxsize=100)
def get_cluster_labels(embeddings_hash: str, n_clusters: int) -> np.ndarray:
    """Get cluster labels with caching based on embeddings hash."""
    cache_file = os.path.join(CACHE_DIR, f"{embeddings_hash}_clusters_{n_clusters}.npy")
    
    # Try to load from cache
    try:
        if os.path.exists(cache_file):
            cluster_labels = np.load(cache_file)
            logger.info(f"Loaded clusters from cache: {cache_file}")
            return cluster_labels
    except Exception as e:
        logger.warning(f"Failed to load clusters cache: {str(e)}")
    
    # Load embeddings
    embeddings_file = os.path.join(CACHE_DIR, f"{embeddings_hash}_embeddings.npy")
    try:
        embeddings = np.load(embeddings_file)
    except Exception:
        logger.error(f"Cannot find embeddings file: {embeddings_file}")
        # Return dummy clusters
        return np.zeros(10, dtype=int)
    
    # Compute clusters
    logger.info(f"Computing {n_clusters} clusters for {len(embeddings)} embeddings")
    
    # Use K-means which is faster than hierarchical clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Save to cache
    try:
        np.save(cache_file, cluster_labels)
        logger.info(f"Saved clusters to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save clusters cache: {str(e)}")
    
    return cluster_labels

def process_csv_and_visualize(csv_file, n_clusters: int = 5, embedding_model: Any = None, encoding: str = 'utf-8') -> Tuple[pd.DataFrame, Dict]:
    """Process CSV file and create Pareto charts with optimized performance."""
    try:
        # Check pandas version to determine if 'errors' parameter is supported
        pd_version = pd.__version__
        has_errors_param = int(pd_version.split('.')[0]) > 1 or (int(pd_version.split('.')[0]) == 1 and int(pd_version.split('.')[1]) >= 3)
        
        # Function to read CSV with version-appropriate parameters
        def read_csv_safe(file, enc):
            try:
                if has_errors_param:
                    return pd.read_csv(file, encoding=enc)
                else:
                    return pd.read_csv(file, encoding=enc)
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with {enc}, trying ISO-8859-1 instead")
                if has_errors_param:
                    return pd.read_csv(file, encoding='ISO-8859-1')
                else:
                    return pd.read_csv(file, encoding='ISO-8859-1')
        
        # Make sure csv_file is either a path, file-like object, or bytes
        if isinstance(csv_file, str):
            logger.info(f"Reading CSV file from path: {csv_file}")
            df = read_csv_safe(csv_file, encoding)
            with open(csv_file, 'rb') as f:
                file_content = f.read()
        elif isinstance(csv_file, bytes):
            logger.info("Reading CSV from bytes")
            file_content = csv_file
            # Create a file-like object from bytes for pandas to read
            from io import BytesIO
            df = read_csv_safe(BytesIO(csv_file), encoding)
        else:
            logger.info("Reading CSV file from upload")
            file_content = csv_file.read()
            csv_file.seek(0)  # Reset position for reading again
            df = read_csv_safe(csv_file, encoding)
            csv_file.seek(0)  # Reset position just in case
        
        # Remove rows where 'Failure_Description_Translated' is "blank" or missing
        if 'Failure_Description_Translated' in df.columns:
            # Clean string values before filtering - with exception handling
            try:
                df['Failure_Description_Translated'] = df['Failure_Description_Translated'].astype(str)
                df['Failure_Description_Translated'] = df['Failure_Description_Translated'].apply(
                    lambda x: x.replace('\xa0', ' ').replace('\x00', '').strip() if isinstance(x, str) else str(x).strip()
                )
                df = df[df["Failure_Description_Translated"].str.strip().str.lower() != "blank"]
            except Exception as e:
                logger.warning(f"Error cleaning Failure_Description_Translated: {str(e)}")
                # Simpler fallback approach
                df = df[df["Failure_Description_Translated"].astype(str).str.strip().str.lower() != "blank"]
        else:
            logger.warning("Failure_Description_Translated column not found")
            
        # Ensure required columns exist with proper cleaning
        required_columns = ["Failure_Description_Translated", "Problem_Analysis_Translated", "Failure_Symptom_Label"]
        for col in required_columns:
            if col in df.columns:
                # Clean string values with exception handling
                try:
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(
                        lambda x: x.replace('\xa0', ' ').replace('\x00', '').strip() if isinstance(x, str) else str(x).strip()
                    )
                except Exception as e:
                    logger.warning(f"Error cleaning column {col}: {str(e)}")
                    # Simple fallback
                    df[col] = df[col].astype(str)
                
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Extract texts
        texts = df["Failure_Description_Translated"].tolist()
        failure_cat = df["Failure Category"].tolist() if "Failure Category" in df.columns else []
        
        # Clean texts by removing non-printable characters
        clean_texts = []
        for text in texts:
            try:
                # Try the full regex first
                cleaned = re.sub(r'[^\x20-\x7E]', ' ', str(text)).strip()
                clean_texts.append(cleaned)
            except Exception as e:
                # Fallback to manual character replacement
                try:
                    text_str = str(text)
                    text_str = text_str.replace('\xa0', ' ')
                    text_str = text_str.replace('\x00', '')
                    clean_texts.append(text_str.strip())
                except:
                    # Last resort - use an empty string
                    clean_texts.append("")
                    logger.warning(f"Failed to clean text: {text[:30]}...")
        
        texts = clean_texts
        
        # Also clean failure_cat if it exists
        if failure_cat:
            clean_failure_cat = []
            for text in failure_cat:
                try:
                    cleaned = re.sub(r'[^\x20-\x7E]', ' ', str(text)).strip()
                    clean_failure_cat.append(cleaned)
                except:
                    clean_failure_cat.append(str(text).strip())
            failure_cat = clean_failure_cat
        
        # Check if Clustering and Embeddings Exist
        clustering_exists = "Cluster_Hierarchical_Label" in df.columns
        embeddings_exist = "Embeddings" in df.columns
        
        if embeddings_exist:
            # Convert string embeddings back to numpy array
            embeddings_list = df["Embeddings"].apply(parse_embedding).dropna().tolist()
            
            # Filter out None values and empty embeddings
            embeddings_list = [emb for emb in embeddings_list if emb is not None and len(emb) > 0]
            
            if not embeddings_list:
                logger.warning("No valid embeddings found in CSV. Generating new embeddings.")
                embedder = get_embedder()
                embeddings = embedder.encode(texts)
                df = df.iloc[:len(embeddings)].copy()
                df["Embeddings"] = [list(embedding) for embedding in embeddings]
            else:
                # Check the dimensions of embeddings and ensure they're consistent
                embedding_shapes = [len(emb) for emb in embeddings_list]
                unique_shapes = set(embedding_shapes)
                
                if len(unique_shapes) > 1:
                    # If we have inconsistent shapes, use the most common shape
                    from collections import Counter
                    shape_counts = Counter(embedding_shapes)
                    most_common_shape = shape_counts.most_common(1)[0][0]
                    logger.warning(f"Found embeddings with inconsistent shapes. Using most common shape: {most_common_shape}")
                    
                    # Filter embeddings to keep only those with the most common shape
                    embeddings_list = [emb for emb in embeddings_list if len(emb) == most_common_shape]
                    
                    # Filter DataFrame to match the filtered embeddings
                    valid_indices = [i for i, emb in enumerate(df["Embeddings"].apply(parse_embedding)) 
                                    if emb is not None and len(emb) == most_common_shape]
                    df = df.iloc[valid_indices].copy().reset_index(drop=True)
                
                # Now we can safely convert to numpy array
                embeddings = np.array(embeddings_list)
        else:
            # Generate new embeddings if not in CSV
            embedder = get_embedder()
            embeddings = embedder.encode(texts)
            # Ensure DataFrame length matches embeddings length
            df = df.iloc[:len(embeddings)].copy()
            df["Embeddings"] = [list(embedding) for embedding in embeddings]
        
        if clustering_exists:
            # Use existing hierarchical cluster labels
            cluster_labels = df["Cluster_Hierarchical_Label"].to_numpy()
        else:
            # Perform Hierarchical Clustering if not in CSV
            condensed_distance_matrix = pdist(embeddings, metric="cosine")
            linkage_matrix = linkage(condensed_distance_matrix, method="average")
            cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
            df["Cluster_Hierarchical_Label"] = cluster_labels
        
        # Store data in session state
        st.session_state["embeddings"] = embeddings
        st.session_state["df"] = df
        st.session_state["cluster_labels"] = cluster_labels
        
        # Perform Dimensionality Reduction with error handling
        try:
            if len(embeddings) > 1 and embeddings.shape[1] > 1:
                # Use t-SNE for visualization
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1), n_iter=1000)
                reduced_embeddings = reducer.fit_transform(embeddings)
                df["Reduced_Embeddings"] = [list(embedding) for embedding in reduced_embeddings]
            else:
                # If we have too few samples or features, create dummy coordinates
                logger.warning("Not enough samples or features for dimensionality reduction. Using dummy coordinates.")
                reduced_embeddings = np.zeros((len(embeddings), 2))
                df["Reduced_Embeddings"] = [list(embedding) for embedding in reduced_embeddings]
        except Exception as e:
            logger.warning(f"Error in dimensionality reduction: {str(e)}. Using dummy coordinates.")
            reduced_embeddings = np.zeros((len(embeddings), 2))
            df["Reduced_Embeddings"] = [list(embedding) for embedding in reduced_embeddings]
        
        # Count Cluster Frequencies
        cluster_counts = df["Cluster_Hierarchical_Label"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster_Hierarchical_Label", "Count"]
        
        # Identify clusters with count > 400
        clusters_to_remove = cluster_counts[cluster_counts["Count"] > 400]["Cluster_Hierarchical_Label"].tolist()
        
        # Remove rows where cluster is in the removal list
        df_filtered = df[~df["Cluster_Hierarchical_Label"].isin(clusters_to_remove)].reset_index(drop=True)
        
        # Recompute cluster counts after filtering
        if not df_filtered.empty:
            cluster_counts = df_filtered["Cluster_Hierarchical_Label"].value_counts().reset_index()
            cluster_counts.columns = ["Cluster_Hierarchical_Label", "Count"]
            cluster_counts = cluster_counts.sort_values(by="Count", ascending=False)
            
            total_count = cluster_counts["Count"].sum()
            cluster_counts["Cumulative %"] = (cluster_counts["Count"].cumsum() / total_count) * 100
            cluster_counts["Sorted Index"] = range(1, len(cluster_counts) + 1)
        else:
            raise ValueError("All clusters with more than 400 occurrences were removed, leaving no data.")
        
        # Create Level 1 Pareto Chart
        fig_level1 = go.Figure()
        
        # Bar Chart: Failure Count
        fig_level1.add_trace(go.Bar(
            x=cluster_counts["Cluster_Hierarchical_Label"],
            y=cluster_counts["Count"],
            name="Failure Mode Count",
            marker=dict(color="royalblue"),
            yaxis="y",
            textposition="outside"
        ))
        
        # Line Chart: Cumulative Percentage
        fig_level1.add_trace(go.Scatter(
            x=cluster_counts["Cluster_Hierarchical_Label"],
            y=round(cluster_counts["Cumulative %"], 2),
            name="Cumulative %",
            mode="lines+markers",
            yaxis="y2",
            marker=dict(color="red", size=6),
            line=dict(color="red", dash="dash")
        ))
        
        # Format Layout
        fig_level1.update_layout(
            title=dict(
                text="📊 Pareto Analysis of Failure Mode",
                font=dict(size=18, family="Arial", color="black"),
                x=0.5
            ),
            xaxis=dict(
                title="Failure Mode Clusters",
                tickangle=-30,
                showgrid=True,
                tickfont=dict(size=16, family="Arial")
            ),
            yaxis=dict(
                title="Failure Mode Count",
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
        
        # Store charts in dictionary
        charts = {
            'level1': fig_level1,
            'cluster_counts': cluster_counts,
            'df_filtered': df_filtered
        }
        
        return df, charts
        
    except Exception as e:
        logger.error(f"Error in process_csv_and_visualize: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def create_pareto_chart(data: pd.Series, title: str) -> plt.Figure:
    """Create a Pareto chart with optimized performance."""
    # Handle empty data
    if len(data) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available", 
               horizontalalignment='center', verticalalignment='center')
        plt.title(title)
        return fig
    
    # Limit data to improve chart readability
    if len(data) > 20:
        data = data.nlargest(20)
        title = f"{title} (Top 20)"
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Calculate cumulative percentage
    total = data.sum()
    if total == 0:
        # Handle zero-sum case
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data with non-zero values", 
               horizontalalignment='center', verticalalignment='center')
        plt.title(title)
        return fig
        
    cumulative = data.cumsum()
    percentage = (data / total) * 100
    cumulative_percentage = (cumulative / total) * 100
    
    # Plot bars - improved readability with autoscaling
    bars = ax1.bar(range(len(data)), data, color='steelblue')
    ax1.set_xticks(range(len(data)))
    
    # Format x-axis labels for better readability
    if isinstance(data.index[0], str):
        # Truncate long labels
        labels = [label[:30] + '...' if len(str(label)) > 30 else label for label in data.index]
    else:
        labels = data.index
        
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Failure Types')
    ax1.set_ylabel('Count', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    # Plot cumulative line
    ax2 = ax1.twinx()
    ax2.plot(range(len(data)), cumulative_percentage, color='red', marker='o', linestyle='-', linewidth=2)
    ax2.set_ylabel('Cumulative Percentage', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 105])  # Leave room for markers
    
    # Add percentage labels above points
    for i, percentage in enumerate(cumulative_percentage):
        ax2.annotate(f'{percentage:.1f}%', 
                    (i, percentage), 
                    textcoords="offset points",
                    xytext=(0,5), 
                    ha='center',
                    fontsize=8)
    
    # Add 80% reference line for Pareto principle
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7)
    ax2.text(len(data)-1, 80, '80%', color='gray', ha='right', va='bottom')
    
    # Set title and adjust layout
    plt.title(title)
    plt.tight_layout()
    
    return fig

def process_level2_failure_item(df, df_filtered, cluster_counts):
    """
    Process Level 2 Pareto analysis for Failure Items.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame
        df_filtered (pd.DataFrame): The filtered DataFrame
        cluster_counts (pd.DataFrame): DataFrame with cluster counts
    """
    # Check if Failure Item clustering exists
    if "Failure_Item_Cluster_Hierarchical_Label" not in df.columns:
        # Generate embeddings and perform clustering for Problem Analysis
        if "Problem_Analysis_Translated" in df.columns:
            texts_item = df["Problem_Analysis_Translated"].tolist()
            model = get_embedder("bert-base-uncased")
            embeddings_item = model.encode(texts_item, show_progress_bar=True)

            # Hierarchical Clustering
            condensed_distance_matrix_item = pdist(embeddings_item, metric="cosine")
            linkage_matrix_item = linkage(condensed_distance_matrix_item, method="average")
            cluster_labels_item = fcluster(linkage_matrix_item, t=50, criterion="maxclust")

            # Store cluster labels in the entire dataset
            df["Failure_Item_Cluster_Hierarchical_Label"] = cluster_labels_item
            st.success("✅ Clustering applied to the entire Problem_Analysis_Translated column.")
        else:
            st.error("⚠️ Problem_Analysis_Translated column not found in the CSV.")
            return

    # Allow user to select a cluster
    selected_cluster_index = st.selectbox(
        "🔍 Second layer: Failure Item", 
        cluster_counts["Cluster_Hierarchical_Label"]
    )

    # Get the actual cluster label from the selected index
    selected_cluster_label = cluster_counts.loc[
        cluster_counts["Cluster_Hierarchical_Label"] == selected_cluster_index, 
        "Cluster_Hierarchical_Label"
    ].values[0]

    # Filter DataFrame for the selected cluster
    df_selected_cluster = df_filtered[df_filtered["Cluster_Hierarchical_Label"] == selected_cluster_label]
    
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

            # Create Level 2 Pareto Chart
            fig_item_cluster = create_pareto_chart(
                data=item_cluster_counts,
                x_column="Failure_Item_Cluster_Hierarchical_Label",
                y_column="Item_Count",
                title=f"📊 Pareto Analysis for {selected_cluster_label}",
                x_title="Failure Item Clusters",
                y_title="Failure Item Count",
                color="indianred"
            )
            st.plotly_chart(fig_item_cluster, use_container_width=True)

            # Process Level 3: Failure Symptom
            process_level3_failure_symptom(df_selected_cluster, item_cluster_counts)
        else:
            st.warning("⚠️ No failure items remain after filtering.")
    else:
        st.warning("⚠️ No data available for this cluster.")

def process_level3_failure_symptom(df_selected_cluster, item_cluster_counts):
    """
    Process Level 3 Pareto analysis for Failure Symptoms.
    
    Parameters:
        df_selected_cluster (pd.DataFrame): DataFrame filtered for the selected cluster
        item_cluster_counts (pd.DataFrame): DataFrame with item cluster counts
    """
    # Allow user to select a Failure Item Cluster
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

        # Create Level 3 Pareto Chart
        st.subheader(f"📊 Level 3: Pareto Analysis of Failure Symptoms in {selected_failure_item_cluster}")
        
        fig_symptom_cluster = create_pareto_chart(
            data=failure_symptom_counts,
            x_column="Failure_Symptom_Label",
            y_column="Symptom_Count",
            title=f"📊 Pareto Analysis for {selected_failure_item_cluster}",
            x_title="Failure Symptom",
            y_title="Failure Symptom Count",
            color="lightgreen"
        )
        st.plotly_chart(fig_symptom_cluster, use_container_width=True)
    else:
        st.warning("⚠️ Failure_Symptom_Label column not found in the CSV.") 
