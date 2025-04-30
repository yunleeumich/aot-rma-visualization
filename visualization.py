import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

def create_pareto_chart(data, x_column, y_column, title, x_title, y_title, color="royalblue"):
    """
    Create a Pareto chart with a bar chart and cumulative percentage line.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data
        x_column (str): Column name for x-axis
        y_column (str): Column name for y-axis (count)
        title (str): Chart title
        x_title (str): X-axis title
        y_title (str): Y-axis title
        color (str): Color for the bars
        
    Returns:
        plotly.graph_objects.Figure: The Pareto chart figure
    """
    # Sort data by count in descending order
    sorted_data = data.sort_values(by=y_column, ascending=False)
    
    # Calculate cumulative percentage
    total = sorted_data[y_column].sum()
    sorted_data["Cumulative %"] = (sorted_data[y_column].cumsum() / total) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=sorted_data[x_column],
        y=sorted_data[y_column],
        name=y_title,
        marker=dict(color=color),
        yaxis="y",
        textposition="outside"
    ))
    
    # Add cumulative percentage line
    fig.add_trace(go.Scatter(
        x=sorted_data[x_column],
        y=round(sorted_data["Cumulative %"], 2),
        name="Cumulative %",
        mode="lines+markers",
        yaxis="y2",
        marker=dict(color="red", size=6),
        line=dict(color="red", dash="dash")
    ))
    
    # Format layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial", color="black"),
            x=0.5  # Center title
        ),
        xaxis=dict(
            title=x_title,
            tickangle=-30,
            showgrid=True,
            tickfont=dict(size=12, family="Arial")
        ),
        yaxis=dict(
            title=y_title,
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
    
    return fig

def hierarchical_clustering_interactive(embeddings, method="ward", n_clusters=5):
    """
    Performs hierarchical clustering and displays an interactive dendrogram in Streamlit.

    Parameters:
        embeddings (numpy.ndarray): The embeddings to cluster.
        method (str): Linkage method to use ('ward', 'single', 'complete', etc.).
        n_clusters (int): Number of clusters to extract.

    Returns:
        labels (numpy.ndarray): Cluster labels for each embedding.
        linkage_matrix (numpy.ndarray): The hierarchical clustering encoded as a linkage matrix.
    """
    if embeddings is None or len(embeddings) == 0:
        st.error("⚠️ No embeddings available. Please upload a CSV file first.")
        return None, None

    # Compute linkage matrix
    linkage_matrix = linkage(embeddings, method=method)

    # Extract cluster labels
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    return labels, linkage_matrix

def plot_dendrogram(linkage_matrix, labels=None, max_d=None, title="Hierarchical Clustering Dendrogram"):
    """
    Plot a dendrogram of the hierarchical clustering.
    
    Parameters:
        linkage_matrix (numpy.ndarray): The hierarchical clustering encoded as a linkage matrix.
        labels (list): Labels for each leaf node.
        max_d (float): A horizontal cut-off line to show where clusters are formed.
        title (str): Title for the dendrogram.
        
    Returns:
        plotly.graph_objects.Figure: The dendrogram figure
    """
    # Create dendrogram
    fig = go.Figure()
    
    # Get dendrogram data
    dendro = dendrogram(linkage_matrix, labels=labels, no_plot=True)
    
    # Add dendrogram trace
    fig.add_trace(go.Scatter(
        x=dendro['icoord'],
        y=dendro['dcoord'],
        mode='lines',
        line=dict(color='black', width=1),
        hoverinfo='none'
    ))
    
    # Add horizontal cut-off line if specified
    if max_d:
        fig.add_shape(
            type="line",
            x0=0,
            y0=max_d,
            x1=len(dendro['icoord']),
            y1=max_d,
            line=dict(color="red", width=2, dash="dash")
        )
    
    # Format layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, family="Arial", color="black"),
            x=0.5
        ),
        xaxis=dict(
            title="Sample Index",
            showticklabels=False
        ),
        yaxis=dict(
            title="Distance"
        ),
        hovermode="closest",
        plot_bgcolor="white",
        height=600
    )
    
    return fig 
