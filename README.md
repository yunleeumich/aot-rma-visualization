# RMA Analytics Dashboard

A dashboard for analyzing RMA (Return Merchandise Authorization) data to identify root causes of failures.

## Features

- Chatbot for querying failure descriptions
- 3-level RMA Pareto Chart for visualizing failure modes
- Text embedding and clustering for failure classification
- Integration with Ollama for LLM-based analysis

## Setup with Poetry

This project uses Poetry for dependency management. Follow these steps to set up the project:

### 1. Install Poetry

If you don't have Poetry installed, run:

```bash
# On Windows
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# On macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
```

Or run the provided setup script:

```bash
python setup_poetry.py
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Activate the Virtual Environment

```bash
poetry shell
```

### 4. Run the Application

```bash
poetry run streamlit run streamlit.py
```

## Dependencies

- Python 3.8
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- SentenceTransformers
- LangChain
- FAISS
- Ollama
- Matplotlib
- HuggingFace Hub

## Project Structure

- `streamlit.py`: Main application file
- `modules/`: Directory containing modularized components
  - `text_processing.py`: Functions for processing text data
  - `visualization.py`: Functions for creating visualizations
  - `embedding.py`: Functions for creating and using embeddings
  - `ollama_integration.py`: Functions for integrating with Ollama
  - `pareto_analysis.py`: Functions for Pareto analysis

## Usage

1. Upload a CSV or TXT file containing RMA data
2. Use the chatbot to query specific failure modes
3. Analyze the 3-level Pareto chart to identify common failure patterns
4. Explore clusters of similar failures
