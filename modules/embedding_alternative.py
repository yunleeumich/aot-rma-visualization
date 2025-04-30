import sys
import logging
from typing import List, Union
from functools import lru_cache
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix for packaging import error
try:
    # Try to ensure packaging is available 
    import pkg_resources
    import packaging
except ImportError:
    logger.info("Installing packaging module...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
    try:
        import packaging
    except ImportError:
        logger.warning("Could not import packaging module - using fallback methods")

# Now try to import torch with error handling
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing torch or transformers: {str(e)}")
    logger.warning("Will use fallback embeddings with reduced functionality")
    TORCH_AVAILABLE = False

# Cache size for embeddings (adjust based on your memory constraints)
CACHE_SIZE = 1000
BATCH_SIZE = 32

@lru_cache(maxsize=CACHE_SIZE)
def get_embedder(model_name: str = "bert-base-uncased"):
    """Get or create the embedder with caching."""
    if not TORCH_AVAILABLE:
        # Return a simple fallback embedder that uses character count as a proxy
        return FallbackEmbedder(model_name)
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        # Fallback to simpler embedder
        return FallbackEmbedder(model_name)

def get_embeddings(texts: Union[str, List[str]], model_name: str = "bert-base-uncased") -> np.ndarray:
    """Get embeddings for input texts with optimized batch processing."""
    if isinstance(texts, str):
        texts = [texts]
    
    if not TORCH_AVAILABLE:
        # Use fallback embedder
        embedder = FallbackEmbedder(model_name)
        return embedder.encode(texts)
    
    tokenizer, model = get_embedder(model_name)
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        # Tokenize with padding and truncation
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**encoded)
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            all_embeddings.append(embeddings)
    
    # Combine all batches
    return np.vstack(all_embeddings)

def get_embedding(text: str, model_name: str = "bert-base-uncased") -> np.ndarray:
    """Get embedding for a single text with caching."""
    return get_embeddings(text, model_name)[0]

class FallbackEmbedder:
    """
    A fallback embedder that doesn't require torch.
    Uses a simple character-based approach instead of ML.
    Much less accurate but works without dependencies.
    """
    def __init__(self, model_name=None):
        self.model_name = model_name or "fallback"
        self.dimension = 128
        
    def encode(self, sentences, batch_size=None, show_progress_bar=False, normalize_embeddings=True):
        """Generate simple fallback embeddings based on character statistics"""
        if isinstance(sentences, str):
            sentences = [sentences]
            
        embeddings = []
        for text in sentences:
            # Create a simple embedding based on character frequency
            embedding = np.zeros(self.dimension)
            
            if not text or len(text) == 0:
                embeddings.append(embedding)
                continue
                
            # Use character code points for first half of embeddings
            for i, char in enumerate(text[:64]):
                pos = i % (self.dimension // 2)
                embedding[pos] += ord(char) / 1000.0
            
            # Use word length statistics for second half
            words = text.split()
            if words:
                word_lengths = [len(word) for word in words]
                avg_len = sum(word_lengths) / len(words)
                embedding[self.dimension // 2] = avg_len / 10.0
                
                # Word count
                embedding[self.dimension // 2 + 1] = len(words) / 100.0
                
                # Character count
                embedding[self.dimension // 2 + 2] = len(text) / 1000.0
                
                # Uppercase ratio
                uppercase_chars = sum(1 for c in text if c.isupper())
                embedding[self.dimension // 2 + 3] = uppercase_chars / len(text)
            
            # Normalize if requested
            if normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                    
            embeddings.append(embedding)
            
        return np.array(embeddings)

class TransformerEmbedder:
    """
    A replacement for SentenceTransformer that uses the transformers library directly
    """
    def __init__(self, model_name="bert-base-uncased"):
        if not TORCH_AVAILABLE:
            logger.warning("Torch not available, using fallback embedder")
            self.fallback = FallbackEmbedder(model_name)
            return
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, sentences, batch_size=8, show_progress_bar=False, normalize_embeddings=True):
        """
        Computes sentence embeddings
        
        :param sentences: List of sentences to encode
        :param batch_size: Batch size for processing
        :param show_progress_bar: Whether to show a progress bar
        :param normalize_embeddings: Whether to normalize embeddings to unit length
        :return: Numpy array with embeddings
        """
        if not TORCH_AVAILABLE:
            return self.fallback.encode(sentences, batch_size, show_progress_bar, normalize_embeddings)
            
        if isinstance(sentences, str):
            sentences = [sentences]
            
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize sentences
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings if requested
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            # Convert to numpy and add to result
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)

# Function to get a pre-configured embedder
def get_embedder(model_name="bert-base-uncased"):
    """
    Returns a TransformerEmbedder instance with the specified model
    """
    return TransformerEmbedder(model_name) 
