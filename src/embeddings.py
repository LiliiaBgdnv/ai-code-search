from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper around SentenceTransformer for encoding text.

    - Picks CUDA if available, otherwise CPU.
    - normalizes embeddings (good for cosine similarity).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        # Load model once and move to device
        self.model = SentenceTransformer(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Embedding dimension is useful for index creation
        self.ndim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Union[str, list[str]]) -> np.ndarray:
        """Encode one string or list of strings -> (n, d) numpy array."""
        if isinstance(texts, str):
            texts = [texts]
        return np.asarray(
            self.model.encode(
                texts,
                device=self.device,
                convert_to_numpy=True,  # we want numpy back
                normalize_embeddings=True,  # better for cosine distance
            )
        )
