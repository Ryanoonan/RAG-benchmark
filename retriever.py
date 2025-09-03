import json
import logging
from typing import Any, Dict, List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """E5-base-v2 based dense retriever"""

    def __init__(self, model_path: str, device: str = "auto"):
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Loading E5 retriever on {self.device}")

        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)

        self.index = None
        self.passages = []

    def load_index(self, index_path: str, passages_path: str):
        """Load pre-built FAISS index and corresponding passages"""
        logger.info(f"Loading pre-built index from {index_path}")
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded index with {self.index.ntotal} passages")

        # Load passages
        logger.info(f"Loading passages from {passages_path}")
        with open(passages_path, "r", encoding="utf-8") as f:
            if passages_path.endswith(".json"):
                self.passages = json.load(f)
            elif passages_path.endswith(".jsonl"):
                self.passages = []
                for line in f:
                    data = json.loads(line.strip())
                    if isinstance(data, str):
                        self.passages.append(data)
                    elif isinstance(data, dict) and "contents" in data:
                        # For our JSONL format with 'contents' field
                        self.passages.append(data["contents"])
                    elif isinstance(data, dict) and "text" in data:
                        self.passages.append(data["text"])
                    else:
                        # Use first string value found
                        for value in data.values():
                            if isinstance(value, str) and len(value.strip()) > 0:
                                self.passages.append(value)
                                break
            else:
                # Assume plain text file with one passage per line
                self.passages = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(self.passages)} passages")

        if len(self.passages) != self.index.ntotal:
            logger.warning(
                f"Mismatch: {len(self.passages)} passages but {self.index.ntotal} in index"
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k passages for a query"""
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(
        self, queries: List[str], top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve top-k passages for a batch of queries"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Add E5 query prefix and encode in batch
        query_embeddings = self.model.encode(
            [f"query: {query}" for query in queries], convert_to_numpy=True
        )

        # Search in batch
        scores, indices = self.index.search(query_embeddings.astype(np.float32), top_k)

        # Process results for each query
        batch_results = []
        for query_idx in range(len(queries)):
            query_results = []
            for score, idx in zip(scores[query_idx], indices[query_idx]):
                query_results.append(
                    {
                        "text": self.passages[idx],
                        "score": float(score),
                        "index": int(idx),
                    }
                )
            batch_results.append(query_results)

        return batch_results
