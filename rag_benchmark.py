#!/usr/bin/env python3
"""
RAG Benchmark Script for PopQA Dataset
Using e5-base-v2 retriever and TinyLlama-1.1B-Chat-v1.0 generator
"""
import argparse
import json
import logging
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import torch
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG benchmark"""

    generator_model_path: str
    # Model paths
    retriever_model_path: str = "intfloat/e5-base-v2"
    # Data paths
    popqa_dataset_path: str = "akariasai/PopQA"
    index_path: str = ""  # Path to pre-built FAISS index
    passages_path: str = ""  # Path to passages file corresponding to the index

    # Retrieval settings
    top_k: int = 5
    max_passage_length: int = 512

    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 0.7

    # Evaluation settings
    batch_size: int = 8
    num_samples: int = 100  # Use -1 for full dataset


class E5Retriever:
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
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Add E5 query prefix
        query_embedding = self.model.encode([f"query: {query}"])

        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append(
                {"text": self.passages[idx], "score": float(score), "index": int(idx)}
            )

        return results

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


class Generator:
    """meta-llama/Meta-Llama-3-8B-Instruct based generator"""

    def __init__(self, model_path: str, device: str = "auto"):
        if not model_path:
            raise ValueError("Generator model path cannot be empty")

        logger.info(f"Loading {model_path} generator")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Generate answer using retrieved passages"""

        # Create RAG prompt
        context = "\n".join(
            [f"Doc {i+1} ({p['text']})" for i, p in enumerate(retrieved_passages)]
        )
        system_prompt = (
            "Answer the question based on the given document."
            "Only give me the answer and do not output any other words."
            "If there are multiple possible answers, provide the single best one."
            f"\nThe following are given documents: \n\n {context}"
        )

        user_prompt = f"Question: {query}"

        messages = [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{user_prompt}"},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
        )

        response = outputs[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(response, skip_special_tokens=True).strip()

    def _build_prompt(
        self, query: str, retrieved_passages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Build RAG prompt messages for a single query"""
        context = "\n".join(
            [f"Doc {i+1} ({p['text']})" for i, p in enumerate(retrieved_passages)]
        )
        system_prompt = (
            "Answer the question based on the given document."
            "Only give me the answer and do not output any other words."
            f"\nThe following are given documents: \n\n {context}"
        )
        user_prompt = f"Question: {query}"

        return [
            {"role": "system", "content": f"{system_prompt}"},
            {"role": "user", "content": f"{user_prompt}"},
        ]

    def generate_batch(
        self,
        queries: List[str],
        retrieved_passages_batch: List[List[Dict[str, Any]]],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> List[str]:
        """Generate answers for a batch of queries using their retrieved passages"""

        # Build prompts for all queries
        all_messages = [
            self._build_prompt(query, retrieved_passages)
            for query, retrieved_passages in zip(queries, retrieved_passages_batch)
        ]

        # Apply chat template for each and tokenize with padding
        prompts = []
        for messages in all_messages:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(prompt)

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode responses
        input_lengths = inputs.input_ids.shape[-1]
        responses = []
        for output in outputs:
            response = output[input_lengths:]
            decoded = self.tokenizer.decode(response, skip_special_tokens=True)
            responses.append(decoded.strip())

        return responses


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval and generation"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = E5Retriever(config.retriever_model_path)
        self.generator = None  # Will be initialized when model path is provided

    def load_generator(self):
        """Load generator model"""
        if not self.config.generator_model_path:
            logger.warning(
                "Generator model path is empty. Generator will not be loaded."
            )
            return
        self.generator = Generator(self.config.generator_model_path)

    def load_popqa_dataset(self) -> List[Dict[str, Any]]:
        """Load PopQA dataset"""
        logger.info(f"Loading PopQA dataset from {self.config.popqa_dataset_path}")

        try:
            dataset = load_dataset(self.config.popqa_dataset_path, split="test")
            samples = list(dataset)

            if self.config.num_samples > 0:
                samples = samples[: self.config.num_samples]

            logger.info(f"Loaded {len(samples)} samples from PopQA")
            return samples

        except Exception as e:
            logger.error(f"Failed to load PopQA dataset: {e}")
            # Fallback to sample data
            return [
                {"question": "What is the capital of France?", "answer": "Paris"},
                {
                    "question": "Who wrote Romeo and Juliet?",
                    "answer": "William Shakespeare",
                },
                {
                    "question": "What is the largest planet in our solar system?",
                    "answer": "Jupiter",
                },
            ]

    def run_benchmark(
        self, output_path: str = "rag_benchmark_results.json"
    ) -> Dict[str, Any]:
        """Run complete RAG benchmark"""
        logger.info("Starting RAG benchmark")

        # Load data
        popqa_data = self.load_popqa_dataset()

        # Load pre-built retrieval index
        if not self.config.index_path or not self.config.passages_path:
            raise ValueError("Both index_path and passages_path must be provided")

        self.retriever.load_index(self.config.index_path, self.config.passages_path)

        # Load generator if path provided
        if self.config.generator_model_path:
            self.load_generator()

        results = []

        # Initialize results file with header
        initial_results = {
            "config": self.config.__dict__,
            "results": [],
            "num_samples": 0,
        }

        output_path_obj = Path(output_path)
        with open(output_path_obj, "w") as f:
            json.dump(initial_results, f, indent=2)

        # Process samples in batches
        for batch_start in range(0, len(popqa_data), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(popqa_data))
            batch_samples = popqa_data[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//self.config.batch_size + 1}/{(len(popqa_data) + self.config.batch_size - 1)//self.config.batch_size} (samples {batch_start+1}-{batch_end})"
            )

            # Extract queries from batch
            queries = [sample["question"] for sample in batch_samples]

            # Retrieve passages in batch
            retrieved_passages_batch = self.retriever.retrieve_batch(
                queries, self.config.top_k
            )

            # Generate answers in batch if generator is available
            generated_answers = [""] * len(queries)
            if self.generator:
                generated_answers = self.generator.generate_batch(
                    queries,
                    retrieved_passages_batch,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )

            # Process batch results
            for i, (sample, retrieved_passages, generated_answer) in enumerate(
                zip(batch_samples, retrieved_passages_batch, generated_answers)
            ):
                query = sample["question"]
                possible_answers = sample.get("possible_answers", [])
                ground_truth = possible_answers[0] if possible_answers else ""

                result = {
                    "question": query,
                    "ground_truth": ground_truth,
                    "possible_answers": possible_answers,
                    "retrieved_passages": retrieved_passages,
                    "generated_answer": generated_answer,
                }
                results.append(result)

                # Log individual result
                sample_idx = batch_start + i + 1
                logger.info(
                    f"Sample {sample_idx} - Question: {query[:50]}{'...' if len(query) > 50 else ''}"
                )
                logger.info(
                    f"Sample {sample_idx} - Generated: {generated_answer[:100]}{'...' if len(generated_answer) > 100 else ''}"
                )

            # Save incremental results after each batch
            current_results = {
                "config": self.config.__dict__,
                "results": results,
                "num_samples": len(results),
            }

            # Compute and add incremental metrics
            if results:
                incremental_metrics = evaluate_results(current_results)
                current_results["metrics"] = incremental_metrics
                logger.info(
                    f"Current F1: {incremental_metrics['avg_f1_score']:.4f}, EM: {incremental_metrics['avg_exact_match']:.4f}"
                )

            with open(output_path_obj, "w") as f:
                json.dump(current_results, f, indent=2)

            logger.info(f"Saved incremental results: {len(results)} samples completed")

        return {
            "config": self.config.__dict__,
            "results": results,
            "num_samples": len(results),
        }


def normalize_answer(s):
    """Lower text, remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """Tokenize text."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    """Compute F1 score between prediction and ground truth."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        logger.warning(
            f"No gold or predicted tokens found, defaulting to 0. Gold: '{a_gold}' -> {gold_toks}, Pred: '{a_pred}' -> {pred_toks}"
        )
        return 0.0

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact_match(a_gold, a_pred):
    """Compute exact match between prediction and ground truth."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_max_f1(possible_answers, a_pred):
    """Compute maximum F1 score against all possible answers."""
    if not possible_answers:
        logger.warning("No possible answers found, defaulting to 0")
        return 0.0
    return max([compute_f1(answer, a_pred) for answer in possible_answers])


def compute_max_exact_match(possible_answers, a_pred):
    """Compute maximum exact match against all possible answers."""
    if not possible_answers:
        return 0.0
    return max([compute_exact_match(answer, a_pred) for answer in possible_answers])


def evaluate_results(results: Dict[str, Any]) -> Dict[str, float]:
    """Comprehensive evaluation metrics including F1 score"""
    samples = results["results"]

    # Retrieval metrics
    avg_retrieval_score = np.mean(
        [
            np.mean([p["score"] for p in sample["retrieved_passages"]])
            for sample in samples
        ]
    )

    # Generation metrics (basic)
    avg_answer_length = np.mean(
        [
            len(sample["generated_answer"].split())
            for sample in samples
            if sample["generated_answer"]
        ]
    )

    # F1 and Exact Match evaluation
    f1_scores = []
    em_scores = []

    for sample in samples:
        if sample["generated_answer"] and sample.get("possible_answers"):
            # Handle possible_answers as either list or string
            possible_answers = sample["possible_answers"]
            if isinstance(possible_answers, str):
                try:
                    import ast

                    possible_answers = ast.literal_eval(possible_answers)
                except (ValueError, SyntaxError):
                    # If it's not a valid Python literal, treat as single answer
                    possible_answers = [possible_answers]

            logger.info(
                f"Generated answer: {sample['generated_answer']}, possible answers: {possible_answers}"
            )
            # Use all possible answers for evaluation (PopQA style)
            f1 = compute_max_f1(possible_answers, sample["generated_answer"])
            em = compute_max_exact_match(possible_answers, sample["generated_answer"])
            f1_scores.append(f1)
            em_scores.append(em)
        else:
            logger.error("No possible answers found for evaluation.")
            raise ValueError("No possible answers found for evaluation.")

    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
    avg_em = np.mean(em_scores) if em_scores else 0.0

    return {
        "avg_retrieval_score": float(avg_retrieval_score),
        "avg_answer_length": float(avg_answer_length),
        "avg_f1_score": float(avg_f1),
        "avg_exact_match": float(avg_em),
        "num_samples": len(samples),
        "num_evaluated_samples": len(f1_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark on PopQA")
    parser.add_argument(
        "--generator_model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Path to generation model",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing samples",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="rag_benchmark_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Path to pre-built FAISS index file",
    )
    parser.add_argument(
        "--passages_path",
        type=str,
        required=True,
        help="Path to passages file corresponding to JSONL index",
    )

    args = parser.parse_args()

    # Create config
    config = RAGConfig(
        generator_model_path=args.generator_model_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        index_path=args.index_path,
        passages_path=args.passages_path,
        temperature=0.0,
    )

    # Run benchmark
    pipeline = RAGPipeline(config)
    results = pipeline.run_benchmark(args.output_path)

    # Evaluate
    metrics = evaluate_results(results)
    results["metrics"] = metrics

    # Save results
    output_path = Path(args.output_path)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Benchmark completed. Results saved to {output_path}")
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
