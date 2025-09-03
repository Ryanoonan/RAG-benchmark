#!/usr/bin/env python3
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset

from eval import evaluate_results
from generator import Llama38bInstructGenerator
from retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG benchmark"""

    generator_model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
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


class RAGPipeline:
    """Main RAG pipeline orchestrating retrieval and generation"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever = Retriever(config.retriever_model_path)
        self.generator = None  # Will be initialized when model path is provided

    def load_generator(self):
        """Load generator model"""
        if not self.config.generator_model_path:
            logger.warning(
                "Generator model path is empty. Generator will not be loaded."
            )
            return
        self.generator = Llama38bInstructGenerator(
            self.config.generator_model_path
        )  # TODO: Replace appropriately

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
            logger.error(f"Failed to load PopQA dataset: {e}, no questions loaded")
            # Fallback to sample data
            return []

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
