#!/usr/bin/env python3
import argparse
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from eval import evaluate_results
from generator import Llama38bInstructGenerator, get_generator_from_path
from retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Mode(StrEnum):
    RAG = "rag"
    BASELINE = "baseline"


@dataclass
class Config:
    """Configuration for benchmark"""

    mode: Mode
    generator_model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    # Model paths
    retriever_model_path: str = "intfloat/e5-base-v2"
    # Data paths
    popqa_dataset_path: str = "akariasai/PopQA"
    index_path: Optional[str] = None  # Path to pre-built FAISS index
    passages_path: Optional[str] = (
        None  # Path to passages file corresponding to the index
    )

    # Retrieval settings
    top_k: int = 5
    max_passage_length: int = 512

    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 0.7

    # Evaluation settings
    batch_size: int = 8
    num_samples: int = 100  # Use -1 for full dataset


class BaselineRunner:
    """Runner for baseline question answering without retrieval"""

    def __init__(self, generator):
        self.generator = generator

    def run_batch(self, queries: List[str], max_new_tokens: int = 100) -> List[str]:
        """Run batch inference without retrieval"""
        return self.generator.generate_batch(queries, None, max_new_tokens)


class RAGRunner:
    """Runner for RAG-based question answering with retrieval"""

    def __init__(self, generator, retriever: Retriever):
        self.generator = generator
        self.retriever = retriever

    def run_batch(
        self, queries: List[str], top_k: int = 5, max_new_tokens: int = 100
    ) -> List[str]:
        """Run batch inference with retrieval"""
        retrieved_passages_batch = self.retriever.retrieve_batch(queries, top_k)
        return self.generator.generate_batch(
            queries, retrieved_passages_batch, max_new_tokens
        )


class QAEngine:
    """Question Answering engine that delegates to a runner"""

    def __init__(self, runner):
        self.runner = runner

    def answer_batch(self, queries: List[str], **kwargs) -> List[str]:
        """Answer a batch of queries using the configured runner"""
        return self.runner.run_batch(queries, **kwargs)


class RAGPipeline:
    """Main pipeline orchestrating retrieval and generation"""

    def __init__(self, config: Config):
        self.config = config

        # Create appropriate runner and QA engine based on mode
        if config.mode == Mode.RAG:
            if not config.index_path or not config.passages_path:
                raise ValueError(
                    "Both index_path and passages_path must be provided for RAG mode"
                )
            if not config.generator_model_path:
                raise ValueError("Generator model path must be provided for RAG mode")
            generator = get_generator_from_path(config.generator_model_path)
            retriever = Retriever(config.retriever_model_path)
            retriever.load_index(config.index_path, config.passages_path)
            runner = RAGRunner(generator, retriever)
        elif config.mode == Mode.BASELINE:
            if not config.generator_model_path:
                raise ValueError(
                    "Generator model path must be provided for baseline mode"
                )
            generator = get_generator_from_path(config.generator_model_path)
            runner = BaselineRunner(generator)
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")

        self.qa_engine = QAEngine(runner)

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

            # Generate answers using QA engine
            generated_answers = self.qa_engine.answer_batch(
                queries,
                top_k=self.config.top_k,
                max_new_tokens=self.config.max_new_tokens,
            )

            # Process batch results
            for i, (sample, generated_answer) in enumerate(
                zip(batch_samples, generated_answers)
            ):
                query = sample["question"]
                possible_answers = sample.get("possible_answers", [])

                result = {
                    "question": query,
                    "possible_answers": possible_answers,
                    "generated_answer": generated_answer,
                    "method": self.config.mode.value,
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
                logger.info(f"Current F1: {incremental_metrics['avg_f1_score']:.4f}")

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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "baseline"],
        default="rag",
        help="Mode to run: 'rag' for retrieval-augmented generation or 'baseline' for direct generation",
    )

    args = parser.parse_args()
    logger.info(f"Creating config with args {args}")
    # Create config
    config = Config(
        mode=Mode(args.mode),
        generator_model_path=args.generator_model_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        index_path=args.index_path if args.mode == "rag" else None,
        passages_path=args.passages_path if args.mode == "rag" else None,
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
