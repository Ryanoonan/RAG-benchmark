import logging
import re
import string
from collections import Counter
from typing import Any, Dict

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_answer(s):
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


def _get_tokens(s):
    """Tokenize text."""
    if not s:
        return []
    return _normalize_answer(s).split()


def _compute_f1(a_gold, a_pred):
    """Compute F1 score between prediction and ground truth."""
    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

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


def _compute_exact_match(a_gold, a_pred):
    """Compute exact match between prediction and ground truth."""
    return int(_normalize_answer(a_gold) == _normalize_answer(a_pred))


def _compute_max_f1(possible_answers, a_pred):
    """Compute maximum F1 score against all possible answers."""
    if not possible_answers:
        logger.warning("No possible answers found, defaulting to 0")
        return 0.0
    return max([_compute_f1(answer, a_pred) for answer in possible_answers])


def _compute_max_exact_match(possible_answers, a_pred):
    """Compute maximum exact match against all possible answers."""
    if not possible_answers:
        return 0.0
    return max([_compute_exact_match(answer, a_pred) for answer in possible_answers])


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
            f1 = _compute_max_f1(possible_answers, sample["generated_answer"])
            em = _compute_max_exact_match(possible_answers, sample["generated_answer"])
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
