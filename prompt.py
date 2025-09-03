#!/usr/bin/env python3
from typing import Any, Dict, List


def build_instruct_rag_prompt(query: str, retrieved_passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Build RAG prompt messages for instruct models with retrieved passages"""
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


def build_baseline_instruct_prompt(query: str) -> List[Dict[str, str]]:
    """Build baseline prompt messages for instruct models without retrieved passages"""
    system_prompt = (
        "Answer the question directly based on your knowledge."
        "Only give me the answer and do not output any other words."
    )
    user_prompt = f"Question: {query}"

    return [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{user_prompt}"},
    ]


def build_rag_prompt(query: str, retrieved_passages: List[Dict[str, Any]]) -> str:
    """Build RAG prompt for base models with retrieved passages"""
    context = "\n".join(
        [f"Doc {i+1}: {p['text']}" for i, p in enumerate(retrieved_passages)]
    )
    return (
        f"Given the following documents, answer the question. "
        f"Only provide the answer without additional explanation.\n\n"
        f"Documents:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )


def build_baseline_prompt(query: str) -> str:
    """Build baseline prompt for base models without retrieved passages"""
    return (
        f"Answer the following question directly. "
        f"Only provide the answer without additional explanation.\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )