#!/usr/bin/env python3
import logging
from abc import ABC
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama38bInstructGenerator:
    """meta-llama/Meta-Llama-3-8B-Instruct based generator"""

    def __init__(self, model_path: str, device: str = "auto"):
        if not model_path:
            raise ValueError("Generator model path cannot be empty")

        logger.info(f"Loading {model_path} generator")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
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