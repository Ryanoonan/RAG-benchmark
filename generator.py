#!/usr/bin/env python3
import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import build_instruct_rag_prompt, build_baseline_instruct_prompt, build_rag_prompt, build_baseline_prompt

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
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate answer using retrieved passages or baseline"""
        passages_batch = [retrieved_passages] if retrieved_passages else None
        return self.generate_batch([query], passages_batch, max_new_tokens)[0]

    def _build_prompt(
        self, query: str, retrieved_passages: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """Build prompt messages for a single query"""
        if retrieved_passages:
            return build_instruct_rag_prompt(query, retrieved_passages)
        else:
            return build_baseline_instruct_prompt(query)

    def generate_batch(
        self,
        queries: List[str],
        retrieved_passages_batch: Optional[List[List[Dict[str, Any]]]] = None,
        max_new_tokens: int = 100,
    ) -> List[str]:
        """Generate answers for a batch of queries using their retrieved passages or baseline"""
        # Build prompts for all queries
        if retrieved_passages_batch is None:
            all_messages = [self._build_prompt(query) for query in queries]
        else:
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


class GPT2MediumGenerator:
    """openai-community/gpt2-medium based generator"""

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

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        query: str,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate answer using retrieved passages or baseline"""
        passages_batch = [retrieved_passages] if retrieved_passages else None
        return self.generate_batch([query], passages_batch, max_new_tokens)[0]

    def _build_prompt(
        self,
        query: str,
        retrieved_passages: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Build prompt for a single query"""
        if retrieved_passages:
            return build_rag_prompt(query, retrieved_passages)
        else:
            return build_baseline_prompt(query)

    def generate_batch(
        self,
        queries: List[str],
        retrieved_passages_batch: Optional[List[List[Dict[str, Any]]]] = None,
        max_new_tokens: int = 100,
    ) -> List[str]:
        """Generate answers for a batch of queries using their retrieved passages or baseline"""
        # Build prompts for all queries
        if retrieved_passages_batch is None:
            prompts = [self._build_prompt(query) for query in queries]
        else:
            prompts = [
                self._build_prompt(query, retrieved_passages)
                for query, retrieved_passages in zip(queries, retrieved_passages_batch)
            ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
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
