#!/usr/bin/env python3
"""
Phase 2 — LLM-based paraphrasing (Plan Section 2.4, optional).
Generates paraphrases of question templates for lexical diversity.

Per Plan 2.4: "Optionally run generated questions through an LLM paraphraser
(one call per template, batched) to avoid every question in the dataset
looking machine-templated; keep the original template + ground truth linked
in metadata so correctness is never at risk from paraphrasing."
"""
import os
import json
import random
from collections import defaultdict

# Check for LLM availability
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Default paraphrase prompts
PARAPHRASE_SYSTEM_PROMPT = """You are a paraphrasing assistant. Given a question template,
generate 2-3 different paraphrases that preserve the meaning and can be answered
from the same audio context. Return ONLY the paraphrases, one per line, no numbering."""


class Paraphraser:
    """LLM-based question paraphraser for lexical diversity."""

    def __init__(self, model_name=None):
        self.model_name = model_name or os.environ.get("PARAPHRASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the LLM for paraphrasing."""
        if not TRANSFORMERS_AVAILABLE:
            return

        try:
            print(f"Loading paraphraser: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            print("Paraphraser loaded")
        except Exception as e:
            print(f"Failed to load paraphraser: {e}")
            self.model = None

    def paraphrase(self, template, num_variants=2):
        """Generate paraphrases of a question template.

        Args:
            template: The question template string
            num_variants: Number of paraphrases to generate

        Returns:
            List of paraphrased strings (may include original if LLM unavailable)
        """
        if self.model is None:
            # Return original with simple variations if LLM unavailable
            return self._simple_variants(template, num_variants)

        try:
            prompt = f"{PARAPHRASE_SYSTEM_PROMPT}\n\nTemplate: {template}\n\nParaphrases:"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract paraphrases from response
            lines = response.split("\n")
            paraphrases = [l.strip() for l in lines if l.strip() and not l.strip().startswith("Template")]

            if len(paraphrases) >= num_variants:
                return paraphrases[:num_variants]
            else:
                # Pad with simple variants
                variants = paraphrases + self._simple_variants(template, num_variants - len(paraphrases))
                return variants[:num_variants]

        except Exception as e:
            print(f"Paraphrase error: {e}")
            return self._simple_variants(template, num_variants)

    def _simple_variants(self, template, num_variants):
        """Generate simple rule-based variants without LLM."""
        variants = [template]

        # Simple pattern-based variations
        replacements = [
            ("How many", "What's the count of"),
            ("does a ", "does the "),
            ("What sound", "Which sound"),
            ("is present", "can be heard"),
            ("right after", "immediately following"),
            ("before or after", "prior to or subsequent to"),
        ]

        for old, new in replacements[:num_variants - 1]:
            if old in template:
                variants.append(template.replace(old, new))
                break

        return variants[:num_variants]


def paraphrase_qa_dataset(input_path, output_path, num_variants=2):
    """Paraphrase questions in a QA dataset.

    Args:
        input_path: Path to input .jsonl file
        output_path: Path to output .jsonl file
        num_variants: Number of paraphrase variants per question
    """
    paraps = Paraphraser()

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            qa = json.loads(line)

            # Generate paraphrases
            original = qa.get("question", "")
            variants = paraps.paraphrase(original, num_variants)

            # Write original first
            qa["question"] = original
            qa["paraphrase_of"] = None
            f_out.write(json.dumps(qa) + "\n")

            # Write variants
            for variant in variants[1:]:  # Skip first (already wrote original)
                qa_var = qa.copy()
                qa_var["question"] = variant
                qa_var["paraphrase_of"] = qa.get("id")
                f_out.write(json.dumps(qa_var) + "\n")

    print(f"Paraphrasing complete: {input_path} -> {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python paraphrase.py <input.jsonl> <output.jsonl> [num_variants]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    num_variants = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    paraphrase_qa_dataset(input_file, output_file, num_variants)