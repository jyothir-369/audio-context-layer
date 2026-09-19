#!/usr/bin/env python3
"""Phase 4 — LLM-grounded QA answerer (Plan Section 3.3, sub-approach 2).
Uses a small instruction-tuned LLM to answer questions given structured context.
Requires transformers library. Falls back gracefully if model unavailable."""
import os
import warnings

# Check for transformers availability
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available — LLM-grounded answerer will not function")

try:
    from qa_answerer import build_context
except ImportError:
    from .qa_answerer import build_context

# Default model: small instruction-tuned model per Plan 3.3 recommendation
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Small enough for CPU, per Plan scope
# Alternative: "microsoft/Phi-3.5-mini-instruct" if more compute available


class LLMGroundedAnswerer:
    """LLM-powered QA answerer using structured context.

    Per Plan 3.3: feed {structured context + question} to a small instruction-tuned LLM
    with a system prompt instructing it to answer only from the given context and
    say "not enough information" if the context doesn't support an answer.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or os.environ.get("LLM_MODEL", DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self._load_model()

    def _load_model(self):
        """Load the LLM and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            self.model = None
            self.tokenizer = None
            return

        try:
            print(f"Loading LLM: {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else "cpu",
                trust_remote_code=True
            )
            # Use pipeline for easier generation
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                do_sample=False,  # Deterministic for reproducibility
                temperature=None,
                top_p=None,
            )
            print(f"LLM loaded successfully")
        except Exception as e:
            warnings.warn(f"Failed to load LLM {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
            self.pipe = None

    def answer(self, question, timeline_result, scene_type=None):
        """Answer a question using the LLM given structured context.

        Args:
            question: The question string
            timeline_result: Output from tag_audio() or similar
            scene_type: Optional scene type (street/kitchen/park/office/construction_site)

        Returns:
            Tuple of (answer_string, answer_type)
        """
        if self.pipe is None:
            return ("LLM not available — install transformers and ensure model can be downloaded", "llm_unavailable")

        # Build structured context from timeline
        context_text = build_context(timeline_result)

        # Add scene type if provided
        if scene_type:
            context_text += f"\nScene type: {scene_type}"

        # Build prompt per Plan 3.3 instructions
        system_prompt = """You are a helpful assistant that answers questions about audio content.
You must answer ONLY based on the provided context. If the context does not contain
enough information to answer the question, say "Not enough information" or
"I cannot determine this from the provided audio context."
Do not make up or hallucinate information."""

        user_prompt = f"""Context (detected audio events):
{context_text}

Question: {question}

Answer based ONLY on the context above:"""

        try:
            # Generate response
            outputs = self.pipe(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_new_tokens=128,
                do_sample=False,
            )

            # Extract assistant response
            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"]
                # Get the last message (assistant's reply)
                for msg in reversed(response):
                    if msg.get("role") == "assistant":
                        answer_text = msg.get("content", "").strip()
                        break
                else:
                    answer_text = response[-1].get("content", "").strip() if response else ""

                # Clean up common artifacts
                answer_text = answer_text.strip()
                if answer_text.startswith("Answer:"):
                    answer_text = answer_text[7:].strip()

                return (answer_text, "llm_grounded")
            else:
                return ("LLM generated empty response", "llm_error")

        except Exception as e:
            return (f"LLM error: {str(e)[:50]}", "llm_error")


def answer(question, timeline_result, scene_type=None):
    """Convenience function for single-question answering."""
    answerer = LLMGroundedAnswerer()
    return answerer.answer(question, timeline_result, scene_type)


if __name__ == "__main__":
    # Quick test with a sample timeline
    sample_timeline = {
        "timeline": [
            {"label": "dog_bark", "start": 2.1, "end": 3.4, "confidence": 0.8},
            {"label": "doorbell", "start": 5.0, "end": 5.6, "confidence": 0.75},
            {"label": "footstep", "start": 7.0, "end": 9.0, "confidence": 0.7},
        ]
    }

    test_questions = [
        "What sounds are present in this audio?",
        "How many times does a dog bark?",
        "Why might the doorbell have rung?",
    ]

    print("Testing LLM-grounded answerer...")
    answerer = LLMGroundedAnswerer()

    for q in test_questions:
        ans, kind = answerer.answer(q, sample_timeline)
        print(f"\nQ: {q}")
        print(f"A: {ans} [{kind}]")