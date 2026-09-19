"""Phase 2 — Question templates by type (Plan Section 2.4, 2.5)."""
TEMPLATES = {
    "perceptual": [
        "What sound is present in this audio?",
        "What kind of environment does this audio suggest?",
    ],
    "counting": [
        "How many times does a {label} occur?",
        "How many {label}s are heard?",
    ],
    "temporal": [
        "What happens right after the {label}?",
        "Does the {label_a} start before or after the {label_b}?",
        "What event happens before the {label}?",
    ],
    "causal": [
        "Why might the {label_a} occur with the {label_b}?",
        "Why does this sound like a {scenario} scene?",
        "Why are {label_a} and {label_b} present together?",
    ],
    "negation": [
        "Is there a {absent_label} in this audio?",
    ],
    "comparative": [
        "Which happened more often, {label_a} or {label_b}?",
    ],
}
