"""Repeat the prompt-boundary cases after the mentions-deduplication rule."""
import run_boundary_review

review = run_boundary_review.review
review.SCENARIOS = [
    (name, utterances + ['그거 줘'] if name == '11_unsupported_mixed' else utterances)
    for name, utterances in review.SCENARIOS
]
review.main()
