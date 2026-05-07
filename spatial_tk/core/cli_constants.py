"""CLI-facing constants with no heavy optional dependencies.

Keeps argparse ``choices`` / help strings accurate without importing scanpy,
decoupler, or squidpy when building the parser.
"""

PRESET_RESOURCE_NAMES = ("panglao", "hallmark", "collectri", "dorothea", "progeny")

ASSIGNMENT_STRATEGY_CHOICES = ("top_positive", "threshold", "top_n_vote")
