"""Human-readable summary of a train.py checkpoint's settings and results.

Stdlib only -- no torch, no numpy, no matplotlib -- so this stays cheap to
import from a CLI that just wants to write a text file. `count_parameters`
duck-types `.numel()` rather than importing torch for a type hint.
"""

from __future__ import annotations

from typing import Any, Mapping

_RUN_SUMMARY_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("network", ("hidden_layers", "hidden_width")),
    ("domain", ("x_max", "y_max", "t_max", "epsilon")),
    ("sampling", ("n_collocation", "n_ic", "n_bc", "pde_at_t0")),
    (
        "optimizer",
        (
            "epochs",
            "lr",
            "adam_tol",
            "plateau_patience",
            "plateau_factor",
            "min_lr",
            "adam_patience",
            "lbfgs_steps",
            "lbfgs_tol",
            "lbfgs_patience",
        ),
    ),
    ("loss weights", ("pde_weight", "ic_weight", "bc_weight", "energy_penalty", "energy_weight")),
]

# Bookkeeping/control flags that don't describe the model or the loss --
# left out of the "other" catch-all in format_run_summary rather than the
# groups above, since printing them wouldn't help someone reading the
# summary later to understand what was trained.
_RUN_SUMMARY_EXCLUDED = {"run_name", "device", "smoke_test", "checkpoint_every", "log_every"}

# PINN.__init__ registers lower_bound/upper_bound as buffers (see
# src/pinn/model.py), so they appear in state_dict() but are not trainable
# parameters -- excluded from count_parameters for that reason.
_STATE_DICT_BUFFERS = frozenset({"lower_bound", "upper_bound"})

# Order to print "final <term>" lines in, when present in history.
_LOSS_TERM_ORDER = ("pde", "ic", "bc", "energy", "total")


def count_parameters(model_state: Mapping[str, Any]) -> int:
    """Trainable parameter count from a checkpoint's "model_state", without
    instantiating the model.

    Args:
        model_state: the "model_state" entry from a train.py checkpoint
            (torch.load(checkpoint_path)["model_state"]), i.e. a state_dict
            of tensors (or anything exposing .numel()).

    Returns:
        Sum of numel() over all entries except the registered buffers.
    """
    return sum(int(v.numel()) for k, v in model_state.items() if k not in _STATE_DICT_BUFFERS)


def _format_training_block(args: dict, history: dict[str, list[float]]) -> list[str]:
    total_steps = len(history["total"]) if "total" in history else 0
    expected_adam = 5 if args.get("smoke_test") else args.get("epochs", 0)
    # Clamp: a checkpoint saved mid-Adam (via --checkpoint-every) has fewer
    # than expected_adam steps recorded, and total_steps must never be
    # exceeded here or lbfgs_evals would go negative.
    adam_steps = min(total_steps, expected_adam)
    lbfgs_evals = total_steps - adam_steps

    lines = [
        "training:",
        f"  recorded steps: {total_steps}",
        f"  adam steps: {adam_steps}",
        # LBFGS(max_iter=20).step(closure) invokes the closure ~20x per call
        # (line-search + gradient evals), and train.py's closure records
        # every invocation -- so this is NOT the same as --lbfgs-steps.
        # It can also fall short of 20x --lbfgs-steps: train.py stops early
        # once the total loss stalls (--lbfgs-patience).
        f"  lbfgs closure evals: {lbfgs_evals}",
    ]
    for name in _LOSS_TERM_ORDER:
        values = history.get(name)
        if values:
            lines.append(f"  final {name}: {values[-1]:.6e}")
    lines.append("")
    return lines


def format_run_summary(
    args: dict,
    *,
    model_state: Mapping[str, Any] | None = None,
    history: dict[str, list[float]] | None = None,
    source: str | None = None,
) -> str:
    """Format a checkpoint's settings and results into a readable run summary.

    Args:
        args: the "args" entry from a train.py checkpoint
            (torch.load(checkpoint_path)["args"]), i.e. vars(argparse.Namespace)
            from train.py's CLI flags.
        model_state: optional "model_state" entry from the same checkpoint --
            when given, adds a trainable-parameter count (see
            count_parameters).
        history: optional "history" entry from the same checkpoint -- when
            given, adds a training block with step counts and each loss
            term's final value.
        source: optional path (or other identifier) of the checkpoint this
            summary was built from, printed as the first line.

    Returns:
        Multi-line string grouping network size, domain/epsilon, sampling
        counts, optimizer schedule, and per-term loss weights under headers
        (see _RUN_SUMMARY_GROUPS), plus optional parameter-count and
        training blocks -- so a run's exact settings and outcome travel
        together as one plain-text file. Any args key not covered by a
        group (e.g. a CLI flag added to train.py after these groups were
        written) is listed under "other" rather than silently dropped.

        With model_state=None, history=None, source=None, the output is
        identical to a bare grouped dump of `args` (no header block).
    """
    lines = []

    if source is not None or model_state is not None:
        lines.append("run summary")
        if source is not None:
            lines.append(f"  source: {source}")
        if model_state is not None:
            lines.append(f"  parameters: {count_parameters(model_state)}")
        lines.append("")

    grouped_keys = {key for _, keys in _RUN_SUMMARY_GROUPS for key in keys}
    for title, keys in _RUN_SUMMARY_GROUPS:
        present = [key for key in keys if key in args]
        if not present:
            continue
        lines.append(title + ":")
        lines.extend(f"  {key}: {args[key]}" for key in present)
        lines.append("")

    if history is not None:
        lines.extend(_format_training_block(args, history))

    leftover = [key for key in args if key not in grouped_keys and key not in _RUN_SUMMARY_EXCLUDED]
    if leftover:
        lines.append("other:")
        lines.extend(f"  {key}: {args[key]}" for key in leftover)
        lines.append("")

    return "\n".join(lines).rstrip()
