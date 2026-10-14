"""Visualize Full-INT8 Top-p and MSB-slice Top-p on attention-like Q/K.

This is a correctness and visualization example, not a performance benchmark.
The integer dot-product helper intentionally materializes products so that the
same implementation runs on CPU without relying on an optimized INT8 GEMM.

Two input distributions are available:
    focused: a few K tokens are strongly correlated with Q;
    diffuse: most QK logits are close, so probability mass is widely spread.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TopPResult:
    """Softmax probabilities and the smallest sorted prefix reaching Top-p."""

    probabilities: torch.Tensor
    sorted_probabilities: torch.Tensor
    sorted_indices: torch.Tensor
    selected_sorted: torch.Tensor
    selected_mask: torch.Tensor


def symmetric_int8_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each token/head independently along ``head_dim``."""

    max_abs = x.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 127.0).clamp_min(torch.finfo(torch.float32).eps)
    quantized = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def signed_msb_slice(x_int8: torch.Tensor, msb_bits: int) -> torch.Tensor:
    """Extract signed MSBs with an arithmetic right shift."""

    if not 1 <= msb_bits <= 8:
        raise ValueError(f"msb_bits must be in [1, 8], got {msb_bits}")
    shift = 8 - msb_bits
    return torch.bitwise_right_shift(x_int8.to(torch.int16), shift).to(torch.int8)


def integer_dot_reference(q_codes: torch.Tensor, k_codes: torch.Tensor) -> torch.Tensor:
    """Compute exact integer-code QK scores for the small visualization case.

    Shapes:
        q_codes: [batch, heads, query_len, head_dim]
        k_codes: [batch, heads, kv_len, head_dim]
        return:  [batch, heads, query_len, kv_len]
    """

    products = q_codes.to(torch.int32).unsqueeze(-2) * k_codes.to(torch.int32).unsqueeze(-3)
    return products.sum(dim=-1)


def quantized_attention_logits(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """Recover scaled attention logits from Full-INT8 integer dot products."""

    head_dim = q_int8.shape[-1]
    integer_dot = integer_dot_reference(q_int8, k_int8).to(torch.float32)
    return integer_dot * q_scale * k_scale.transpose(-1, -2) / math.sqrt(head_dim)


def msb_attention_logits(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    msb_bits: int,
) -> torch.Tensor:
    """Approximate attention logits with signed MSB x MSB products."""

    head_dim = q_int8.shape[-1]
    shift = 8 - msb_bits
    q_msb = signed_msb_slice(q_int8, msb_bits)
    k_msb = signed_msb_slice(k_int8, msb_bits)
    integer_dot = integer_dot_reference(q_msb, k_msb).to(torch.float32)

    # Both operands lost ``shift`` low bits, so restore their approximate scale.
    reconstruction = float(2 ** (2 * shift))
    return (
        integer_dot
        * reconstruction
        * q_scale
        * k_scale.transpose(-1, -2)
        / math.sqrt(head_dim)
    )


def top_p_from_logits(logits: torch.Tensor, threshold: float) -> TopPResult:
    """Apply softmax and retain the smallest probability prefix reaching p."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"top-p threshold must be in (0, 1], got {threshold}")

    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)

    # Keep the token that crosses p: cumulative mass before that token is < p.
    selected_sorted = cumulative - sorted_probabilities < threshold
    selected_sorted[..., 0] = True

    selected_mask = torch.zeros_like(selected_sorted, dtype=torch.bool)
    selected_mask.scatter_(-1, sorted_indices, selected_sorted)
    return TopPResult(
        probabilities=probabilities,
        sorted_probabilities=sorted_probabilities,
        sorted_indices=sorted_indices,
        selected_sorted=selected_sorted,
        selected_mask=selected_mask,
    )


def make_focused_attention_case(
    num_heads: int,
    seq_len: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create focused attention dominated by a few relevant KV tokens."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn((1, num_heads, 1, head_dim), generator=generator)
    keys = 0.35 * torch.randn(
        (1, num_heads, seq_len, head_dim), generator=generator
    )

    # Imitate an attention sink and two semantic matches. Their QK logits are
    # much larger than the background, concentrating most probability mass.
    base_indices = [0, seq_len // 3, 2 * seq_len // 3]
    correlations = [0.60, 0.48, 0.38]
    for head in range(num_heads):
        for base_index, correlation in zip(base_indices, correlations):
            token_index = (base_index + 7 * head) % seq_len
            noise = 0.35 * torch.randn((head_dim,), generator=generator)
            keys[0, head, token_index] = (
                correlation * query[0, head, 0]
                + math.sqrt(1.0 - correlation**2) * noise
            )

    return query, keys


def make_diffuse_attention_case(
    num_heads: int,
    seq_len: int,
    head_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create diffuse attention whose probability mass spans many KV tokens."""

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # A smaller Q magnitude produces low-variance QK/sqrt(D) logits. Softmax
    # therefore stays close to uniform instead of being dominated by a few K.
    query = 0.35 * torch.randn(
        (1, num_heads, 1, head_dim), generator=generator
    )
    keys = torch.randn((1, num_heads, seq_len, head_dim), generator=generator)
    return query, keys


def select_row(result: TopPResult, head: int) -> TopPResult:
    """Take batch 0, query 0 and one head for plotting."""

    return TopPResult(
        probabilities=result.probabilities[0, head, 0],
        sorted_probabilities=result.sorted_probabilities[0, head, 0],
        sorted_indices=result.sorted_indices[0, head, 0],
        selected_sorted=result.selected_sorted[0, head, 0],
        selected_mask=result.selected_mask[0, head, 0],
    )


def print_comparison(ref: TopPResult, out: TopPResult, threshold: float) -> None:
    """Print Top-p sets and their agreement for the plotted attention head."""

    ref_indices = ref.sorted_indices[ref.selected_sorted]
    out_indices = out.sorted_indices[out.selected_sorted]
    intersection = ref.selected_mask & out.selected_mask

    ref_count = int(ref.selected_mask.sum().item())
    out_count = int(out.selected_mask.sum().item())
    intersection_count = int(intersection.sum().item())
    union_count = int((ref.selected_mask | out.selected_mask).sum().item())

    recall = intersection_count / ref_count
    precision = intersection_count / out_count
    jaccard = intersection_count / union_count
    ref_mass_kept_by_out = float(ref.probabilities[out.selected_mask].sum().item())

    print(f"Top-p threshold       : {threshold:.4f}")
    print(f"REF selected count    : {ref_count}")
    print(f"OUT selected count    : {out_count}")
    print(f"OUT recall vs REF     : {recall:.4f}")
    print(f"OUT precision vs REF  : {precision:.4f}")
    print(f"Jaccard(REF, OUT)     : {jaccard:.4f}")
    print(f"REF mass kept by OUT  : {ref_mass_kept_by_out:.4f}")
    print(f"REF selected indices  : {ref_indices.tolist()}")
    print(f"OUT selected indices  : {out_indices.tolist()}")


def plot_probability_bar_charts(
    ref: TopPResult,
    out: TopPResult,
    case_name: str,
    threshold: float,
    msb_bits: int,
    output_path: Path,
    show: bool,
) -> None:
    """Draw REF and OUT bar charts aligned to REF's probability order."""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error

    figure, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    panels = (
        (ref, "REF: Full INT8 x INT8 + Softmax", "#c92513", "#efb2aa"),
        (
            out,
            f"OUT: MSB({msb_bits}-bit) x MSB({msb_bits}-bit) + Softmax",
            "#365fa9",
            "#aebfe0",
        ),
    )
    aligned_indices = ref.sorted_indices
    original_indices = aligned_indices.detach().cpu().tolist()

    for axis, (result, title, selected_color, unselected_color) in zip(axes, panels):
        # Both panels use the original token IDs sorted by REF probability.
        probabilities = result.probabilities[aligned_indices].detach().cpu().tolist()
        selected = result.selected_mask[aligned_indices].detach().cpu().tolist()
        ranks = list(range(len(probabilities)))
        colors = [selected_color if keep else unselected_color for keep in selected]

        axis.bar(ranks, probabilities, width=0.9, color=colors, edgecolor="none")
        selected_count = sum(selected)
        selected_mass = sum(
            probability
            for probability, keep in zip(probabilities, selected)
            if keep
        )
        axis.axvline(
            selected_count - 0.5,
            color=selected_color,
            linestyle="--",
            linewidth=1.6,
        )
        axis.set_title(title, fontsize=13, fontweight="bold")
        axis.set_xlabel("Original KV token ID (ordered by REF probability)")
        axis.grid(axis="y", color="#d6d6d6", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.text(
            0.98,
            0.96,
            f"Top-p={threshold:g}\nselected={selected_count}\nmass={selected_mass:.3f}",
            transform=axis.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            color=selected_color,
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": selected_color, "alpha": 0.9},
        )

        # The bars are probability ranks; tick labels show original KV-cache indices.
        tick_count = min(12, len(ranks))
        tick_positions = sorted({round(i * (len(ranks) - 1) / max(1, tick_count - 1)) for i in range(tick_count)})
        axis.set_xticks(tick_positions)
        axis.set_xticklabels([str(original_indices[i]) for i in tick_positions], rotation=45)

    axes[0].set_ylabel("Probability p")
    axes[1].text(
        0.98,
        0.78,
        "X-axis aligned with REF\n"
        "Same original KV token IDs\n"
        "Blue dashed line = MSB Top-p budget",
        transform=axes[1].transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        color="#365fa9",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#365fa9", "alpha": 0.9},
    )
    figure.suptitle(
        f"{case_name.title()} Attention: Full INT8 Reference vs MSB Approximation",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Bar chart saved to    : {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(figure)


def plot_aligned_id_comparison(
    ref: TopPResult,
    out: TopPResult,
    case_name: str,
    msb_bits: int,
    output_path: Path,
    show: bool,
) -> None:
    """Compare REF and OUT probabilities for the same original KV token ID."""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error

    # Use REF's descending order for both series. OUT is gathered with the same
    # original token indices, so every neighboring pair belongs to one KV ID.
    aligned_indices = ref.sorted_indices
    ref_probabilities = ref.probabilities[aligned_indices].detach().cpu().tolist()
    out_probabilities = out.probabilities[aligned_indices].detach().cpu().tolist()
    original_indices = aligned_indices.detach().cpu().tolist()

    ranks = list(range(len(original_indices)))
    width = 0.42
    ref_positions = [rank - width / 2 for rank in ranks]
    out_positions = [rank + width / 2 for rank in ranks]

    figure, axis = plt.subplots(figsize=(16, 5.5))
    axis.bar(
        ref_positions,
        ref_probabilities,
        width=width,
        color="#c92513",
        label="REF: Full INT8 x INT8",
    )
    axis.bar(
        out_positions,
        out_probabilities,
        width=width,
        color="#365fa9",
        label=f"OUT: MSB({msb_bits}-bit) x MSB({msb_bits}-bit)",
    )
    ref_count = int(ref.selected_mask.sum().item())
    out_count = int(out.selected_mask.sum().item())
    axis.axvline(
        ref_count - 0.5,
        color="#c92513",
        linestyle="--",
        linewidth=1.6,
        label=f"REF Top-p budget = {ref_count}",
    )
    axis.axvline(
        out_count - 0.5,
        color="#365fa9",
        linestyle="--",
        linewidth=1.6,
        label=f"MSB Top-p budget = {out_count}",
    )

    tick_count = min(16, len(ranks))
    tick_positions = sorted(
        {
            round(i * (len(ranks) - 1) / max(1, tick_count - 1))
            for i in range(tick_count)
        }
    )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [str(original_indices[i]) for i in tick_positions], rotation=45
    )
    axis.set_xlabel("Original KV token ID (ordered by REF probability)")
    axis.set_ylabel("Probability p")
    axis.set_title(
        f"{case_name.title()} Attention: Same-ID Probability Comparison",
        fontsize=15,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#d6d6d6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Aligned-ID chart saved : {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(figure)


def plot_same_id_probability_ratio(
    ref: TopPResult,
    out: TopPResult,
    case_name: str,
    output_path: Path,
    show: bool,
) -> None:
    """Plot MSB probability / Full-INT8 probability for each original KV ID."""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error

    # Both probability tensors are indexed by the original KV token ID. Use the
    # REF order only for display, then gather both values with identical IDs.
    aligned_indices = ref.sorted_indices
    epsilon = torch.finfo(ref.probabilities.dtype).tiny
    ref_probabilities = ref.probabilities[aligned_indices].clamp_min(epsilon)
    out_probabilities = out.probabilities[aligned_indices].clamp_min(epsilon)
    ratios = out_probabilities / ref_probabilities

    original_indices = aligned_indices.detach().cpu().tolist()
    ratio_values = ratios.detach().cpu().tolist()
    ranks = list(range(len(original_indices)))
    colors = ["#365fa9" if ratio >= 1.0 else "#c92513" for ratio in ratio_values]

    figure, axis = plt.subplots(figsize=(16, 5.5))
    axis.bar(ranks, ratio_values, width=0.9, color=colors, edgecolor="none")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    axis.set_yscale("log", base=2)

    tick_count = min(16, len(ranks))
    tick_positions = sorted(
        {
            round(i * (len(ranks) - 1) / max(1, tick_count - 1))
            for i in range(tick_count)
        }
    )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [str(original_indices[i]) for i in tick_positions], rotation=45
    )
    axis.set_xlabel("Original KV token ID (ordered by REF probability)")
    axis.set_ylabel("Probability ratio: MSB / Full INT8 (log2 scale)")
    axis.set_title(
        f"{case_name.title()} Attention: Same-ID Probability Ratio",
        fontsize=15,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#d6d6d6", linewidth=0.8)
    axis.set_axisbelow(True)

    # Weight distortion by REF probability so negligible tail tokens do not
    # dominate the summary merely because their denominator is very small.
    weighted_abs_log2_ratio = float(
        (ref_probabilities * torch.abs(torch.log2(ratios))).sum().item()
    )
    axis.text(
        0.99,
        0.97,
        "blue: MSB overestimates\n"
        "red: MSB underestimates\n"
        f"REF-weighted |log2 ratio|={weighted_abs_log2_ratio:.4f}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Same-ID ratio saved    : {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(figure)


def compute_same_id_qk_score_ratio(
    msb_logits: torch.Tensor,
    full_int8_logits: torch.Tensor,
    minimum_denominator: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute reconstructed MSB QK / Full-INT8 QK before Softmax.

    The returned boolean mask excludes non-finite values and Full-INT8 scores
    too close to zero, whose ratios would be undefined or numerically unstable.
    """

    if msb_logits.shape != full_int8_logits.shape:
        raise ValueError(
            "MSB and Full-INT8 logits must have the same shape, got "
            f"{tuple(msb_logits.shape)} and {tuple(full_int8_logits.shape)}"
        )
    if minimum_denominator <= 0.0:
        raise ValueError("minimum_denominator must be positive")

    msb_scores = msb_logits.to(torch.float32)
    full_scores = full_int8_logits.to(torch.float32)
    valid_mask = (
        torch.isfinite(msb_scores)
        & torch.isfinite(full_scores)
        & (full_scores.abs() >= minimum_denominator)
    )
    ratios = torch.full_like(full_scores, torch.nan)
    ratios[valid_mask] = msb_scores[valid_mask] / full_scores[valid_mask]
    return ratios, valid_mask


def plot_same_id_qk_score_ratio(
    ratios: torch.Tensor,
    valid_mask: torch.Tensor,
    full_int8_logits: torch.Tensor,
    msb_logits: torch.Tensor,
    aligned_indices: torch.Tensor,
    case_name: str,
    output_path: Path,
    show: bool,
) -> None:
    """Plot same-ID pre-Softmax QK score ratios in Full-INT8 order."""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error

    aligned_ratios = ratios[aligned_indices]
    aligned_valid = valid_mask[aligned_indices]
    original_indices = aligned_indices.detach().cpu().tolist()
    ratio_values = aligned_ratios.detach().cpu().tolist()
    valid_values = aligned_valid.detach().cpu().tolist()
    ranks = list(range(len(original_indices)))

    # Blue means that MSB and Full-INT8 scores have the same sign; red marks a
    # sign reversal. Invalid near-zero denominators are shown as zero-height gray.
    plotted_values = [ratio if valid else 0.0 for ratio, valid in zip(ratio_values, valid_values)]
    colors = [
        "#b0b0b0" if not valid else ("#365fa9" if ratio >= 0.0 else "#c92513")
        for ratio, valid in zip(ratio_values, valid_values)
    ]

    figure, axis = plt.subplots(figsize=(16, 5.5))
    axis.bar(ranks, plotted_values, width=0.9, color=colors, edgecolor="none")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.4)
    axis.axhline(0.0, color="#777777", linewidth=0.9)
    axis.set_yscale("symlog", linthresh=0.25, base=2)

    tick_count = min(16, len(ranks))
    tick_positions = sorted(
        {
            round(i * (len(ranks) - 1) / max(1, tick_count - 1))
            for i in range(tick_count)
        }
    )
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(
        [str(original_indices[i]) for i in tick_positions], rotation=45
    )
    axis.set_xlabel("Original KV token ID (ordered by Full-INT8 QK score)")
    axis.set_ylabel("QK score ratio: reconstructed MSB / Full INT8")
    axis.set_title(
        f"{case_name.title()} Attention: Same-ID Pre-Softmax QK Score Ratio",
        fontsize=15,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#d6d6d6", linewidth=0.8)
    axis.set_axisbelow(True)

    full_valid = full_int8_logits[valid_mask]
    msb_valid = msb_logits[valid_mask]
    sign_agreement = float(
        (torch.sign(full_valid) == torch.sign(msb_valid)).float().mean().item()
    )
    mean_abs_score_error = float((msb_valid - full_valid).abs().mean().item())
    axis.text(
        0.99,
        0.97,
        "ratio=1: identical QK score\n"
        "blue: same sign; red: sign reversal\n"
        f"sign agreement={sign_agreement:.1%}\n"
        f"mean |MSB-Full|={mean_abs_score_error:.4f}\n"
        f"valid IDs={int(valid_mask.sum().item())}/{valid_mask.numel()}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"QK-score ratio saved   : {output_path.resolve()}")

    if show:
        plt.show()
    plt.close(figure)


def plot_recall_budget_curve(
    ref: TopPResult,
    out: TopPResult,
    case_name: str,
    ref_threshold: float,
    output_path: Path,
    show: bool,
) -> None:
    """Plot Full-INT8 Top-p recall as the MSB candidate budget grows."""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error

    # B0 is every KV token in this standalone case. Bpre(k) is the first k
    # tokens in the MSB probability ranking, for every possible budget k.
    b0 = ref.probabilities.numel()
    msb_order = out.sorted_indices
    ref_selected_in_msb_order = ref.selected_mask[msb_order].to(torch.float32)
    ref_probability_in_msb_order = ref.probabilities[msb_order]

    ref_count = int(ref.selected_mask.sum().item())
    recalls = ref_selected_in_msb_order.cumsum(dim=-1) / ref_count
    retained_mass = ref_probability_in_msb_order.cumsum(dim=-1)
    budget_counts = torch.arange(1, b0 + 1, dtype=torch.float32)
    budget_ratios = budget_counts / b0

    x_values = budget_ratios.detach().cpu().tolist()
    recall_values = recalls.detach().cpu().tolist()
    retained_mass_values = retained_mass.detach().cpu().tolist()

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        x_values,
        recall_values,
        color="#365fa9",
        linewidth=2.2,
        label="Recall of Full-INT8 Top-p set",
    )
    axis.plot(
        x_values,
        retained_mass_values,
        color="#2b8a3e",
        linewidth=2.2,
        label="Retained Full-INT8 probability mass",
    )

    target_lines = []
    for target, color in ((0.95, "#e67700"), (0.99, "#c2255c"), (1.00, "#6f42c1")):
        matching = torch.nonzero(recalls >= target, as_tuple=False)
        minimum_budget = int(matching[0].item()) + 1
        minimum_ratio = minimum_budget / b0
        target_lines.append(
            f"Recall {target:.0%}: Bpre={minimum_budget}/{b0} ({minimum_ratio:.1%})"
        )
        axis.axhline(target, color=color, linestyle=":", linewidth=1.0)
        axis.scatter([minimum_ratio], [target], color=color, s=35, zorder=3)

    axis.text(
        0.98,
        0.05,
        "\n".join(target_lines),
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#777777", "alpha": 0.9},
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("MSB candidate budget Bpre / B0")
    axis.set_ylabel("Recall / retained probability mass")
    axis.set_title(
        f"{case_name.title()} Attention: MSB Recall-Budget Curve\n"
        f"Reference = Full-INT8 Top-p (p={ref_threshold:g})",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(color="#d6d6d6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", bbox_to_anchor=(1.0, 0.25))

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Recall-budget saved    : {output_path.resolve()}")
    for line in target_lines:
        print(f"  {line}")

    if show:
        plt.show()
    plt.close(figure)


def plot_top_p_threshold_token_counts(
    ref_threshold,
    full_int8_logits: torch.Tensor,
    msb_logits: torch.Tensor,
    case_name: str,
    output_path: Path,
    show: bool,
) -> None:
    """Compare MSB token counts over p=0.90..0.99 with Full-INT8 at p=0.90."""

    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error

    if full_int8_logits.ndim != 1 or msb_logits.ndim != 1:
        raise ValueError("full_int8_logits and msb_logits must be one-dimensional rows")
    if full_int8_logits.shape != msb_logits.shape:
        raise ValueError("Full-INT8 and MSB logit rows must have the same shape")

    # ref_threshold = 0.90
    msb_thresholds = [ref_threshold + 0.01 * index for index in range(10)]
    ref_count = int(
        top_p_from_logits(full_int8_logits, ref_threshold).selected_mask.sum().item()
    )
    msb_counts = [
        int(top_p_from_logits(msb_logits, threshold).selected_mask.sum().item())
        for threshold in msb_thresholds
    ]

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        msb_thresholds,
        msb_counts,
        color="#365fa9",
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="MSB prior retained tokens",
    )
    axis.axhline(
        ref_count,
        color="#c92a2a",
        linestyle="--",
        linewidth=2.2,
        label=f"Full-INT8 baseline: Top-p={ref_threshold}, tokens={ref_count}",
    )

    for threshold, count in zip(msb_thresholds, msb_counts):
        axis.annotate(
            str(count),
            (threshold, count),
            xytext=(0, 7),
            textcoords="offset points",
            horizontalalignment="center",
            color="#24457d",
            fontsize=9,
        )

    axis.set_xticks(msb_thresholds)
    axis.set_xticklabels([f"{threshold:.2f}" for threshold in msb_thresholds])
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("MSB prior Top-p threshold")
    axis.set_ylabel("Retained token count")
    axis.set_title(
        f"{case_name.title()} Attention: Top-p Threshold vs Retained Tokens\n"
        f"Full-INT8 threshold fixed at {ref_threshold}",
        fontsize=14,
        fontweight="bold",
    )
    axis.grid(color="#d6d6d6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="best")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Threshold-count saved  : {output_path.resolve()}")
    print(f"  Full-INT8 p={ref_threshold}     : {ref_count} tokens")
    for threshold, count in zip(msb_thresholds, msb_counts):
        print(f"  MSB p={threshold:.2f}          : {count} tokens")

    if show:
        plt.show()
    plt.close(figure)


@torch.inference_mode()
def run_single_case(args: argparse.Namespace, case_name: str) -> None:
    """Run one focused or diffuse REF/OUT comparison case."""

    case_builders = {
        "focused": make_focused_attention_case,
        "diffuse": make_diffuse_attention_case,
    }
    query, keys = case_builders[case_name](
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        seed=args.seed,
    )
    q_int8, q_scale = symmetric_int8_per_token(query)
    k_int8, k_scale = symmetric_int8_per_token(keys)

    full_int8_logits = quantized_attention_logits(q_int8, k_int8, q_scale, k_scale)
    msb_logits = msb_attention_logits(
        q_int8,
        k_int8,
        q_scale,
        k_scale,
        msb_bits=args.msb_bits,
    )

    # Names requested by the experiment: REF is Full-INT8, OUT is MSB-only.
    ref = top_p_from_logits(full_int8_logits, args.top_p)
    out = top_p_from_logits(msb_logits, args.top_p)
    ref_row = select_row(ref, args.head)
    out_row = select_row(out, args.head)

    output_path = args.output_dir / f"MSBTop_p_{case_name}_bar_chart.png"
    aligned_output_path = (
        args.output_dir / f"MSBTop_p_{case_name}_same_id_grouped_bar_chart.png"
    )
    ratio_output_path = (
        args.output_dir / f"MSBTop_p_{case_name}_same_id_ratio_bar_chart.png"
    )
    qk_ratio_output_path = (
        args.output_dir / f"MSBTop_p_{case_name}_same_id_qk_score_ratio_bar_chart.png"
    )
    recall_budget_output_path = (
        args.output_dir / f"MSBTop_p_{case_name}_recall_budget_curve.png"
    )
    threshold_count_output_path = (
        args.output_dir / f"MSBTop_p_{case_name}_threshold_token_count_curve.png"
    )
    print(f"\n[{case_name.upper()} ATTENTION]")
    print("Attention-like tensor shapes")
    print(f"Q                       : {tuple(query.shape)}")
    print(f"K                       : {tuple(keys.shape)}")
    print(f"attention probabilities : {tuple(ref.probabilities.shape)}")
    print(f"plotted batch/head/query: 0/{args.head}/0")
    print_comparison(ref_row, out_row, args.top_p)
    plot_probability_bar_charts(
        ref=ref_row,
        out=out_row,
        case_name=case_name,
        threshold=args.top_p,
        msb_bits=args.msb_bits,
        output_path=output_path,
        show=args.show,
    )
    plot_aligned_id_comparison(
        ref=ref_row,
        out=out_row,
        case_name=case_name,
        msb_bits=args.msb_bits,
        output_path=aligned_output_path,
        show=args.show,
    )
    plot_same_id_probability_ratio(
        ref=ref_row,
        out=out_row,
        case_name=case_name,
        output_path=ratio_output_path,
        show=args.show,
    )
    full_int8_logits_row = full_int8_logits[0, args.head, 0]
    msb_logits_row = msb_logits[0, args.head, 0]
    qk_score_ratios, valid_qk_ratio_mask = compute_same_id_qk_score_ratio(
        msb_logits=msb_logits_row,
        full_int8_logits=full_int8_logits_row,
    )
    plot_same_id_qk_score_ratio(
        ratios=qk_score_ratios,
        valid_mask=valid_qk_ratio_mask,
        full_int8_logits=full_int8_logits_row,
        msb_logits=msb_logits_row,
        aligned_indices=ref_row.sorted_indices,
        case_name=case_name,
        output_path=qk_ratio_output_path,
        show=args.show,
    )
    plot_recall_budget_curve(
        ref=ref_row,
        out=out_row,
        case_name=case_name,
        ref_threshold=args.top_p,
        output_path=recall_budget_output_path,
        show=args.show,
    )
    plot_top_p_threshold_token_counts(
        ref_threshold=args.top_p,
        full_int8_logits=full_int8_logits_row,
        msb_logits=msb_logits_row,
        case_name=case_name,
        output_path=threshold_count_output_path,
        show=args.show,
    )


def run_cases(args: argparse.Namespace) -> None:
    """Validate arguments and run the selected attention case or both."""

    if args.num_heads < 1:
        raise ValueError("num_heads must be positive")
    if args.seq_len < 4:
        raise ValueError("seq_len must be at least 4")
    if args.head_dim < 1:
        raise ValueError("head_dim must be positive")
    if not 0 <= args.head < args.num_heads:
        raise ValueError(f"head must be in [0, {args.num_heads - 1}]")

    selected_cases = ("focused", "diffuse") if args.case == "both" else (args.case,)
    for case_name in selected_cases:
        run_single_case(args, case_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Full-INT8 and MSB-slice Top-p on attention-like Q/K."
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head", type=int, default=0, help="Attention head to visualize.")
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--msb-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--case",
        choices=("focused", "diffuse", "both"),
        default="both",
        help="Select one attention distribution or generate both (default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory in which bar-chart PNG files are saved.",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot window.")
    return parser.parse_args()


if __name__ == "__main__":
    run_cases(parse_args())
