# Non-RS measured results after RS cancellation

The user cancelled RS-INT8 experiments. This directory contains 42 measured non-RS WikiText-2 groups and exactly 42 per-group saved-K ratio tables. The ratio denominator is the causally allowed K-position set in the historical dense INT-Flash trace; that dense baseline saved no importance mask. Captures cover layers 0, 16, and 31, two query slots and 32 query heads per layer. The ratios describe these six trace slots, not the whole test corpus.

- [PPL table](TABLE_PPL.md)
- [Important-K ratio tables](TABLE_K_RATIOS.md)
- [Independent review](REVIEW.json)

Full local evidence, including per-head checks, is under `/output/project/experiments/stp-nonrs-cancelled-summary-v1`. Historical FP/INT dense PPL values remain measured but lack a matching archived full release gate. No RS result is included.
The published Markdown copy has a normalized final newline; the authoritative hash-bound source is the local evidence directory. CSV files retain their source line endings.
