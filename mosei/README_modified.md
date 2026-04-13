# MOSEI DHM + Anti-collapse Hypergraph

This version keeps the paper-aligned sequence-level hypergraph, then adds anti-collapse mechanisms.

Main changes relative to the paper-aligned version:
1. Sequence-level shared nodes are kept for HGL as before.
2. Edge weights are no longer only slot-based learnable diagonals. They become:
   - base learnable logits
   - plus content-adaptive residual scores from edge representations
   - with separate cross-edge and intra-edge weighting networks
3. Hypergraph anti-collapse regularization is added into the training loss:
   - minimum global edge std
   - minimum cross/intra gap
   - minimum top-bottom edge spread
4. Training logs now show edge std / cross-intra gap / edge spread explicitly.
