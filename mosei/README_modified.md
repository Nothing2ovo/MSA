This patch targets two specific issues in the uploaded MOSEI codebase:

1. Hypergraph higher-order structure was not really being learned.
   - Added edge_type-aware and prior-aware hyperedge weighting.
   - Cross-modal and intra-modal hyperedges now carry similarity priors.
   - Added hypergraph structure regularization to avoid near-constant edge weights.
   - Added extra diagnostics: edge_weight_std, cross_edge_weight_std, intra_edge_weight_std, cross_intra_gap.

2. 4-token fusion was almost always collapsing toward text-private.
   - Reworked token fusion into a shared-hypergraph-dominant design.
   - Added shared-prior logits, shared minimum weight, and dominance-margin regularization.
   - Private tokens are preserved as supplements instead of being the main route.
   - TGIB pooling now uses token weights rather than plain mean pooling.

Files:
- hypergraph.py
- model.py
- utils.py
- train.py
- data_loader.py (copied unchanged for completeness)

Use these files to replace the originals in your project directory.


Paper-aligned hypergraph revision:
- Hypergraph now consumes shared sequence nodes directly [B,3,T,D].
- Cross-modal hyperedges are built per sample/time step.
- Intra-modal hyperedges are built within-batch using same-modality top-k neighbors.
- Hypergraph edge weights use learnable diagonal slots rather than content-conditioned MLP.
- Extra custom hypergraph regularization has been removed from optimization to stay closer to the paper objective.
