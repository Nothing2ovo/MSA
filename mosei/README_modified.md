# MOSEI DHM + Multi-layer Hypergraph Regularization

This version applies the first-priority change only:
- keep the anti-collapse hypergraph structure,
- change hypergraph regularization from last-layer-only to multi-layer regularization,
- emphasize edge std / edge spread,
- reduce the relative emphasis on cross/intra gap,
- add late-layer anti-collapse preservation terms.

Modified files:
- hypergraph.py
- utils.py
