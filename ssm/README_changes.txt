Official-Mamba Shared Selective State Mixer patch

Main changes:
1. Removed the shared hypergraph branch from model.py.
2. Added shared_mamba.py, which calls official mamba_ssm.Mamba directly.
3. Replaced shared hypergraph processing with SharedSelectiveStateMixer:
   - intra-modal shared Mamba scan for text/vision/audio shared sequences;
   - text-centered cross-modal Mamba scan for vision-text and audio-text shared interactions;
   - optional lightweight self-attention after cross-modal Mamba, following CHM's hybrid idea;
   - gated fusion of intra-modal and cross-modal shared summaries.
4. Updated utils.py interfaces from hypergraph_* to shared_mixer_*.
5. Added shared_aux_loss: refined shared token has an auxiliary prediction head, making shared output directly sentiment-discriminative.
6. Updated train.py logs, config, history, and final result writing to use shared_mixer metrics.

Before training:
python -m pip install -U pip
python -m pip install "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2"

Current experimental variant:
1. SharedSelectiveStateMixer is changed to an intra-modal Mamba mixer only.
   It scans text/vision/audio shared sequences independently and no longer
   performs text-centered cross-modal pair scans inside the Mamba branch.
2. The shared branch now emits three shared tokens:
   text-shared, vision-shared, and audio-shared.
3. Final fusion is changed from 4-token fusion to 6-token fusion:
   text-shared / vision-shared / audio-shared /
   text-private / vision-private / audio-private.
4. Cross-modal interaction is now concentrated in the downstream token-level
   Transformer fusion, while Mamba focuses on continuous intra-modal dynamics.
5. The 6-token prior is rebalanced to reduce text-private dominance, and the
   shared mixer regularizer now constrains refined shared-token alignment,
   anti-collapse, modality-attention balance, and token-norm balance.
6. Residual passthroughs are removed from the shared Mamba path, TMoE experts,
   token fusion block, and shared feature builder. These modules now return
   their transformed outputs directly instead of original features plus a
   correction term.

Core model path:
Factorized shared/private features -> shared branch: intra-modal official-Mamba mixer -> private branch: Transformer/TMoE experts -> 6-token fusion -> sentiment prediction.
