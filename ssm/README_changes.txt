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

Core model path:
Factorized shared/private features -> shared branch: official-Mamba Shared Selective State Mixer -> private branch: Transformer/TMoE experts -> 4-token fusion -> sentiment prediction.
