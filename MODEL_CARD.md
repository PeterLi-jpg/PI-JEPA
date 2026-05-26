# Model Card: PI-JEPA (3D Operator-Split Multi-Fidelity Variant)

Following Mitchell et al. ([arXiv:1810.03993](https://arxiv.org/abs/1810.03993)).
This card covers the PI-JEPA system as extended in this follow-up paper:
true Lie-Trotter (or Strang) chained predictors, K per-stage decoders,
per-sub-operator physics residuals, optional spectral physics, optional
multi-fidelity Tier-1 continuation, rectangular 3D Fourier encoder with
attention.

---

## Model details

- **Developed by**: PeterLi-jpg, extending the original PI-JEPA by Yee & Koh (2026).
- **Model name**: PI-JEPA (system name unchanged from original).
- **Version**: this fork, 2026-05-23 snapshot.
- **Architecture summary**:
  - Encoder: Fourier-enhanced 3D ViT-style backbone, ~8.1M params for a
    `(C=1, T=24, H=96, W=96)` config with `embed_dim=192, patch=(4,8,8)`.
  - Target encoder: EMA copy of encoder (frozen during fine-tuning).
  - Predictor bank: K transformer predictors chained via
    `Predictor.forward_chained` (true Lie-Trotter) or via Strang splitting.
  - Decoders: K independent `Decoder3D` instances (per_stage=true) for
    per-sub-operator physics residuals.
- **License**: code under MIT (matches the original repo).

## Intended uses

- **Primary**: self-supervised pretraining on unlabeled subsurface
  parameter fields (permeability, porosity), then fine-tuning on a
  small labeled corpus (≤500 trajectories) for surrogate modeling of
  CO₂ storage / multiphase flow / reactive transport.
- **Secondary**: cross-domain transfer (pretrain on one PDE family,
  fine-tune on another) and multi-fidelity continuation pretraining
  (parameter fields → coarse sims → fine sims).

## Out-of-scope uses

- Operational decision-making on a real CO₂ storage site — this is a
  research surrogate, not a certified production model. Always validate
  against full-physics solvers for any safety-critical decision.
- Long-rollout extrapolation beyond the training horizon (the original
  paper noted post-injection-phase error growth; this fork does not yet
  ablate that).
- Domains where the (T, H, W) tensor convention doesn't map cleanly
  (e.g., point-cloud / mesh data — use GNS or MeshGraphNets there).

## Factors

- **Spatial axes**: cubic or rectangular (D, H, W) supported. Aspect
  ratios far from 1:1 work but were primarily smoke-tested at 24×96×96
  (CCSNet) and 32³ (synthetic Darcy).
- **Channel count**: from 1 (synthetic Darcy) to 12 (FNO4CO2). Other
  multi-physics datasets (multi-species ADR) supported in principle.
- **Time as a third spatial axis** (U-FNO convention): supported via
  the time-preserving loader mode (`t_index=None`).

## Metrics

Evaluated with bootstrap 95% CI over ≥5 seeds (NeurIPS Q7-compliant):

- `relative_l2` — primary surrogate accuracy metric
- `nrmse` — scale-independent RMSE
- `max_err` — worst-case voxel error
- `conservation_residual` — temporal mass-conservation check
- `fourier_band_rmse` — low/mid/high spectral-band accuracy
- `saturation_iou` — plume binarization IoU (CO₂-specific)
- `plume_centroid_error` — mass-weighted centroid distance

All implementations in `PI-JEPA/eval/paper_metrics.py`.

## Training data

- **Pretraining**: unlabeled inputs only (no PDE solves required).
  Sources: synthetic GRF Darcy fields (32³), CCSNet permeability fields
  (96×200 radial × 24 time), FNO4CO2 multi-channel fields, optional
  Tier-1 coarse-grid simulations.
- **Fine-tuning**: small labeled corpus (10-500 trajectories typical)
  from the matching solver outputs.

See `DATASHEET.md` for the synthetic-data datasheet.

## Evaluation data

- Held-out test split per dataset, never seen during pretraining or
  fine-tuning.
- For OOD generalization probes: new Pe/Da regimes (PDEBench ADR), new
  well locations (SPE10 via OPM), or new permeability statistics.

## Quantitative analyses

To be filled with the paper's headline tables. The ablation orchestrator
(`scripts/run_ablations.py`) and cross-domain runner
(`scripts/run_cross_domain.py`) produce these tables as JSON.

Ablation factors that will appear in the paper:
- Lie-Trotter chain vs Strang vs monolithic
- Per-stage decoders vs shared decoder
- FD physics vs spectral physics vs no physics
- Multi-fidelity vs Tier-0-only
- With vs without VICReg collapse prevention

## Ethical considerations

- This is a physical-simulation surrogate; the main risks are
  **operational misuse** (using surrogate predictions for binding
  storage permitting decisions without solver verification) and
  **dataset misrepresentation** (claiming generalization beyond the
  parameter envelope the model was trained on).
- No human subjects, no personal data, no socially-sensitive labels.

## Caveats and recommendations

- The original PI-JEPA paper's *implementation* did not deliver several
  of its stated contributions (per-sub-operator chain, per-stage residuals,
  two-phase saturation supervision); this fork re-delivers them. Any
  comparisons against published PI-JEPA numbers should account for this.
- The spectral physics residual assumes periodic BCs; on Dirichlet-BC
  problems it introduces a small bias compared to the FD residual. The
  paper reports both variants.
- Multi-fidelity Tier-1 uses a Python CG solver — not a production
  reservoir simulator. For real CCS applications use OPM Flow / ECLIPSE
  outputs as Tier 1.

## Reproducibility

- `requirements.lock` — pinned dependency snapshot
- `setup.sh` — environment bootstrap
- `DATASHEET.md` — synthetic data documentation
- `PI-JEPA/utils/compute_disclosure.py` — per-run hardware + wall-clock + carbon JSON
- All experiment scripts emit the same JSON schema for paper-table generation
