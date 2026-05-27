# PI-JEPA Follow-Up Paper — Working Outline

**Working title (TBD):** Label-Free Pretraining for PDE Surrogates: PI-JEPA Done Right

**System name (fixed):** PI-JEPA (extending Yee & Koh 2026).

**Target venue:** NeurIPS / ICML / ICLR main track.

---

## Branch and code-hygiene notes (audit findings, permanent record)

Brandon shipped 26 new files in `upstream/fourier-jepa` (the revision branch).
A code audit found which are safe to use:

**Safe and wired** (use as-is): `models/brooks_corey_conditioner.py`,
`models/pvt_conditioner.py`, `training/curriculum.py`,
`training/learned_weights.py`, `physics/spectral_residual.py` (only
when `H == W == self.resolution`), `data/irregular_grid.py` (NaN-fill only),
`training/multi_fidelity.py` (sampler utility only, wired via
`data/combined_pool.py`).

**Safe but orphan** (real code, must invoke manually): `benchmarks/pod_baseline.py`
(steady-state only; `rollout()` is fake), `eval/uq_ensemble.py`,
`scripts/generate_{spe10,compositional,real_field,sgs_corpus}_data.py`,
`scripts/download_data.py`.

**DO NOT USE** (audit found bugs / dead code):
- `models/fourier_encoder_3d.py` — **cubic-only**, rectangular inputs silently
  squashed via `adaptive_avg_pool3d`. Workaround: resize everything to a cube
  (we use `64³`).
- `physics/latent_flux.py` — loss is minimized when all patch embeddings are
  identical → representation collapse. Documented in its docstring.
- `physics/tpfa.py` — 2D pressure-only despite "3D" framing; insufficient for
  the 3D Darcy claim.
- `models/well_conditioner.py` — wired in `pretrainer.py:582-583` but
  `batch.get('well_controls')` always returns None on every existing dataset.
  Dead-by-data.
- `training/adaptive_collocation.py` — `sample()` never called; inert.
- `scripts/figures/*.py` (six scripts) — all fall back to synthetic data;
  use `scripts/make_paper_figures.py` instead.
- `eval/optimization.py` — random search labeled as optimization; fake
  timing.

The "cubic-only encoder" is the most critical: any pipeline configured for
rectangular volumes will produce **silently wrong** embeddings (no error).
All our scripts that touch the encoder pass `--resize-cube 64` to enforce
the cube.

---

## Focused thesis (one sentence)

> PI-JEPA — pretrained on **free unlabeled** parameter fields with a **true
> operator-split chain** — beats supervised baselines and from-scratch PI-JEPA
> when **labeled** PDE solves are scarce.

That's the whole claim. Three nouns: **unlabeled fields**, **labeled solves**,
**PDEs**. Everything else in this codebase (multi-fidelity, Strang splitting,
spectral residuals, cross-domain transfer, 5-dataset evaluation) is *extension*
material moved to the appendix.

---

## Main paper experiments

| Table / figure | What it shows | Compute |
|---|---|---|
| **Figure 1: Sample-efficiency curve** | Relative ℓ₂ vs N_ℓ ∈ {10, 25, 50, 100, 250} for {PI-JEPA, PI-JEPA-from-scratch, FNO3D}, 5 seeds, bootstrap CI | 75 fine-tunes + 5 pretrains |
| **Table 1: Headline numbers** | Same as Figure 1 but tabular for citation | (free from Fig 1) |
| **Table 2: Focused ablation** | 3 variants at N_ℓ = 100: full / no-chain / no-physics | 15 pretrains + 15 fine-tunes |
| **Figure 2: Qualitative panels** | Best/worst predictions at N_ℓ = 100 (truth vs pred vs error) | free (uses ckpt) |

**Total compute**: ~110 runs, ~190 A100-hours, ~$285 at $1.50/hr.

Driver: `./scripts/run_focused_paper.sh outputs_focused/v1 darcy_3d_synthetic`
(or `... ccsnet` once the CCSNet pretrain finishes on the test split).

---

## Appendix material (built but not load-bearing)

These exist in the codebase as ablation rows / extension experiments. They're
documented but the main paper's claim does not depend on them.

---

## What the original paper claimed vs. delivered

| Original claim | What the original code actually did | What this follow-up delivers |
|---|---|---|
| Per-sub-operator predictor chain (Lie-Trotter) | Each predictor overwrote target with `mask_token` — chain degenerated to additive ensemble | **True chained predictors via `forward_chained`**; verified stage 0 ≠ stage 1 by mean abs 0.59 |
| K decoders, one per sub-operator | One decoder applied to final output only | **K independent Decoder3D instances** built per-stage via `decoder.per_stage=True` |
| Per-sub-operator PDE residuals summed in the loss (Eq. 6) | One physics residual on the final decoded field | **Per-stage residuals** iterated over `stage_outputs`, summed with ramping λ_p |
| Two-phase Darcy with pressure + saturation (Eq. 10, 11) | Loader silently threw saturation away; only first timestep of pressure used | **Time-preserving loaders** for CCSNet/FNO4CO2; pressure + saturation both available |
| 3D capability (future work in original) | 2D 64×64 only | **3D rectangular grids** via FourierJEPAEncoder3D with `image_size = [T, H, W]` |
| Multi-channel inputs | 1-channel input only | **Multi-channel** via `encoder.in_channels` config (verified on FNO4CO2's 12-channel input) |
| Physics-informed claim with neutral physics-residual ablation (-8.1%) | Used a broken finite-difference residual that pushed `p` toward `K` (admitted in code comments) | **Brooks-Corey Eq. 10** relperm form available via `brooks_corey_rel_perm`; spectral-residual variant on roadmap |

---

## Extension table (appendix material)

| Extension | What it adds | Where it lives in code | Status |
|---|---|---|---|
| **Multi-fidelity Tier-1** | Adds cheap coarse-grid simulations between unlabeled fields and expensive labeled solves | `scripts/generate_darcy_tier1.py`, `PI-JEPA/data/multifidelity.py`, `configs/darcy_3d_mf_smoke.yaml` | Working; smoke-test passed |
| **Strang splitting** | Higher-order operator splitting (2nd order) vs Lie-Trotter (1st order) | `model.predictor.splitting: strang` | Working; ablation row |
| **Spectral physics residuals** | rFFT-based exact derivatives vs FD residuals | `physics.residual_type: spectral` | Working; validated to machine epsilon |
| **CCSNet (real CO₂ data)** | Real-world subsurface field test instead of synthetic | `PI-JEPA/data/ccsnet_loader.py`, `configs/ccsnet_3d_*.yaml` | Test split downloaded; train split downloading |
| **FNO4CO2** | Multi-channel 12-input CO₂ data | `PI-JEPA/data/fno4co2_loader.py` | dP test downloaded; train downloading |
| **PDEBench ADR** | Reactive transport dataset | (loader pending download completion) | Downloading |
| **Cross-domain transfer matrix** | Pretrain-on-A / fine-tune-on-B for all (A,B) | `scripts/run_cross_domain.py` | Working |
| **Additional baselines (U-FNO3D, PINO3D)** | Stronger comparison set | `PI-JEPA/benchmarks/{ufno_3d,pino_3d}.py` | Working |

If the main result is convincing, these become "we also verified..." in
the appendix. If the main result is weak, these become alternative claims
the paper falls back on.

---

## Three contributions this paper makes

### (i) Methodological: true operator-split JEPA
- Per-stage predictor chain (Lie-Trotter); ablation against monolithic and additive-ensemble baselines
- Per-stage decoders + per-stage physics residuals; ablation against shared-decoder
- Honest comparison: report cases where per-stage doesn't help

### (ii) Practical: multi-fidelity pretraining
Three-tier hierarchy that exploits the *real* CO₂-storage data asymmetry:
- Tier 0 (free): permeability/porosity parameter fields generated by GRF
- Tier 1 (cheap): coarse-grid simulations (e.g., 16³ or 32×32×8) — minutes per run
- Tier 2 (expensive): fine-grid simulations (e.g., 64³+ or 96×200×24) — hours per run

Pretrain on Tier 0 → continue-pretrain on Tier 1 → fine-tune on Tier 2.
The original paper listed this as future work; we deliver it.

### (iii) Honest fix for the negative physics-residual finding
Original paper's finite-difference physics residual was neutral-to-harmful. We:
- Reproduce the original neutral result with the FD residual
- Add **spectral physics residuals** (compute derivatives via rfftn instead of FD)
- Report whether spectral residuals turn the negative result positive
- Caveat: if it doesn't help either, that's still a real result we'll report

---

## Datasets (5)

| Dataset | Type | Native shape | Status |
|---|---|---|---|
| **CCSNet** | 2D radial CO₂ + time + multi-target | (1600, H=96, W=200, T=24, C=1) | Test split downloaded; train_x in progress |
| **FNO4CO2 (U-FNO)** | 2D radial CO₂ + time + multi-channel input | (4500, H=96, W=200, T=24, C=12) | dP_test_a/u done; dP_train_a in progress |
| **PDEBench ADR** | 2D + time + multi-species | (10000, T, H, W, n_species) | Slow DaRUS download |
| **Fourier-MIONet** | 2D + time + scalar parametric (injection rate) | Same as CCSNet + scalar | Needs your OneDrive download |
| **SPE10 (via OPM Flow)** | TRUE 3D Cartesian (60×220×85) heterogeneous | OPM ECLIPSE deck output | Needs OPM Flow install |

---

## Baselines (8)

Per top-venue conventions (see research notes):

| Baseline | Family | Status in this codebase |
|---|---|---|
| FNO | Fourier neural operator | wrapper exists, needs rectangular-3D verification |
| U-FNO | FNO + U-Net | not yet wrapped |
| Nested FNO | basin-scale 3D FNO | not yet wrapped |
| DeepONet | branch-trunk operator | wrapper exists |
| U-DeepONet | U-Net + DeepONet | not yet wrapped |
| CCSNet | CNN encoder-decoder | not yet wrapped (CCSNet trained_models/ has pretrained weights) |
| Fourier-MIONet | multi-input neural operator | not yet wrapped |
| PINO | FNO + PDE residual | wrapper exists, needs shape-fix verification |
| Transolver or FactFormer | transformer operator | not yet wrapped — required for top venue |
| GNS or MeshGraphNets | graph operator | not yet wrapped — required for top venue |

---

## Experiments

### Headline tables
- **Table 1: Main results.** For each dataset, relative ℓ₂ error at N_ℓ = {10, 25, 50, 100, 250, 500} fine-tuning samples, mean ± 95% bootstrap CI over 5 seeds, for PI-JEPA vs all baselines.
- **Table 2: Ablations on CCSNet.** Full | -per-stage chain | -per-stage residuals | -multi-fidelity | -spectral residuals | -VICReg | -temporal masking
- **Table 3: Cross-domain transfer.** Pretrain on Dataset A, fine-tune on Dataset B, all (A, B) pairs.

### OOD/generalization probes
- **Pe/Da sweep on PDEBench ADR** (9 regimes, evaluate on held-out (Pe, Da))
- **New-well generalization on SPE10** (500-1000 OPM runs, varied well location)
- **Long-rollout stability** (CCSNet 30-yr vs FNO4CO2 30-yr) — error vs rollout step

### CO₂-specific metrics (beyond relative ℓ₂)
- IoU on saturation plume
- Plume migration distance error
- Mass conservation error per timestep
- CO₂ breakthrough time
- Pressure buildup at injector

### Reproducibility
- 5+ seeds per cell, bootstrap CI
- HPO budget parity (same trial count for all baselines, disclosed in appendix)
- Code release: MIT/Apache, anonymized GitHub during review
- Datasheet for synthetic Tier 1 data
- Compute disclosure table (GPU model, wall time, total GPU-hours, CodeCarbon kgCO₂eq)

---

## Where we are now

**Smoke tests passed end-to-end:**
1. Synthetic 3D Darcy (physics off + on)
2. CCSNet 2D collapsed (64×64)
3. CCSNet 3D time-preserved (24×96×96) physics off
4. CCSNet 3D + per-sub-operator physics
5. FNO4CO2 lite 3D time-preserved (12-channel, 24×96×96)

**Downloads in progress:** CCSNet (32G/100G+), FNO4CO2 (34G/80G+), PDEBench (448M/13G), Fourier-MIONet (needs manual).

**Engineering remaining:**
- Wire FNO/DeepONet/PINO baselines for rectangular 3D
- CCS-specific baseline wrappers (U-FNO, CCSNet-model, Fourier-MIONet-model)
- Transolver or FactFormer (transformer operator family)
- Multi-fidelity Tier 1 coarse-grid generator
- Spectral physics residuals
- Reactive-transport physics decision (Eq. 14 vs Eq. 15 mixing)
- OPM Flow install + SPE10 generation
- Eval harness with bootstrap CI + multi-seed + per-dataset reports
- Reproducibility scaffolding (datasheet, compute table, env lock)

**Compute target:** NVIDIA Brev for full runs (once org access sorted). Local MPS for smoke + iteration.
