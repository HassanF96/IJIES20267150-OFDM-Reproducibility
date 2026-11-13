# Risk-calibrated neural residual OFDM acceptance

Research implementation and retained-data reproduction for IJIES manuscript 20267150, **Risk-Calibrated Acceptance of Neural Residual OFDM Channel Estimates**.

Public release: `v1.0.0`. Authors: Hasan Faez Yahya, Ivan Abd Al-Zahra Hashim, Mohammed Husham Alkhafaji.

```sh
git clone --branch v1.0.0 https://github.com/HassanF96/IJIES20267150-OFDM-Reproducibility.git
cd IJIES20267150-OFDM-Reproducibility
```

## Scope and contents

This release contains receiver, training and evaluation source; fixed configurations and split/seed manifests; all five residual and all five locally adapted HELENA checkpoints; train-only prior statistics; original and Study 2 calibration/test arrays; and retained post-review phase, higher-Doppler and measured timing records. `METHODS.md` describes the frozen numerical implementation. `results/` contains descriptive retained-data summaries. No manuscript, referee correspondence, portal record, internal project audit, personal cache or Git history from the parent project is included.

This is a retained-output reproduction package. Original waveform campaign entry points retain their historical Git freeze/custody guards: a fresh checkout does **not** reproduce the entire historical waveform campaign with one command. No neural training or communications simulation was rerun for this release. The original calibration certificates, numeric arrays and checkpoints are preserved byte for byte; historical DLL locations in their metadata are provenance strings, not runtime dependencies. They do not point to files shipped in this repository.

## Environment

Use Python 3.13 and a compatible scientific Python environment. The recorded experiment used Python 3.13.15, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.0+cu126, Sionna 2.2.0 and pandas 3.0.6 on Windows 11, with one native thread for timing. `RECORDED_ENVIRONMENT.json` is the recorded runtime; `requirements-installed.txt` is the original package inventory, **not** a platform-independent binary lock. The commands below use the selected environment's `python`; activate that environment first. CPU replay is supported. Different software/hardware can change numerics and timing.

## Verified commands (from the repository root)

```sh
python -B verify_release.py
python -B revision/scripts/audit_certificates.py --root .
python -B revision/scripts/helena_summary.py
python -B revision/scripts/convergence_history.py
python -B revision/scripts/training_reproduction.py --help
```

The first checks release hashes, parses source and checks all ten serialized checkpoint archives without importing PyTorch, inference or training. The independent certificate audit replays the 195 sealed certificates and 12,960 tested rows from saved calibration arrays; it does not fit thresholds or change certificates. HELENA summaries use the exact same 8,192 Study 1 frame IDs for five fixed models, including poor-convergence seed 5104. Pointwise paired intervals are post hoc descriptive, not a multiplicity-controlled or trained-population superiority claim. Convergence plots use all 120 recorded development epochs of each seed. Generated outputs go under `reproduction_outputs/`.

## Optional retraining and historical receiver fixture

The following are supplied entry points, **not executed for this release**:

```sh
python -B revision/scripts/training_reproduction.py --mode smoke --model both --seed 5101
python -B revision/scripts/training_reproduction.py --mode full --model both --seed 5101
python -B revision/scripts/cleanroom_fixture.py
```

Training is isolated to a new digest-bound `revision/training_replay/` directory and cannot read retained final calibration/test outcomes or overwrite fixed weights. Full mode retrains the selected architectures, not architecture search. Run each seed 5101–5105 separately if desired. The historical receiver fixture includes a newly generated synthetic engineering frame and writes only its own `revision/verification/` outputs; it is outside the retained-data verification above. Full retraining and new simulation are optional work for a reproducer, not claims made by this release.

## Integrity and provenance

`MANIFEST.csv` lists the size and SHA-256 of every release file except itself and `SHA256SUMS.txt`. `SHA256SUMS.txt` additionally binds the manifest and excludes only itself. Git working metadata and generated outputs are excluded. The author-submitted reproduction ZIP is built from exactly this release tree; its external SHA-256 is reported in Supplement C8 and the GitHub release assets. GitHub's automatically generated source ZIP is a different archive and has a different hash.

The historical v2 archive remains separately preserved by the authors for provenance; it is not overwritten or claimed to be byte-identical to this public release. Frozen negative findings, calibration decisions, raw arrays, and all five-model results are retained. HELENA is a disclosed local adaptation, not an exact reproduction or an assessment of the published method's intrinsic quality.

Original implementation code is available under the MIT license in `LICENSE`. Third-party dependencies retain their own licenses; no third-party package binaries are redistributed. All bundled numeric research artifacts are author-generated and accompany this reproducibility release.
