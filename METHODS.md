# E7 frozen receiver and estimator definitions

Status: FROZEN BEFORE FINAL DATA. Governing baseline: `1b21c1f32deaeb7b9fc0312b88a35ff88e906e17`. The unchanged tested implementations in `src/e5`, `src/e6`, `src/estimators`, `src/ofdm`, `src/channel`, and `src/interference` are bound in the freeze manifest. This document and `configs/final` fix the numerical implementation. No estimator or gate is refitted on calibration/test. All complex receiver calculations are complex128, summaries float64; neural inference is float32, converted back to complex128.

## Physical and information boundary

The frozen frame is 128 FFT bins by 8 symbols, CP16, 96 active signed bins [-48,-1] union [1,48], sampling1.92MHz and spacing15kHz. Unitary OFDM transforms and serialized-frame linear convolution are unchanged. Pilots have value (1+j)/sqrt(2); data and pilot ensemble energy are1. Staggered pilot positions are active-list indices `ell % stride :: stride` in symbol ell. Strides4,8,16 mean192,96,48 pilots per frame. Independent proper-complex Gaussian frame-static tap laws A/B/C and Bernoulli time bursts are given exactly in `distributions.yaml`. They are generic laws, not named standardized channel profiles. Family C has never been realized during E7.

Practical inputs are the received frame, public grid/pilot symbols and locations, constellation, frozen train-only statistics and frozen weights. No true channel, true noise/interference parameters, transmitted data, harm labels or oracle information reaches a practical estimator, equalizer or gate. Noise scale is `v=max(median(|Y_null|²)/ln(2),1e-10)`, common to a full candidate pair; G2 recomputes the same null-only estimate for its partial/full calls. These nulls can contain interference. This is not an oracle noise estimate.

## Baseline and residual candidate

Delay basis `B[k,ell,d]=exp(-2*pi*j*k*d/128)`, d=0,...,16. On allowed pilots F, `A=B[F]*X_p[F,None]`, `y=Y[F]`. The frozen train-only mean mu and covariance R are loaded from the hashed `pilot_statistics.npz`. Their development estimation used4096 train frames, pilot ridge1e-6, centered sample covariance, noise subtraction, nonnegative eigenvalue projection, .1 diagonal shrinkage and eigenvalue floor1e-8. No statistics are recomputed in E8.

Writing C=Cholesky(R), W=A C, the baseline coefficients are

`a_b = mu + C (W^H W + (v+1e-8)I)^(-1) W^H (y-A mu)`.

The baseline is H_b=B a_b with inactive bins zero. A linear solve is used, not an explicit inverse. This is the practical LMMSE-like estimator with fixed train statistics, not matched oracle LMMSE.

Let `s=sqrt(max(mean_F |Y_p/X_p|²,1e-12))`. The seven CNN input channels are sparse-LS real/imag divided by s, H_b real/imag divided by s, allowed-pilot mask, active mask, and the constant v/s². Disallowed pilots are zero before every feature/normalization operation. The width48 CNN has layers `7→48(3x3)→48(3x3)→48(3x3)→2(1x1)` with padding1 on3x3 and ReLU after the first three convolutions, no batch norm/dropout. Its output is multiplied by the active mask. `H_n=H_b+s*(output_real+j*output_imag)`. Parameter count44738 is asserted from the loaded models. All five existing best checkpoints are immutable.

## G3-hard

For each data resource element and candidate h, `q=conj(h)/(|h|²+v/Es)`, `z=qY`, gain=`q h`, and effective variance=`|q|²v`. The hard decision is exactly `d=argmin_{a in constellation}|Y-h*a|²`, first constellation-array entry on an exact tie. This receive-domain form is the numerically stable gain-aware MMSE decision; no division by a vanishing gain occurs. Constellation ordering, Gray bit mapping and unit-energy normalization are the existing `src/ofdm/modulation.py` code, hash-bound.

Define `u_h=|Y-h*d|²/v`. With `k=sqrt(ln(20))`, define radial Huber on squared radius as `rho(u)=u` for u<=k², and `2*k*sqrt(u)-k²` otherwise. `J_hard(h)=mean_data rho(u_h)`. The paired score is `T_hard=J_hard(H_n)-J_hard(H_b)`. Accept the full neural estimate exactly when `T_hard<=tau` and the score/candidates are finite; otherwise use H_b. No pilot/phase anchor enters the primary score. Negative scores favor the neural candidate. Mean is over all data resource elements with equal weight; no trimming, whitening, learned weight, variance subtraction or test-dependent normalization is permitted.

The implemented finite-data contract raises an error on nonfinite candidates/scores. E8 records and stops on that implementation error, preserving the frame ID; it cannot silently omit the frame or change the method. A calibrated null threshold is a distinct valid reject-all fallback. Accepted final decoding uses the same gain-aware equalizer/detector and candidate; rejected decoding uses the baseline.

## G3-soft

For each candidate, uniform-prior posterior at temperature1 is

`p(a|Y,h)=exp(-(|Y-ha|²-min_b|Y-hb|²)/v)/sum_b exp(-(|Y-hb|²-min_c|Y-hc|²)/v)`.

The actual implementation uses stable log-sum-exp. Let `mu_s=sum_a p(a)a`, `var_s=sum_a p(a)|a|²-|mu_s|²`. A value below `-1e-10*Es` is an error; roundoff-sized negative variance is clamped to0. The expected squared receive-domain discrepancy is `u_soft=(|Y-h*mu_s|²+|h|²*var_s)/v`. Huber is applied AFTER this expectation, then averaged over data. `T_soft=mean rho(u_soft,n)-mean rho(u_soft,b)`, accepted in the same lower-score direction. There is no entropy filter, temperature tuning or phase anchor. The `g3_mean_only` ablation drops var_s; `g3_soft_mean` replaces rho by the identity. Neither replaces the frozen secondary soft comparator.

## Ordinary confidence and equivalent controls

All paired controls accept smaller neural-minus-baseline discrepancy; definitions below are exact. `_neural` means the single neural statistic with no paired subtraction.

| ID | Frozen scalar |
|---|---|
| g1 | `-log1p(Es*mean_active|H_b|²/v)`; estimated-SNR control; baseline only before decision |
| evm / evm_neural | mean `|conj(h)Y/max(|h|²,1e-12)-d|²/Es`; paired / neural absolute |
| evm_robust / evm_robust_neural | rho applied to the same unweighted EVM summands, then mean; paired / neural absolute |
| weighted_mean / weighted_mean_neural | mean `|Y-hd|²/v`; paired / neural absolute |
| weighted_normalized / weighted_normalized_neural | `mean|Y-hd|²/max(Es*mean|h|²,1e-12)`; paired / neural absolute |
| weighted_robust_neural | mean rho(`|Y-H_n*d_n|²/v`) |
| weighted_evm_identity | exact robust weighted-EVM equivalent of G3-hard, evaluated with the canonical G3-hard score and shared certificate |
| llr / llr_neural | `C_b-C_n` / `-C_n`, where `C=mean_data,bits |tanh(LLR/2)|`, LLR=`log P(bit=1)-log P(bit=0)` from the posterior above |
| pilot | full-estimate paired mean rho(`|Y_p-h*X_p|²/v`) on all known pilots |
| pilot_mean | same full-pilot paired discrepancy with ordinary mean |
| g3_anchor | `T_hard+T_pilot`, coefficient exactly1, comparator only |

Away from the EVM floor, weighted EVM identity follows from `|h|²*|Y/h-d|²/v=|Y-hd|²/v`; Es cancels the normalized-EVM convention. On floored/zero channels the identity control uses direct receive residual, not an inaccurate division. The separate arithmetic identity diagnostic uses relative/absolute tolerance1e-10. The same canonical score, thresholds, decisions and certificate prevent floating-point duplicate testing. `g3_hard_mean` aliases `weighted_mean`. Neither alias contributes an independent multiplicity slot or a novelty claim.

## G2

Use the E6 selected50% geometry-aware holdout and Huber aggregation. `configs/final/g2_geometry.json` freezes the full flattened fit/heldout index lists for each stride. Geometry is label/observation independent. Its construction starts with the combined full-pilot Gram of the support17 delay basis plus1e-6I, processes symbols0..7 and signed-frequency bins in ascending order, and greedily removes floor(.5*number_of_symbol_pilots) entries maximizing the real log determinant of the remaining Gram. Nonpositive determinant is inadmissible; exact ties choose the first signed-frequency entry. The common Gram is updated after every removal, including across symbols. The registered function and resulting masks are both hashed.

The partial baseline and partial neural candidate use ONLY retained pilots, the same trained weights, null-based v and partial-pilot scale. The validation score is the paired radial-Huber mean of `|Y_hold-h_partial*X_hold|²/v` on heldout pilots. Apply its frozen calibration threshold, then fit the baseline on all pilots, and on acceptance run the full-pilot neural correction. On rejection return the all-pilot baseline. Counts per frame:2 classical estimator calls (partial and full),1 partial network call plus1 accepted full network call,1 heldout validation pass,1 full refit; no reuse of a partial estimate as the final candidate. Labels for its calibration are harm/use of this final full candidate, not partial-fit harm.

At stride16, only24 pilots remain for the partial estimator across8 symbols. The documented delay-design and covariance conditioning limitations remain. No mask, shrinkage, model, ridge or refit change is allowed after seeing final sparse-pilot results. E6 does not support the assertion that partial/full mismatch is inherently harmful.

## HELENA

HELENA is a modern comparator under the disclosed local adaptation. It is not claimed to be an exact reproduction. Existing recovery_0.001 checkpoints for5101–5105 are the final ones; no repair/replacement/extra training is allowed.

The input is unnormalized sparse complex LS on96 signed-order active carriers by8 symbols, real/imag channels, zeros on data positions. Convolutions2→32 with12x2 and32→2 with6x7 kernels use exact SAME asymmetric padding and ReLU; dropout=.1 during training. Twelve-carrier frequency patches, NHWC flattening192, produce8 tokens, embedded192→64. Q/K/V64→256 form4 heads of width64 with attention scale1/8; projected256→64, residual and LayerNorm eps=.001. Mean-token squeeze/excitation64→16(ReLU)→64(sigmoid), reconstruction64→192, unpatchification and sparse-input skip produce the channel. There are97714 parameters. No practical noise/truth feature is supplied.

Glorot convolution/linear weights with zero biases are used, except disclosed attention initializations: Q/K/V uniform ±sqrt(6/(256+4096)), output uniform ±sqrt(6/512); layer norm weights1/bias0. Training used4096 E6 train frames,1536 V-design frames, full inputs, batch64, Adam lr=.001, betas(.9,.999), epsilon1e-7, no weight decay/clipping, unnormalized MSE, maximum120 epochs. ReduceLROnPlateau factor.8, patience40, min_lr1e-5, relative threshold1e-4, cooldown0, eps1e-8. Early stop after50 epochs without strict validation-MSE improvement; earliest strict best checkpoint, no min_delta. All five reached120 epochs in the recorded adaptation. Existing initialization/optimizer/state bytes are authoritative; no TensorFlow parity is claimed.

Differences from the source include96x8 versus612x14 geometry, generic project channels/bursts/modulation/pilot mixture instead of source setup, parameter count, PyTorch training semantics and the disclosed bounded learning-rate adaptation. Report every seed separately and the arithmetic equal-seed aggregate, seed SD and range. Seed5104 is genuine poor convergence/training instability under this fixed adaptation. Never relabel it as an implementation defect merely because it underperforms, and never report only successful seeds. No claim that the proposed estimator universally outperforms the original HELENA is eligible.

## Traditional and ideal references

LS uses signed-frequency linear interpolation independently per symbol, separate real/imag interpolation and constant edge extrapolation. Regularized delay LS uses support17 and ridge1e-6. Block-Huber uses the fixed train prior, symbol blocks, k_s=sqrt(GammaPPF(.95,n_s)/n_s),5 IRLS updates after an initial MAP solve, weights min(1,k_s/sqrt(max(normalized_block_energy,1e-30)));6 solves total. These are unchanged registered implementations.

Perfect CSI uses true channel and true background N0 only in the ideal namespace. Matched-statistics LMMSE uses the exact family delays/powers and true background N0, solve floor1e-12; it does not know realized burst locations/covariance. Oracle gate chooses the full neural candidate iff its true linear NMSE is strictly smaller than baseline, else baseline. Oracle gate minimizes NMSE, not BER; it is not asserted to minimize bit error on every frame. All ideal references are labeled unavailable to a practical receiver.
