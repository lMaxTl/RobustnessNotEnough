# Implementation spec — Extraction economics of adversarially robust CIFAR-10 classifiers

**Audience:** an autonomous coding agent (Claude Code or equivalent) with a single GPU.
**Status:** implementation spec. Every number in §3 is a hard requirement and has an acceptance test in §14.

---

## 0. How to use this document

Implement modules in the order M1 → M10. Each module has a **contract** (public API), a **definition** (exact math or algorithm), and **acceptance tests**. Do not proceed past a module until its acceptance tests pass.

Three rules that override anything else you might infer:

1. **Never invent a number.** If a constant is not in this spec or in a cited source, measure it and record how. Fabricated FLOPs counts destroy the entire paper.
2. **Never change the student architecture, initialisation seed, optimiser, or query pool between victim conditions.** The whole experiment is a controlled comparison across victims. Any drift in the attacker configuration is a confound.
3. **Log everything to the schema in §12.** Analysis reads only that file. If a field is missing the analysis will silently produce a wrong figure.

---

## 1. Scientific claim under test

> The FLOPs cost of *producing* adversarial robustness by training diverges as the target approaches the scaling-law asymptote (~90% AutoAttack accuracy on CIFAR-10). The FLOPs cost of *extracting* that robustness through a query API is approximately flat in the victim's robustness level. Therefore the leverage available to a model thief grows without bound, and adversarial robustness cannot protect itself. Post-training IP protection is structurally required.

**Primary quantity:** the robustness transfer ratio

$$\tau(R_v) = \frac{\mathrm{AA}(\text{student extracted from victim } v)}{\mathrm{AA}(\text{victim } v)}$$

as a function of victim AutoAttack accuracy $R_v$.

**Falsification conditions.** Report these honestly if they occur; a clean negative result is publishable and a laundered positive one is not.

- **F1.** $\tau(R_v)$ decreases steeply in $R_v$ (slope significantly negative, bootstrap 95% CI excluding 0). This means robustness *is* partially self-protecting.
- **F2.** The temperature-matched control (§9.4) reproduces the effect. This means you measured label smoothing, not robustness.
- **F3.** Attacker FLOPs grow super-linearly in $R_v$ at fixed $\tau$, at a rate comparable to the defender curve.

---

## 2. Deliverables

| ID | Artifact |
|----|----------|
| D1 | `costmodel` module reproducing Bartoldson et al. Table 3 exactly |
| D2 | `results.jsonl` — one record per extraction run |
| D3 | `margins.jsonl` — per-victim boundary geometry |
| D4 | `defenses.jsonl` — per-defense attacker-cost multiplier and defender utility cost |
| D5 | Figure 1: defender vs attacker FLOPs by robustness target (log y) |
| D6 | Figure 2: $\tau$ vs $R_v$ with per-attack series and bootstrap CI |
| D7 | Figure 3: margin CDF per victim, with the $8/255$ line marked |
| D8 | Table 1: leverage table at $R \in \{0.737, 0.80, 0.85, 0.88\}$ |
| D9 | `REPRO.md` — exact commands, wall-clock, and hardware |

---

## 3. Constants (verbatim — do not alter)

All from Bartoldson et al., *Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies*, ICML 2024.

```python
# Approach 2 parametric loss, Appendix F.2.1 (restricted fit)
A       =  6.69
B       =  9.89
E       =  0.48
ALPHA   =  0.24     # exponent on N
BETA    =  0.23     # exponent on D
EPS     =  0.16     # quality -> irreducible loss
ZETA    = -0.28     # quality -> data-reducible loss

# Unrestricted-fit variant, Appendix F.2.1 -- use for a robustness check only
A_ALT, B_ALT, E_ALT   = 7.60, 9.70, 0.49
ALPHA_ALT, BETA_ALT   = 0.25, 0.23
EPS_ALT, ZETA_ALT     = 0.14, -0.25

# Loss -> accuracy regression, Appendix F.2.2 (R^2 = 0.98)
ACC_SLOPE     = -0.7496
ACC_INTERCEPT =  1.2575

# FLOPs constraint for adversarial training, Section 3.2.2
# C ~= K_ADV * N * D, <1% relative error over 9000+ observations
K_ADV = 7822.0

# Appendix B: forward-pass equivalents per training iteration
FWD_ADV = 27.0   # 10-step PGD + TRADES
FWD_STD =  3.0   # standard fwd + bwd
```

**Derived, compute at import:**

```
a     = BETA / (ALPHA + BETA)          = 0.489362
b     = ALPHA / (ALPHA + BETA)         = 0.510638
kappa = ALPHA*BETA / (ALPHA + BETA)    = 0.117447
inv_k = 1 / kappa                      = 8.514493
```

---

## 4. M1 — `costmodel.py` (defender side, no GPU)

### 4.1 Contract

```python
def E_prime(fid: float) -> float
def B_prime(fid: float) -> float
def loss(N: float, D: float, fid: float) -> float
def acc_from_loss(L: float) -> float
def G(fid: float) -> float
def optimal_alloc(C: float, fid: float) -> tuple[float, float]   # (N*, D*)
def optimal_loss(C: float, fid: float) -> float
def optimal_accuracy(C: float, fid: float) -> float
def flops_for_accuracy(R: float, fid: float) -> float             # inverse, closed form
def asymptotic_accuracy(fid: float) -> float
```

### 4.2 Definitions

Quality is defined as $\text{Quality} = 1/\text{FID}$, so $\text{Quality}^{-1} = \text{FID}$ and every $\log(1+\text{Quality}^{-1})$ becomes $\log(1+\text{FID})$. At FID $=0$ these collapse to the base constants.

$$E' = \exp\!\big(\log E + \log(1+\text{FID})\cdot\epsilon\big), \qquad B' = \exp\!\big(\log B + \log(1+\text{FID})\cdot\zeta\big)$$

$$L(N, D, \text{FID}) = \frac{A}{N^{\alpha}} + \frac{B'}{D^{\beta}} + E'$$

$$\mathrm{Acc}(L) = 1.2575 - 0.7496\,L$$

Compute-optimal allocation under $C = K_{\text{adv}} N D$ (their Eq. 4):

$$G = \left(\frac{\alpha A}{\beta B'}\right)^{\frac{1}{\alpha+\beta}}, \qquad N^{*} = G\left(\frac{C}{K_{\text{adv}}}\right)^{a}, \qquad D^{*} = G^{-1}\left(\frac{C}{K_{\text{adv}}}\right)^{b}$$

Substituting back, both terms acquire the same exponent $-\kappa$:

$$L^{*}(C) = E' + \underbrace{\left(A\,G^{-\alpha} + B'\,G^{\beta}\right)}_{\textstyle \mathcal{K}(\text{FID})}\left(\frac{C}{K_{\text{adv}}}\right)^{-\kappa}$$

Inverting for a target AutoAttack accuracy $R$:

$$C_{\text{def}}(R, \text{FID}) = K_{\text{adv}}\left[\frac{\mathcal{K}(\text{FID})}{\dfrac{1.2575 - R}{0.7496} - E'}\right]^{1/\kappa}$$

Asymptote: $\mathrm{Acc}_{\infty}(\text{FID}) = 1.2575 - 0.7496\,E'(\text{FID})$.

`flops_for_accuracy` must raise `ValueError` when $R \ge \mathrm{Acc}_{\infty}(\text{FID})$ — the target is unreachable at any compute. Do not clamp, do not return `inf` silently.

### 4.3 Acceptance tests (all must pass to 2 decimal places)

These reproduce the paper's own published values and are the only validation you get.

| Test | Input | Expected | Source |
|------|-------|----------|--------|
| T1.1 | `E_prime(1.76)` | 0.56 | Table 3, Approach 2, $E'$ column |
| T1.2 | `optimal_loss(1e25, fid=1.76)` | 0.61 | Table 3, $L^*$ at $10^{25}$ FLOPs |
| T1.3 | `optimal_loss(1e25, fid=0.0)` | 0.53 | Table 3, hypothetical FID=0 |
| T1.4 | `a`, `b` | 0.49, 0.51 (paper rounds to 0.48, 0.52) | Table 3 |
| T1.5 | `asymptotic_accuracy(0.0)` | 0.898 | Figure 1, ~90% asymptote |
| T1.6 | `asymptotic_accuracy(1.65)` | 0.837 | Figure 1, FID=1.65 band |
| T1.7 | `optimal_accuracy(1e21, fid=0.0)` | 0.778 | Consistent with Fig. 1 purple curve above green at $10^{21}$ |
| T1.8 | `flops_for_accuracy(0.7371, 0.0)` | 8.3e19 (order $10^{20}$) | — |
| T1.9 | `flops_for_accuracy(0.898, 0.0)` | raises `ValueError` | — |

If T1.2 and T1.3 fail, your `G` or your $\mathcal{K}$ is wrong. Re-derive rather than tuning.

### 4.4 Reference table to emit

```
R      C_def(R, FID=0)   [FLOPs]
0.737  8.3e19
0.760  3.1e20
0.780  1.2e21
0.800  5.7e21
0.820  4.0e22
0.840  5.1e23
0.850  2.6e24
0.860  1.9e25
0.880  1.2e28
0.890  1.4e31
0.898  divergent
```

Regenerate these from your implementation; the table above is a sanity target, not ground truth.

---

## 5. M2 — `flops.py` (attacker side)

### 5.1 The problem with heuristics

The Kaplan $6ND$ heuristic was calibrated for transformers where parameters dominate. It badly underestimates convolutional nets on 32×32 inputs, because convolution weights are reused across spatial positions. Bartoldson's empirical constant implies

$$\text{FLOPs per forward pass} \approx \frac{K_{\text{adv}}}{27}\,N \approx 289.7\,N$$

which is ~145× the $2ND$ heuristic. **Do not mix conventions.** The defender curve uses $K_{\text{adv}}$; the attacker side must use *measured* counts, with the ratio-derived constants as a cross-check only.

### 5.2 Contract

```python
def measure_forward_flops(model: nn.Module, input_shape=(1,3,32,32)) -> int
def measure_train_step_flops(model: nn.Module, batch_size: int) -> int
def attack_cost(victim, student, n_queries: int, n_epochs: int,
                queries_per_sample: int = 1) -> dict
```

`measure_forward_flops` must use `torch.utils.flop_counter.FlopCounterMode` under `torch.no_grad()`. `measure_train_step_flops` measures a real forward + backward + optimiser step.

`attack_cost` returns:

```python
{
  "c_query":    F_fwd_victim * n_queries * queries_per_sample,
  "c_student":  F_train_student_per_sample * n_queries * n_epochs,
  "c_total":    c_query + c_student,
  "c_query_heuristic":   289.7 * N_victim * n_queries * queries_per_sample,
  "c_student_heuristic": 869.1 * N_student * n_queries * n_epochs,
  "method": "flop_counter",
}
```

where $869.1 = K_{\text{adv}} \cdot (3/27)$ is the standard-training analogue of the paper's constant.

### 5.3 Notes to record in the paper

- The query term is **not** negligible against a 366M-parameter victim. At $Q = 2\times10^5$ it lands near $2\times10^{16}$, roughly 10% of student training cost. State this rather than assuming it away.
- Report the measured/heuristic ratio for at least one model. If it exceeds 2×, investigate before publishing any FLOPs figure.
- Attacks with multi-step query synthesis (JBDA, BEST) pay `queries_per_sample > 1` on the *student* for gradient steps, not the victim. Count these separately as `c_synthesis`.

---

## 6. M3 — `victims.py`

### 6.1 Model set

Load via `robustbench.utils.load_model(model_name=..., dataset='cifar10', threat_model='Linf')`. Verified keys:

| Key | Arch | ~AA | Role |
|-----|------|-----|------|
| `Standard` | WRN-28-10 | 0.0 | non-robust control |
| `Carmon2019Unlabeled` | WRN-28-10 | ~59.5 | low |
| `Gowal2021Improving_28_10_ddpm_100m` | WRN-28-10 | ~63.4 | low-mid |
| `Wang2023Better_WRN-28-10` | WRN-28-10 | ~67.3 | mid |
| `Wang2023Better_WRN-70-16` | WRN-70-16 | ~70.7 | prior SOTA |
| `Peng2023Robust` | RaWRN-70-16 | ~71.1 | high |
| `Bartoldson2024Adversarial_WRN-82-8` | WRN-82-8 | ~70.8 | compute-optimal |
| `Bartoldson2024Adversarial_WRN-94-16` | WRN-94-16 | ~73.7 | current SOTA |

**Verify every AA number yourself** by re-running AutoAttack (§8). The values above are approximate leaderboard figures used for planning; the paper must report your measurements.

The Bartoldson entries are `DMWideResNet` with `activation_fn=nn.SiLU` and built-in CIFAR-10 mean/std normalisation.

### 6.2 Critical: normalisation

RobustBench models carry their own input normalisation. Feed them **raw `[0,1]` tensors**. Do not apply `transforms.Normalize`. Double normalisation will silently drop AA accuracy by tens of points and you will misattribute it to something interesting.

Add an assertion: for `Bartoldson2024Adversarial_WRN-94-16`, clean accuracy on the full CIFAR-10 test set must exceed 92%. Abort the run if not.

### 6.3 Contract

```python
@dataclass
class Victim:
    name: str
    model: nn.Module
    n_params: int
    fwd_flops: int
    clean_acc: float
    aa_acc: float

def load_victim(name: str, device='cuda') -> Victim
def query(victim: Victim, x: Tensor, defense: Defense | None = None) -> Tensor
```

`query` returns **softmax posteriors** by default, applies `defense` if given, and increments a per-victim query counter that is written to results. The counter is the ground truth for the query budget — never trust the nominal `Q`.

---

## 7. M4 — `students.py`

Three architectures. The primary is fixed across all conditions.

| Name | Params | Role |
|------|--------|------|
| `resnet18_cifar` | ~11.2M | **primary** — used in every run |
| `wrn_16_4` | ~2.7M | capacity ablation |
| `wrn_28_10` | ~36.5M | capacity ablation |

`resnet18_cifar` is the CIFAR variant: 3×3 stem, stride 1, no max-pool. Do not use torchvision's ImageNet ResNet-18.

**Initialisation must be identical across victim conditions for a given seed.** Implement:

```python
def build_student(arch: str, seed: int) -> nn.Module:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    ...
```

and assert that `build_student('resnet18_cifar', 0)` produces bit-identical parameters on two calls.

---

## 8. M5 — `extract.py`

### 8.1 Query pools

| Pool ID | Source | Note |
|---------|--------|------|
| `cifar100` | CIFAR-100 train, 50k | **primary** — out of distribution, realistic attacker |
| `tinyimagenet` | downsampled to 32×32, 100k | secondary |
| `cifar10_train` | CIFAR-10 train images, labels discarded | **upper bound only** — report separately, never as the headline |
| `uniform_noise` | U[0,1] | lower-bound control |

The `cifar10_train` condition inflates $\tau$ because the attacker gets the victim's own training distribution for free. Include it, label it clearly as an upper bound, and do not average it with the others.

### 8.2 Attacks

All attacks are black-box: the attacker sees only the output of `query()`.

**A1 — `random` (KnockoffNets-style).** Sample $Q$ images uniformly from the pool, query once each, train the student on the soft labels with KL divergence. Baseline.

**A2 — `jbda` (Jacobian-based augmentation).** Seed with $Q_0 = Q/2^{\rho}$ pool images. For $\rho$ rounds: query the current set, train the student, then augment each $x$ with $x + \lambda\,\mathrm{sign}(\nabla_x \mathcal{L}_{\text{student}}(x, \hat{y}))$, $\lambda = 0.1$. Stop when the budget is exhausted. Use $\rho = 5$.

**A3 — `best` (Boundary Entropy Searching Thief).** The robustness-aware attack. For each seed image $x_0$ from the pool, produce an *uncertain example* by ascending the student's prediction entropy:

```
x = x_0
for t in range(T):                      # T = 10
    p = softmax(student(x))
    loss = KL(p || uniform)             # drive toward uniform confidence
    x = x - eta * sign(grad_x loss)
    x = clip(x, 0, 1)
```

Then query the victim at $x$ and add $(x, \text{victim}(x))$ to the transfer set. Interleave: alternate between generating a batch of uncertain examples and training the student on the accumulated set. Use $\eta = 2/255$, batch 256, 10 interleave rounds.

Gradients come from the **student**, never the victim — this is what makes it black-box. Count the $T$ student forward/backward passes in `c_synthesis`.

**A4 — `noise` control.** Uniform noise queries. Should produce near-zero fidelity; if it doesn't, your evaluation is broken.

### 8.3 Student training (identical across all conditions)

```
loss        : KL(student_log_softmax || victim_posterior), temperature 1.0
optimiser   : SGD, momentum 0.9, nesterov, weight_decay 5e-4
lr          : 0.1, cosine annealing to 0
epochs      : 100
batch_size  : 256
augment     : random crop (pad 4) + horizontal flip
```

**The student is trained with standard (non-adversarial) training.** This is the point of the experiment: the attacker never pays the 9× adversarial-training premium. Any adversarial training on the student side invalidates the cost argument.

### 8.4 Contract

```python
def extract(victim: Victim, student_arch: str, attack: str, pool: str,
            n_queries: int, seed: int, defense: Defense | None = None,
            epochs: int = 100) -> ExtractionResult
```

---

## 9. M6 — `evaluate.py`

### 9.1 Metrics

| Metric | Definition |
|--------|------------|
| `clean_acc` | student top-1 on CIFAR-10 test (10,000) |
| `aa_acc` | student accuracy under AutoAttack, $\ell_\infty$, $\varepsilon = 8/255$ |
| `fidelity_clean` | fraction of test set where `argmax student == argmax victim` |
| `fidelity_adv` | same, on victim-generated adversarial examples |
| `tau` | `aa_acc(student) / aa_acc(victim)` |
| `delta_abs` | `aa_acc(student) - aa_acc(student_from_Standard)` at matched $Q$, attack, pool, seed |

### 9.2 AutoAttack configuration

```python
from autoattack import AutoAttack
adv = AutoAttack(model_fn, norm='Linf', eps=8/255, version='standard')
```

- Development / sweeps: first 1,000 test images. Record `n_eval=1000`.
- Final reported numbers: full 10,000 test images. Record `n_eval=10000`.
- Never mix the two in one figure. Add `n_eval` to every result record and assert consistency at plot time.

Students trained on soft labels can have low-confidence, near-uniform outputs on some inputs. Verify the student is not gradient-masking: if `aa_acc` exceeds PGD-40 accuracy, something is wrong — AutoAttack should always be at least as strong. Assert this.

### 9.3 The $\tau$ denominator

Victim `Standard` has $\mathrm{AA} = 0$, so $\tau$ is undefined. Handle explicitly: exclude `Standard` from $\tau$ plots and use it only for `delta_abs` and as the control baseline.

### 9.4 Required control — temperature matching

**This is the single most important control in the paper.** Without it a reviewer will argue you measured label smoothing.

Robust models produce higher-entropy posteriors than standard models. That alone could make distillation easier. To separate the effects:

1. Measure mean predictive entropy $H_v$ of each robust victim on the query pool.
2. For the `Standard` victim, find temperature $T$ such that `softmax(logits/T)` has mean entropy $H_v$.
3. Run the full extraction against this temperature-scaled `Standard` victim.
4. Compare student AA accuracy.

If the temperature-matched control reaches comparable AA accuracy, the effect is entropy, not robustness — report that (falsification F2) and reframe the paper. Log these runs with `victim_name="Standard_T{T}"` and `is_control=true`.

---

## 10. M7 — `margins.py` (E3, the mechanism)

### 10.1 Purpose

Establish that the irreducible error the source paper proves is unavoidable *is itself* the attacker's supervision signal. Without this, the paper is a rerun of prior work.

### 10.2 Method

For each victim, compute the minimal $\ell_\infty$ perturbation to flip the prediction on each of the first 2,000 test images, using FAB-T:

```python
adv = AutoAttack(model_fn, norm='Linf', eps=8/255, version='custom',
                 attacks_to_run=['fab-t'])
adv.fab.n_restarts = 3
adv.fab.n_iter = 100
```

FAB returns the perturbed point; the margin is $\|x_{\text{adv}} - x\|_\infty$. Where FAB fails to flip within its budget, record the margin as right-censored at the search bound and use a survival estimator, not a plain mean.

### 10.3 Outputs per victim

- Margin distribution (write all 2,000 values to `margins.jsonl`)
- Median margin, IQR
- `frac_margin_le_8_255` — should approximately equal $1 - \mathrm{AA}$; assert agreement within 3 points as a validation of both pipelines
- Estimated near-boundary density $\rho_{8/255}$

### 10.4 The link to be established

Define **sample efficiency** $Q_{90}(v)$ = the query budget at which `fidelity_clean` first reaches 0.90 for victim $v$ (interpolate between measured budgets; report as censored if never reached).

Then regress $\log Q_{90}$ on `frac_margin_le_8_255` across victims. Report Spearman $\rho$ and a bootstrap CI. With only 7–8 victims this is underpowered — state that plainly and treat it as suggestive, not confirmatory. Do not report a p-value as if $n=8$ supported one.

---

## 11. M8 — `defenses.py` (E4)

### 11.1 Interface

```python
class Defense(Protocol):
    name: str
    def __call__(self, posteriors: Tensor) -> Tensor
    def utility_cost(self, victim: Victim) -> float   # drop in victim clean acc
```

### 11.2 Defenses to implement

| ID | Description | Params |
|----|-------------|--------|
| `none` | identity | — |
| `top1` | one-hot the argmax | — |
| `round_k` | round posteriors to $k$ decimals, renormalise | $k \in \{1,2,3\}$ |
| `topk` | keep top-$k$ probabilities, zero rest, renormalise | $k \in \{2,3\}$ |
| `mad` | prediction poisoning: perturb posterior within an $\ell_1$ ball to maximise angular deviation of the attacker's gradient estimate, subject to preserving argmax | $\epsilon_{\text{mad}} \in \{0.5, 0.99\}$ |
| `prada` | detection, not perturbation: KS test on the pairwise-distance distribution of a user's query stream; flag when $p < 0.05$ | window 100 |

`prada` does not modify outputs — it returns `posteriors` unchanged and records the query index at which detection fires. That index is the defense's effective budget cap.

### 11.3 Watermarking

Backdoor watermarking requires modifying the victim, which you cannot do to RobustBench checkpoints. Train **one** WRN-28-10 with TRADES ($\beta = 5$, PGD-10, $\varepsilon = 8/255$, 100 epochs) plus a trigger-pattern backdoor on 1% of training data. This is the only adversarial training run in the project (~8 GPU-hours). Measure whether the watermark survives extraction (student accuracy on triggers vs chance).

### 11.4 The reported quantity

For each defense $d$:

$$\text{cost-effectiveness}(d) = \frac{\log_{10}\big(C_{\text{att}}(d) / C_{\text{att}}(\texttt{none})\big)}{\Delta \text{clean acc of victim}}$$

i.e. orders of magnitude of attacker cost added per point of defender utility sacrificed. $C_{\text{att}}(d)$ is the attacker cost required to reach the *same* student AA accuracy as achieved with `none`; if unreachable within 10× the largest budget, record as censored and report a lower bound.

---

## 12. M9 — result schema

`results.jsonl`, one JSON object per line. Every field required; use `null` only where the schema says nullable.

```json
{
  "run_id": "uuid4",
  "timestamp": "ISO-8601",
  "git_sha": "abc123",
  "victim_name": "Bartoldson2024Adversarial_WRN-94-16",
  "victim_params": 366000000,
  "victim_clean_acc": 0.9378,
  "victim_aa_acc": 0.7371,
  "is_control": false,
  "student_arch": "resnet18_cifar",
  "student_params": 11173962,
  "attack": "best",
  "pool": "cifar100",
  "defense": "none",
  "defense_params": {},
  "n_queries_nominal": 50000,
  "n_queries_actual": 50000,
  "epochs": 100,
  "seed": 0,
  "student_clean_acc": 0.0,
  "student_aa_acc": 0.0,
  "student_pgd40_acc": 0.0,
  "fidelity_clean": 0.0,
  "fidelity_adv": 0.0,
  "tau": null,
  "n_eval": 10000,
  "c_query": 0,
  "c_synthesis": 0,
  "c_student": 0,
  "c_total": 0,
  "c_query_heuristic": 0,
  "c_student_heuristic": 0,
  "wallclock_sec": 0.0,
  "gpu": "A100-40GB"
}
```

Writes must be append-only and flushed per record so a crashed sweep loses at most one run.

---

## 13. M10 — experiment grids

### 13.1 E2 — main sweep

```
victims  : all 8 from §6.1
attacks  : {random, best}
pool     : cifar100
budgets  : {5_000, 20_000, 50_000, 200_000}
student  : resnet18_cifar
seeds    : {0, 1, 2}
defense  : none
```

$8 \times 2 \times 4 \times 3 = 192$ runs. Use `n_eval=1000` for the sweep; re-evaluate the $Q = 200{,}000$ cell at `n_eval=10000` for the paper (48 runs).

### 13.2 E2b — ablations (at $Q = 50{,}000$, seed 0 only)

```
attack ablation  : {jbda, noise}          x all victims
pool ablation    : {tinyimagenet, cifar10_train, uniform_noise}  x {Standard, Wang2023Better_WRN-70-16, Bartoldson2024Adversarial_WRN-94-16}
capacity ablation: {wrn_16_4, wrn_28_10}  x {Standard, Bartoldson2024Adversarial_WRN-94-16}
```

### 13.3 E2c — temperature-matched control (§9.4)

One control run per robust victim, at $Q = 50{,}000$, attack `best`, 3 seeds. 21 runs.

### 13.4 E3 — margins

All 8 victims, 2,000 images each. No sweep.

### 13.5 E4 — defenses

```
victims  : {Wang2023Better_WRN-70-16, Bartoldson2024Adversarial_WRN-94-16}
defenses : {none, top1, round_1, round_2, topk_2, mad_0.5, mad_0.99, prada}
attack   : best
budget   : 50_000
seeds    : {0, 1, 2}
```

Plus the watermarked WRN-28-10 across `{none, top1}`, 3 seeds.

---

## 14. M11 — `analysis.py` (E5)

### 14.1 The $\tau$ regression

Fit $\tau = m R_v + c$ by weighted least squares, weights $= 1/\mathrm{Var}(\tau)$ across seeds. Report $m$ with a 10,000-sample bootstrap 95% CI (resample victims, not seeds — victims are the unit of generalisation).

Fit separately per attack and per budget. Do not pool.

### 14.2 Leverage

For a target robustness $R$, the attacker must extract from a victim at $R_v$ such that $\tau(R_v)\,R_v = R$, i.e. solve for $R_v$. Then:

$$\text{leverage}(R) = \frac{C_{\text{def}}(R_v,\ \text{FID}{=}0)}{C_{\text{att}}(R_v)}$$

Propagate the bootstrap CI on $\tau$ through to a CI on leverage.

### 14.3 Extrapolation honesty

Your measured $R_v$ range is roughly $[0.595, 0.737]$. The interesting claims live at $[0.80, 0.89]$. **Every extrapolated number must be shaded and labelled as extrapolation in the figure, and the caption must state the measured range.** Provide the leverage table at both:

- $R = 0.737$ (interpolated, defensible)
- $R \in \{0.80, 0.85, 0.88\}$ (extrapolated, CI-shaded)

A reviewer will attack this. Pre-empt it in the limitations section rather than letting them find it.

---

## 15. Environment

```
python >= 3.10
torch >= 2.1            # torch.utils.flop_counter requires 2.0+
torchvision
robustbench @ git+https://github.com/RobustBench/robustbench@v1.1
autoattack @ git+https://github.com/fra31/auto-attack
numpy, scipy, pandas, matplotlib
```

Determinism:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

Record `torch.__version__`, CUDA version, and GPU model in `REPRO.md`. Exact bit-reproducibility across GPUs is not expected; seed-level reproducibility on the same hardware is required.

---

## 16. Runtime budget

| Stage | Runs | Per run | Total (A100) |
|-------|------|---------|--------------|
| M1 cost model | — | — | seconds |
| Victim AA verification | 8 | 20–90 min | ~6 h |
| E2 main sweep (n_eval=1000) | 192 | ~50 min | ~160 h |
| E2 final eval (n_eval=10000) | 48 | ~40 min extra | ~32 h |
| E2b ablations | ~40 | ~50 min | ~33 h |
| E2c control | 21 | ~50 min | ~18 h |
| E3 margins | 8 | ~40 min | ~5 h |
| E4 defenses | 54 | ~50 min | ~45 h |
| Watermark victim training | 1 | ~8 h | 8 h |

Roughly **300 GPU-hours**, ~13 days on one A100 or under 2 days on 8. Cut the $Q=5{,}000$ cell and the `jbda` ablation first if constrained.

Note for the paper: the entire attack-side study costs on the order of $10^{18}$ FLOPs against a victim that cost over $10^{21}$ to train. That ratio is itself a result — state it.

---

## 17. Pitfalls (read before writing code)

1. **Double normalisation.** §6.2. The most common and most silent failure.
2. **Student init drift.** If the student's random init differs across victims, $\tau$ differences are noise. Assert bit-identical init per seed.
3. **Mixing `n_eval`.** A 1,000-image AA estimate has ~1.5 points of standard error. Never plot it against a 10,000-image number.
4. **Gradient masking on the student.** Distilled students can look robust without being robust. Always report PGD-40 alongside AA and assert AA ≤ PGD-40.
5. **Query counting.** Count actual `query()` invocations, not the nominal budget. Attacks with rejection or filtering will diverge.
6. **FLOPs convention mixing.** §5.1. Defender curve uses $K_{\text{adv}}$; attacker uses measured counts. Never present a comparison built from two conventions without saying so.
7. **`cifar10_train` pool leakage.** It is an upper bound, not the headline result.
8. **Over-claiming the source paper.** Bartoldson et al. argue robustness is *closer* to human level than the leaderboard suggests, because ~10% of the benchmark's adversarial images are invalid (they no longer depict their label). Their corrected figures are 79.49% for the SOTA model on valid data against a 96.46% human ceiling. Cite the corrected numbers and re-derive the defender curve against them as a robustness check. The argument survives — the divergence just shifts right — and this pre-empts the most informed possible reviewer.
9. **$n=8$ victims.** Every cross-victim regression is underpowered. Report effect sizes and CIs; do not lean on significance tests.
10. **Prior work.** Khaled & Cheriet (arXiv 2207.10561) and Yue et al. (ICLR 2023, BEST) already show robust models are more extractable and that robustness transfers. Your contribution is the *calibrated cost model and its divergence*, not the transfer finding. Position accordingly in the introduction or the paper will read as a replication.

---

## 18. Suggested repo layout

```
robustness-economics/
├── README.md
├── REPRO.md
├── pyproject.toml
├── configs/
│   ├── e2_main.yaml
│   ├── e2b_ablations.yaml
│   ├── e2c_control.yaml
│   ├── e3_margins.yaml
│   └── e4_defenses.yaml
├── src/
│   ├── costmodel.py        # M1
│   ├── flops.py            # M2
│   ├── victims.py          # M3
│   ├── students.py         # M4
│   ├── extract.py          # M5
│   ├── evaluate.py         # M6
│   ├── margins.py          # M7
│   ├── defenses.py         # M8
│   ├── schema.py           # M9
│   ├── analysis.py         # M11
│   └── figures.py
├── scripts/
│   ├── run_sweep.py
│   ├── verify_victims.py
│   └── make_figures.py
├── tests/
│   ├── test_costmodel.py   # T1.1 - T1.9, must pass first
│   ├── test_flops.py
│   └── test_determinism.py
└── results/
    ├── results.jsonl
    ├── margins.jsonl
    └── defenses.jsonl
```

Implement and pass `tests/test_costmodel.py` before touching a GPU.