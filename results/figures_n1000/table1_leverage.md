**Table 1.** Leverage = $C_\mathrm{def}(R)/C_\mathrm{att}(R)$ for attack `best`, budget 200000: FLOPs to *train* a model at AutoAttack accuracy $R$ divided by FLOPs to *extract* one from a victim at $R$. Measured victim AutoAttack range is [0.596, 0.737]. Targets outside this range are extrapolations of the fitted tau(R_v) and are shaded in every figure (spec.md S14.3). Fitted $\tau = +0.338 R_v -0.030$ (95% CI on the slope [-0.638, +0.727], 7 victims). The last column is the spec.md §14.2 reading, in which $R$ is the robustness the attacker ends up with and the required victim solves $\tau(R_v)R_v = R$.

| target R | $C_{def}$ [FLOPs] | $C_{att}$ [FLOPs] | leverage | log10 leverage 95% CI | student AA obtained | status | §14.2 matched $R_v$ |
|---|---|---|---|---|---|---|---|
| 0.737 | 8.25e+19 | 1.09e+17 | 761 | [2.85, 2.92] | 0.161 | **extrapolated** | unreachable |
| 0.800 | 5.71e+21 | 1.18e+17 | 4.84e+04 | [4.61, 4.75] | 0.192 | **extrapolated** | unreachable |
| 0.850 | 2.56e+24 | 1.26e+17 | 2.03e+07 | [7.20, 7.40] | 0.218 | **extrapolated** | unreachable |
| 0.880 | 1.19e+28 | 1.31e+17 | 9.05e+10 | [10.83, 11.06] | 0.235 | **extrapolated** | unreachable |