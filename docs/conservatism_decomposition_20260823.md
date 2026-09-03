# SC-PCP 保守性分解审计（2026-08-23）

本文档冻结并解释 standard synthetic setting 上的旧 SC-PCP 保守性诊断。该诊断使用 100 个配对 seeds（`0..99`）；每个 seed 对所有冻结 schedules 使用同一条独立 common-random-number（CRN）流，并分别运行 50,000 条 fresh target-policy trajectories。所有层的 selection target 均为

\[
1-\alpha=0.90.
\]

这里的“同 target”指相同 coverage level、模拟器、模型和随机数耦合，不指相同 deployment distribution。不同半径会改变策略、动作和后续状态，因此各层评价的是各自诱导的 \(P_{q^{(L)}}\)。

## 五层定义

为了与原始 A--E 设计一致，五层定义如下。

| Layer | 冻结定义 |
|---|---|
| A | **Greedy Sequential Oracle**：使用自由的 stagewise grids，在 oracle rollout surface 上逐阶段选择并提交半径。它是 greedy sequential reference，不是已证明的连续空间全局最优解。 |
| B | **Current Profiled Oracle**：限制为旧 profile family \(q_t=s b_t\)，使用 Phase 0 oracle coverage 和 oracle width surface 选择满足所有阶段 target 的最窄候选。 |
| C | **Profile + exact/oracle point selection**：在与旧 paper pipeline 对齐的 profile、scale grid 和 candidate schedules 上，使用 oracle point coverage 选择。 |
| D | **Profile + learned-COT point selection**：保持同一 candidate family，使用 learned-COT Hájek point coverage 和 learned-COT estimated width，不使用 lower bound。 |
| E | **Old practical SC-PCP Bootstrap/LCB**：保持同一 candidate family、learned-COT weights、estimated width 和 ordered selector，仅将 point coverage surface 换成 patient-cluster bootstrap lower-bound surface。 |

本协议中 **B 与 C 是定义上合并的桥接层**。运行器要求 Phase 0 与旧 paper 输入的 scale grid、stage profile 和 candidate schedules bitwise identical，并要求 Phase 0 保存的 profiled schedule 与恢复出的 C schedule 完全一致。因此

\[
B\equiv C,\qquad W_C/W_B=1,\qquad \log(W_C/W_B)=0.
\]

artifact 为避免复制同一 schedule，仅保存 `A/C/D/E` 四行；其中 `C` 同时代表上述 B 和 C。这个合并不是实验后观察到“接近零”，而是恢复协议强制的 identity。

## 配对 width 分解

每个 seed 内的 width 是 fresh-rollout micro average normalized width。跨 seed 报告 geometric mean；ratio 在 paired log-width 上计算。95% CI 使用 10,000 次 seed-cluster percentile bootstrap，固定 RNG seed `271828`。

| Comparison | Geometric width ratio | 95% CI | Mean log overhead |
|---|---:|---:|---:|
| B/A = C/A | 1.035498 | [1.033110, 1.037960] | 0.034882546 |
| C/B | 1.000000 | [1.000000, 1.000000] | 0 |
| D/C | 1.005132 | [1.002045, 1.008321] | 0.005118946 |
| E/D | 1.025412 | [1.024333, 1.026504] | 0.025094658 |
| E/A | 1.067262 | [1.064164, 1.070483] | 0.065096150 |

各层 geometric mean width 为：

| Layer | Width | 95% CI |
|---|---:|---:|
| A | 1.832800 | [1.829747, 1.835831] |
| B = C | 1.897861 | [1.892559, 1.903299] |
| D | 1.907601 | [1.901619, 1.913756] |
| E | 1.956077 | [1.949417, 1.962849] |

在 log-width 尺度上，分解精确 telescopes：

\[
\underbrace{0.034882546}_{A\to B\ \text{(profiled bridge)}}
+\underbrace{0}_{B\to C\ \text{(identity)}}
+\underbrace{0.005118946}_{C\to D\ \text{(COT-point bridge)}}
+\underbrace{0.025094658}_{D\to E\ \text{(LCB guard)}}
=\underbrace{0.065096150}_{A\to E}.
\]

数值闭合误差为 \(2.78\times10^{-16}\)。相对于总 log overhead，三个非零项的 accounting shares 分别为：A→B 53.59%，C→D 7.86%，D→E 38.55%。这些是 realized-pipeline accounting shares，不是三个机制的纯因果贡献。

## Fresh coverage

主 coverage 口径为

\[
\operatorname{WSC}=\min_t\frac{1}{S}\sum_{s=1}^{S}\widehat C_{s,t},
\]

而 MeanCov 是 seed 与 stage 的整体平均。区间同样使用上述 paired seed bootstrap。

| Layer | WSC | WSC 95% CI | MeanCov | MeanCov 95% CI | Seeds with \(\min_t\widehat C_{s,t}\ge0.90\) |
|---|---:|---:|---:|---:|---:|
| A | 0.901618 | [0.900518, 0.901945] | 0.902657 | [0.902364, 0.902950] | 1/100 |
| B = C | 0.915788 | [0.913471, 0.916045] | 0.916468 | [0.915540, 0.917424] | 65/100 |
| D | 0.917698 | [0.915320, 0.918163] | 0.918410 | [0.917388, 0.919462] | 72/100 |
| E | 0.926971 | [0.924736, 0.927407] | 0.927616 | [0.926644, 0.928619] | 100/100 |

最后一列不是本文的 marginal WSC estimand。它要求单个 seed 的 12 个 Monte Carlo stage estimates 全部不低于 0.90，因此不能用 A 的 `1/100` 推断 marginal target 失败；A 的 WSC 及其 bootstrap lower endpoint 均高于 0.90。

## 可以和不可以怎样解释

### A→B（artifact 中为 A→C）不是纯 profile cost

A 与 B 不只改变 profile constraint。A 使用自由 stage grids 和 greedy committed search；B 使用 profiled scale grid 并在完整 profile family 中做 joint feasible minimum-width selection。因此 A→B 同时包含：

1. profile restriction；
2. stage-grid 与 profiled-grid construction 的差异；
3. greedy sequential search 与 family-wide selection 的差异。

所以 `C/A = 1.035498` 应称为“free greedy reference 到 profiled oracle 的 realized width ratio”，不能称为 profile restriction 的纯因果代价。

### C→D 不是纯 transport-estimation cost

C→D 同时将 oracle coverage 换成 learned-COT point coverage、oracle rollout width 换成 learned-COT estimated width，并将 all-feasible minimum-width selection 换成 widest-to-narrowest contiguous ordered-prefix selection。因此该项混合了 coverage transport estimation、width estimation 与 search-rule restriction。`D/C = 1.005132` 只说明这组联合替换的净 width overhead 很小。

### D→E 是最干净的相邻比较，但 `formal=false`

D 与 E 共用 candidate family、learned-COT weights、estimated width 和 ordered selector；主要变化是 point surface 到 bootstrap lower-bound surface。因此 `E/D = 1.025412` 是最可信的旧 statistical-guard conservatism 估计。

不过旧 E 使用 `ratio_bound_source: none`，bootstrap 将 fitted、capped COT weights 当作固定，没有覆盖 COT nuisance estimation error、model misspecification 或 clipping bias。因此它不是完整的 finite-sample transport certificate；该层的正式证书状态应记录为 **`formal=false`**。此外，E 相对 D 的 WSC 同时提高约 0.00927，所以 2.54% width 增量伴随明显 overcoverage，不能解释为等 coverage 下的纯效率损失。

## 与当前主方法的边界

本分解诊断的是已经退役的 `profile + learned COT + bootstrap/LCB + ordered-IUT` 路径。当前仓库唯一的 paper method 是 committed-prefix marginal SC-PCP：它使用自由 stagewise radii、结构性单步 ratio-capped target policy 下不截断的累计 prefix importance product，以及 Hájek point calibration；不再使用 fixed profile、learned COT coverage surface、coverage LCB 或 ordered-IUT。

因此本文档可以支持如下开发结论：旧 LCB guard 是显著保守来源，且旧 learned-COT point bridge 的净 width overhead 较小，这促使方法改为直接 target-policy marginal calibration。不能把上述 53.59%/7.86%/38.55% 当作当前 marginal SC-PCP 的内部成分分解。

当前方法定义见 [`docs/final_method.md`](final_method.md) 和 [`src/scpcp/marginal_prefix.py`](../src/scpcp/marginal_prefix.py)。

## Artifact 与复现边界

- Artifact root（相对）：[`results/work/conservatism_decomposition_standard_fresh_20260821`](../results/work/conservatism_decomposition_standard_fresh_20260821)
- Artifact root（绝对）：[`/home/ubuntu/zmh/sc-pcp/results/work/conservatism_decomposition_standard_fresh_20260821`](/home/ubuntu/zmh/sc-pcp/results/work/conservatism_decomposition_standard_fresh_20260821)
- PDF（相对）：[`conservatism_decomposition.pdf`](../results/work/conservatism_decomposition_standard_fresh_20260821/conservatism_decomposition.pdf)
- PDF（绝对）：[`/home/ubuntu/zmh/sc-pcp/results/work/conservatism_decomposition_standard_fresh_20260821/conservatism_decomposition.pdf`](/home/ubuntu/zmh/sc-pcp/results/work/conservatism_decomposition_standard_fresh_20260821/conservatism_decomposition.pdf)
- Frozen config：[`config.yaml`](../results/work/conservatism_decomposition_standard_fresh_20260821/config.yaml)
- Study provenance：[`study_metadata.json`](../results/work/conservatism_decomposition_standard_fresh_20260821/study_metadata.json)
- Completion manifest：[`study_status.json`](../results/work/conservatism_decomposition_standard_fresh_20260821/study_status.json)
- Recovery implementation：[`src/scpcp/conservatism_decomposition.py`](../src/scpcp/conservatism_decomposition.py)
- Fresh common-CRN runner：[`scripts/run_conservatism_decomposition.py`](../scripts/run_conservatism_decomposition.py)
- Paired summary implementation：[`scripts/summarize_conservatism_decomposition.py`](../scripts/summarize_conservatism_decomposition.py)

全部 100 个 output seed artifacts 仍可通过现有 validator，且 A/C 对现存 Phase 0 输入的 schedule mismatch 为 0、bitwise replay error 为 0。旧 paper input directory 已清理，而 decomposition artifacts 没有复制 D/E 的完整 `cot_diagonal`、`cot_lower_bounds` 和 candidate-width surfaces；因此目前可以审计冻结结果与当时保存的 fingerprints，但不能仅凭当前工作区从头重建旧 D/E selection。
