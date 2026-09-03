# 三项正式实验的完整记录（2026-08-25）

本文档冻结并解释 2026-08-25 完成的三项正式研究：500-instance exact
finite-MDP 结构诊断、controlled semi-synthetic 六方法比较，以及 100-seed
equal-marginal copula 机制门。结果中既保留正结果，也保留预注册门未通过的
NO-GO；任何表格都不构成“在所有 setting 上普遍 SOTA”的声明。

2026-08-26 后续完成的 horizon×overlap、calibration-size、propensity 与 strict-split
正式诊断见 [`formal_experiments_20260826.md`](formal_experiments_20260826.md)。它们
扩展统计边界与 robustness 证据，不覆盖或重算本页三个 frozen studies。

SC-PCP 的唯一正式定义见 [`final_method.md`](final_method.md)。自然/production
paper suite 的 48 个唯一 setting--method cells（RQ1 30 行、RQ3 24 个展示行，
其中 \(\beta=1\) 的 6 行复用 RQ1 Synthetic）及完整 baseline 说明已经记录在
[`method_and_complete_baseline_results_20260824.md`](method_and_complete_baseline_results_20260824.md)，
这里不重复抄写，而只报告本轮三个彼此隔离的正式研究。

## 1. 总结先行

1. **完整 prefix 的 identification 得到 exact finite-MDP 支持。** 在 500 个成对
   MDP instances 中，full-prefix estimand 在 M0--M3 的 population identification
   bias 都为数值零；省略 history 或 current-action ratio 时，只在与该省略相容的
   特殊机制中无偏。这个结果验证的是 transport identity，不是 finite-sample
   coverage theorem。
2. **Controlled 六方法实验给出有方向性的、而非普遍支配的结果。** 在强负 shift
   下，SC-PCP 相对 Standard CP 的 WSC 提高 3.46 pp（\(\gamma=-4\)）和
   1.86 pp（\(\gamma=-2\)），但需要更宽的集合，而且两个点估计仍略低于 90%。
   在正 shift 下，Standard CP 的 WSC 反而高于 SC-PCP，而 SC-PCP 更窄。MFCS
   通常 coverage 更高但明显更宽；SPCI 通常更窄但 coverage 更低。不存在一个方法
   在所有 \(\gamma\) 上同时支配 coverage 和 width。
3. **Orthogonal copula 正式门是 NO-GO。** 因果方向、placebo、equal-marginal
   audit 和 overlap 全部通过，而且 Q90/coverage 的方向性 effect 的 95% paired CI
   排除了零；但 effect magnitude 只有 0.80--0.93% relative Q90 shift 和
   0.30--0.35 pp coverage shift，未达到预先冻结的 3% 和 1.5 pp 门槛。因此不得
   启动该 benchmark 的六方法比较，也不得把它写成主正结果。

## 2. Provenance 与不可混用边界

三项 formal artifacts 都绑定同一正式运行源树：

```text
source_tree_sha256 = 7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643
```

对应的只读源快照是
[`results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz`](../results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz)：

```text
sha256 = 2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b
size   = 2,036,776 bytes
scope  = configs/, pyproject.toml, scripts/, src/, tools/
```

正式运行完成后的测试/维护改动使本文起草时的活动源树 hash 变为
`e2233832cb2f52f3208be2b4b69e44fe042945da6c5b10fec8ae3fd76fcb121d`。
这不改变已经生成且由 hash 锁定的 formal artifacts；它意味着直接用当前源树对旧
目录执行 `--resume` 应当 fail closed。若要逐字节复现正式运行代码，应在新的隔离
目录中使用上述 tar snapshot，而不是覆盖当前工作区。

| Study | 完整 artifact | 关键文件 hash |
|---|---|---|
| Exact finite MDP | [`results/work/exact_finite_mdp_20260825`](../results/work/exact_finite_mdp_20260825) | `summary.json`: `9e3ab3f8f1fadd42685068f735ccd58c55e9be7179d9ed8863f4b83d80e647d6`; config: `51e08d3bff5686a12ac7b1625e73e8aebd927c991c07f3189ac4e3ad30a8cfda` |
| Controlled 六方法 | [`results/work/controlled_six_method_confirm20_20260825`](../results/work/controlled_six_method_confirm20_20260825) | `summary.json`: `d8533ca5db0c6a3943fed1751f4d450846dcbff17df305a33197a105cc474670`; active config: `a9023266d72b6aff04ab446a3236097bd24d10dc1f15b504aeb688c0bbbf9979` |
| Orthogonal copula | [`results/work/copula_mechanism_v1_20260825`](../results/work/copula_mechanism_v1_20260825) | `summary.json`: `53ad4938801effe5c9373f6096245c2633f9891b28cd22af63025dd959be3df0`; `gate.json`: `4f4378e47bc49f846651d24e598292cd3758f6039767e90e89deeb41458fbb7d`; config: `8ddaf311e54a3ef588b1b57f22a87f346ab42f33e86af875fd1a611eb4322cd3` |

每个 artifact 都有 complete marker 和内部 provenance。Exact MDP 和 copula
另外用 manifest 锁定 payload；controlled study 锁定 config、source、seed-to-device
映射、方法集合和 RNG stream contract，并对每个 seed artifact 逐一验证。三套正式
seed namespaces 互不重叠：finite MDP 为 52081、52100--52599 与
52600--52611；controlled 为 91000--91190（步长 10）；copula 为
94000--94198 的偶数。

## 3. 共同 estimand 与 claim boundary

Controlled study 的主 coverage 指标严格是

\[
\operatorname{WSC}
=\min_t\frac{1}{S}\sum_{s=1}^{S} C_{s,t},
\]

即 `min_t mean_seed(C_seed,t)`，不是 `mean_seed(min_t C_seed,t)`，也不是跨阶段
MeanCov。Width 是在目标策略下估计的 mean normalized prediction-box width。
WSC 的 95% 区间对完整 seed-stage vector 做 percentile bootstrap；width 区间是
selected seeds 上的 Student-\(t\) interval；SC-PCP 与 baseline 的差异使用相同 seed
的 paired percentile bootstrap。每个 \(\gamma\) 使用一套共享 bootstrap index
matrix，共 10,000 resamples。

Coverage 与 width 都是最终半径自身诱导的 target-policy trajectory law 下的
per-step marginal estimand。SC-PCP 的允许声明仍是 **plug-in、渐近的 per-step
marginal coverage**；不是 finite-sample、distribution-free、PAC、data-conditional
或 episode-wise simultaneous coverage。Exact MDP 是 diagnostic identification
study；copula 是 mechanism gate；两者都不产生 paper method row，也不改变唯一的
canonical SC-PCP selector。

## 4. Study A：500-instance exact finite MDP

### 4.1 Protocol

每个 instance 有 8 states、3 actions、4 stages 和每阶段 7 个 radius candidates，
因此完整枚举 \(7^4=2401\) 个 schedules。500 个 bounded-random instances 在四种
机制间使用 paired problem randomness：

| 机制 | 结构 |
|---|---|
| M0 no feedback | target policy 等于 behavior；全部 transport estimands 应重合 |
| M1 current only | radius 只改变当前动作对应的 score law |
| M2 history only | 历史动作改变 state occupancy；当前动作对 score 无直接作用 |
| M3 full feedback | 历史 occupancy 与当前动作都会改变 score law |

比较四个 population estimands：unweighted、history-only、current-only 与
full-prefix。表中数值是每个 instance 的整张 coverage surface 相对 exact target
surface 的 RMSE，再在 500 instances 上汇总；方括号是经验 5%--95% 分位数。

### 4.2 Population identification

| 机制 | Estimand | Mean RMSE [5%, 95%] | Mean max-absolute bias |
|---|---|---:|---:|
| M0 | Unweighted | \(7.80\times10^{-17}\) [\(4.20\times10^{-17}\), \(1.26\times10^{-16}\)] | \(1.97\times10^{-16}\) |
| M0 | History only | 0 | 0 |
| M0 | Current only | \(7.80\times10^{-17}\) [\(4.20\times10^{-17}\), \(1.26\times10^{-16}\)] | \(1.97\times10^{-16}\) |
| M0 | **Full prefix** | **0** | **0** |
| M1 | Unweighted | 0.08498 [0.07477, 0.09487] | 0.17024 |
| M1 | History only | 0.08498 [0.07477, 0.09487] | 0.17024 |
| M1 | Current only | \(1.01\times10^{-16}\) [\(6.83\times10^{-17}\), \(1.54\times10^{-16}\)] | \(3.77\times10^{-16}\) |
| M1 | **Full prefix** | **0** | **0** |
| M2 | Unweighted | 0.09960 [0.09050, 0.10915] | 0.34624 |
| M2 | History only | \(1.28\times10^{-16}\) [\(1.01\times10^{-16}\), \(1.56\times10^{-16}\)] | \(5.23\times10^{-16}\) |
| M2 | Current only | 0.09960 [0.09050, 0.10915] | 0.34624 |
| M2 | **Full prefix** | **0** | **0** |
| M3 | Unweighted | 0.10543 [0.09566, 0.11545] | 0.36999 |
| M3 | History only | 0.05130 [0.04622, 0.05608] | 0.10602 |
| M3 | Current only | 0.08794 [0.08137, 0.09518] | 0.28833 |
| M3 | **Full prefix** | **0** | **0** |

这个 factorial pattern 比单个正例更重要：M1 只需要 current ratio，M2 只需要
history ratios，M3 则只有 full prefix 普遍 identification-correct。它直接反驳了
“只用 current action ratio 已经足够”或“只 transport occupancy 已经足够”的一般性
说法。

### 4.3 Finite logged-sample recovery 与 overlap 代价

另取前 4 个 problem instances，每个用 3 个 paired logged replicates、每 replicate
3,000 条 trajectories。Full-prefix population bias 仍为零，但 sampling error 随
累计 weights 增长：

| 机制 | Full-prefix sampling RMSE，12 replicates mean [5%, 95%] | Mean max-absolute sampling error | Mean median ESS/n | Mean minimum ESS/n |
|---|---:|---:|---:|---:|
| M0 | 0.00486 [0.00362, 0.00625] | 0.01252 | 1.0000 | 1.0000 |
| M1 | 0.03041 [0.02448, 0.03540] | 0.31587 | 0.0987 | 0.00166 |
| M2 | 0.03336 [0.02532, 0.04094] | 0.33522 | 0.0988 | 0.00162 |
| M3 | 0.03246 [0.02527, 0.04014] | 0.32917 | 0.0988 | 0.00162 |

因此 exact identification 并不消除 curse of horizon：正确 estimator 可以同时是
高方差 estimator。这里的大 max error 与最低约 0.09% 的 ESS/n 必须保留，不能只
报告 population zero bias。

### 4.4 Global enumeration 与 greedy diagnostic

所有 500 instances、四种机制下都存在 global feasible schedule，greedy 也都找到
feasible schedule。但 M2 中 greedy 并不总是 global width optimum：absolute width
regret mean 0.01435、95% 分位数 0.175、maximum 0.175；relative regret mean
0.283%。M0、M1、M3 的该构造下 regret 为 0。这个结果说明 causal
committed-prefix greedy rule 不应被描述成一般 \(K^T\) global optimizer；它并不否定
该 rule 作为论文中预先定义的 sequential calibration procedure。

## 5. Study B：controlled shift 下六个 canonical methods

### 5.1 Protocol 与信息预算

这是隔离的 same-kernel semi-synthetic calibration-stress benchmark。20 个预设 seeds
在 \(\gamma\in\{-4,-2,0,+2,+4\}\) 下成对运行，且每个 setting **恰好**包含
`Standard CP`、`ACI`、`MFCS`、`SPCI`、`PRC`、`SC-PCP` 六个 canonical methods。
没有 ablation、oracle、COT、DR 或 profile row 被伪装成 baseline。

**Post-freeze reporting convention（不改变 protocol）。** 若论文只突出一个
controlled cell，默认展示 \(\gamma=-4\)，并标为 `displayed hero stress endpoint`。
冻结 primary cell 仍是 \(\gamma=-2\)，\(\gamma=-4\) 仍是 stress；这不形成新的
primary estimand、gate 或 method-selection rule，完整五点结果继续决定正式解释。
该单-cell 展示见
[`figure_controlled_stress_stage_profile.pdf`](../results/paper_controlled_stress_stage_profile_20260826/figure_controlled_stress_stage_profile.pdf)；
完整 signed curve 与六方法表仍见本 study 的 formal figure/table。

所有方法共享 3,000 条 logged calibration trajectories，其中前 1,000 条只用于冻结
grid；每个 method/seed/\(\gamma\) 用 20,000 条匹配的 fresh target-policy trajectories
评估。ACI、SPCI、PRC 另各获得 2,000 条 on-policy adaptation trajectories，按
667/667/666 三轮使用；Standard CP、MFCS、SC-PCP 不获得 target adaptation data。
累计 prefix weights 不截断，以 float64 log stabilization 计算；target policy 的
单步 ratio cap 3 是 policy definition，不是事后 weight clipping。

### 5.2 完整主结果

WSC 是百分比；其方括号为 95% seed-vector bootstrap CI。Width 方括号为 95%
Student-\(t\) CI。所有方法在所有 setting 都成功选择 20/20 seeds；该 100% selection
rate 以全部预设 seeds 为分母，Wilson 95% CI 为 [83.89%, 100%]。最后一列是每 seed
额外使用的 target-policy adaptation trajectories，不包括共同的 20,000 条只读评估
rollouts。

| \(\gamma\) | Method | WSC [95% CI] | Mean normalized width [95% CI] | Selection | Target adaptation |
|---:|---|---:|---:|---:|---:|
| -4 | Standard CP | 86.37% [85.90, 86.52] | 4.316 [4.160, 4.473] | 20/20 | 0 |
| -4 | ACI | 88.13% [87.77, 88.15] | 4.668 [4.507, 4.829] | 20/20 | 2,000 |
| -4 | MFCS | 91.81% [90.94, 92.24] | 6.034 [5.638, 6.430] | 20/20 | 0 |
| -4 | SPCI | 89.46% [88.98, 89.64] | 5.084 [4.912, 5.256] | 20/20 | 2,000 |
| -4 | PRC | 87.79% [87.13, 87.93] | 4.697 [4.535, 4.858] | 20/20 | 2,000 |
| -4 | SC-PCP | 89.83% [89.37, 90.04] | 5.187 [5.008, 5.365] | 20/20 | 0 |
| -2 | Standard CP | 87.88% [87.42, 87.99] | 3.219 [3.089, 3.348] | 20/20 | 0 |
| -2 | ACI | 88.76% [88.51, 88.86] | 3.393 [3.259, 3.527] | 20/20 | 2,000 |
| -2 | MFCS | 92.19% [91.44, 92.48] | 4.522 [4.242, 4.803] | 20/20 | 0 |
| -2 | SPCI | 89.58% [89.12, 89.72] | 3.642 [3.490, 3.794] | 20/20 | 2,000 |
| -2 | PRC | 89.27% [88.59, 89.40] | 3.577 [3.432, 3.723] | 20/20 | 2,000 |
| -2 | SC-PCP | 89.74% [89.37, 89.90] | 3.720 [3.585, 3.855] | 20/20 | 0 |
| 0 | Standard CP | 89.86% [89.53, 89.91] | 2.272 [2.158, 2.387] | 20/20 | 0 |
| 0 | ACI | 89.94% [89.71, 89.99] | 2.272 [2.152, 2.393] | 20/20 | 2,000 |
| 0 | MFCS | 91.89% [91.24, 92.17] | 2.699 [2.524, 2.873] | 20/20 | 0 |
| 0 | SPCI | 89.41% [88.95, 89.71] | 2.265 [2.133, 2.398] | 20/20 | 2,000 |
| 0 | PRC | 90.92% [90.41, 91.01] | 2.470 [2.345, 2.595] | 20/20 | 2,000 |
| 0 | SC-PCP | 89.96% [89.63, 90.09] | 2.313 [2.184, 2.442] | 20/20 | 0 |
| +2 | Standard CP | 90.54% [90.24, 90.65] | 1.792 [1.682, 1.902] | 20/20 | 0 |
| +2 | ACI | 90.26% [89.99, 90.32] | 1.749 [1.640, 1.859] | 20/20 | 2,000 |
| +2 | MFCS | 91.70% [91.21, 91.95] | 1.945 [1.833, 2.058] | 20/20 | 0 |
| +2 | SPCI | 89.80% [89.21, 89.91] | 1.725 [1.608, 1.841] | 20/20 | 2,000 |
| +2 | PRC | 91.89% [91.40, 92.13] | 1.967 [1.859, 2.075] | 20/20 | 2,000 |
| +2 | SC-PCP | 90.11% [89.84, 90.18] | 1.739 [1.631, 1.847] | 20/20 | 0 |
| +4 | Standard CP | 90.69% [90.37, 90.92] | 1.677 [1.571, 1.783] | 20/20 | 0 |
| +4 | ACI | 90.33% [90.06, 90.38] | 1.629 [1.526, 1.732] | 20/20 | 2,000 |
| +4 | MFCS | 91.92% [91.36, 92.20] | 1.812 [1.697, 1.926] | 20/20 | 0 |
| +4 | SPCI | 89.86% [89.21, 89.91] | 1.602 [1.494, 1.709] | 20/20 | 2,000 |
| +4 | PRC | 92.28% [91.65, 92.63] | 1.855 [1.734, 1.975] | 20/20 | 2,000 |
| +4 | SC-PCP | 90.15% [89.84, 90.22] | 1.618 [1.518, 1.718] | 20/20 | 0 |

逐阶段 coverage vectors、source WSC、worst-stage index、ESS 与 maximum normalized
weight share 都保存在 `summary.json`；本文表格没有用 MeanCov 替换主 WSC。
特别地，本次 all-six formal artifact 中 SC-PCP 的未四舍五入 WSC 是
`0.8982774734497070`（\(\gamma=-4\)）和 `0.8973674744367599`
（\(\gamma=-2\)）。它们是当前 controlled 六方法比较应引用的数值；旧的
two-method confirm 中约 `0.9011` 的数值属于另一份 artifact/protocol，不能替换、
拼接或挑选进本表。

### 5.3 完整 paired SC-PCP comparisons

\(\Delta\)WSC 定义为 `SC-PCP minus baseline`，正数表示 SC-PCP coverage 更高。
Width ratio 定义为 `SC-PCP / baseline` 的 paired geometric mean，小于 1 表示
SC-PCP 更窄。二者方括号均为同 seed paired 95% bootstrap CI。

| \(\gamma\) | Baseline | \(\Delta\)WSC [95% CI] | Width ratio [95% CI] |
|---:|---|---:|---:|
| -4 | Standard CP | +3.46 pp [3.13, 3.86] | 1.202 [1.192, 1.213] |
| -4 | ACI | +1.70 pp [1.42, 2.06] | 1.111 [1.103, 1.119] |
| -4 | MFCS | -1.98 pp [-2.68, -1.08] | 0.865 [0.824, 0.907] |
| -4 | SPCI | +0.37 pp [-0.04, 0.83] | 1.020 [1.009, 1.032] |
| -4 | PRC | +2.04 pp [1.63, 2.74] | 1.104 [1.089, 1.119] |
| -2 | Standard CP | +1.86 pp [1.60, 2.34] | 1.156 [1.144, 1.170] |
| -2 | ACI | +0.97 pp [0.71, 1.21] | 1.097 [1.089, 1.105] |
| -2 | MFCS | -2.45 pp [-2.93, -1.67] | 0.827 [0.796, 0.860] |
| -2 | SPCI | +0.15 pp [-0.22, 0.69] | 1.022 [1.010, 1.036] |
| -2 | PRC | +0.47 pp [0.27, 1.04] | 1.041 [1.024, 1.059] |
| 0 | Standard CP | +0.09 pp [-0.05, 0.35] | 1.017 [1.008, 1.026] |
| 0 | ACI | +0.01 pp [-0.20, 0.22] | 1.017 [1.009, 1.025] |
| 0 | MFCS | -1.94 pp [-2.32, -1.35] | 0.860 [0.830, 0.888] |
| 0 | SPCI | +0.55 pp [0.10, 0.99] | 1.022 [1.007, 1.036] |
| 0 | PRC | -0.96 pp [-1.18, -0.51] | 0.936 [0.922, 0.950] |
| +2 | Standard CP | -0.43 pp [-0.59, -0.29] | 0.970 [0.966, 0.975] |
| +2 | ACI | -0.15 pp [-0.28, -0.01] | 0.994 [0.990, 0.999] |
| +2 | MFCS | -1.59 pp [-1.92, -1.21] | 0.893 [0.876, 0.909] |
| +2 | SPCI | +0.31 pp [0.09, 0.85] | 1.010 [0.999, 1.021] |
| +2 | PRC | -1.78 pp [-2.10, -1.40] | 0.883 [0.865, 0.900] |
| +4 | Standard CP | -0.55 pp [-0.80, -0.43] | 0.965 [0.959, 0.970] |
| +4 | ACI | -0.18 pp [-0.34, -0.05] | 0.993 [0.990, 0.997] |
| +4 | MFCS | -1.77 pp [-2.11, -1.38] | 0.893 [0.880, 0.907] |
| +4 | SPCI | +0.29 pp [0.11, 0.86] | 1.012 [1.001, 1.022] |
| +4 | PRC | -2.14 pp [-2.54, -1.68] | 0.873 [0.852, 0.891] |

### 5.4 可以写进论文的解释

- 强负 shift 下，Standard CP 明确 undercovers；SC-PCP 修复了其中一部分，并显著
  优于 Standard CP、ACI 和 PRC，但代价是更宽。SC-PCP 对 MFCS 更窄，却少约
  2--2.5 pp coverage。它在 \(\gamma=-4,-2\) 的 WSC 点估计为 89.83% 和
  89.74%，不能写成“达到有限样本 90% 保证”。
- \(\gamma=0\) 时 SC-PCP 与 Standard CP/ACI 的 paired WSC difference CI 包含
  0，符合没有方向性 transport advantage 的预期；SC-PCP 比 SPCI coverage 高但
  也更宽，比 MFCS/PRC coverage 低但更窄。
- 正 shift 下，Standard CP 变得保守而不是失败；SC-PCP 相对它降低 WSC
  0.43--0.55 pp，同时缩窄 width 3.0--3.5%。这说明 correction 的作用不是机械地
  增大区间，而是针对 target score law 重新校准。
- MFCS 和 PRC 在部分 setting 给出最高 WSC，但通常以更大 width 换取。SPCI
  常给出最窄或接近最窄的集合，但 WSC 多次低于 90%。把任何一方称为全面胜出都
  会忽略 coverage--efficiency trade-off 和不同的信息预算。
- Controlled experiment 支持“prediction-mediated treatment policy 能改变 score
  law，从而使 logging-law calibration 漂移”这一受控机制陈述；它不能替代自然
  clinical setting 中 shift 强度的经验结论，也不能推出所有真实部署都会发生同等
  大小的 shift。

## 6. Study C：equal-marginal orthogonal copula gate

### 6.1 Frozen design

该 benchmark 刻意使每个 observed regime/action cell 中的两个 standardized
residual coordinates 都各自严格服从 \(N(0,1)\)，只改变两坐标相关性：easy regime
相关系数 0.9，hard regime 为 0。半径只进入 nonanticipating action policy；固定
\(\beta\) 后 source 与 target 使用完全同一个 transition/outcome kernel。预设链条是

\[
q\rightarrow\pi_q\rightarrow A_t\rightarrow H_{t+1}
\rightarrow\text{copula}\rightarrow Q_{0.9}(R_t)
\rightarrow\text{same-radius coverage}.
\]

正式设计使用 100 个未查看 seeds（94000--94198 偶数）、每 seed 50,000 trajectories、
12 stages、late stages 4--11、\(\beta\in\{-1,-0.5,0,0.5,1\}\)、
\(\kappa\in\{0,0.5,1\}\)、radius \(\in\{1.7,1.9,2.1\}\)，primary cell 为
radius 1.9 与 \(|\beta|=\kappa=1\)。所有 cells 共享 float64 common random numbers。

工程期曾查看 seeds 1 和 93000，并试过更强的 excluded copula cells。它们全部记录在
`manifest.json.engineering_contamination`，不作科学使用。正式 v1 保留 probe 前的
0.9/0 correlation、\(\beta=\pm1\)、policy shift 1.5 和 radius 1.9，并迁移到上述
100 个 untouched seeds；没有根据 probe 结果放宽 formal gates。

### 6.2 完整 gate 结果

方向性 effects 按 seed 配对，并在 late stages 平均。Coverage 数值以下用 percentage
points；relative Q90 使用百分比。Gate 是 conjunction：任何必需 magnitude check
失败即整体 NO-GO。

| Check | Observed | Frozen threshold | Result |
|---|---:|---:|---|
| Equal-marginal maximum absolute mean | 0.01641 | \(\le 0.020\) | PASS |
| Equal-marginal maximum variance error | 0.02280 | \(\le 0.030\) | PASS |
| Declared copula correlation error | 0.01711 | \(\le 0.030\) | PASS |
| \(\kappa=0\) policy TV placebo | 0 | \(\le10^{-12}\) | PASS |
| \(\kappa=0\) hard-prevalence placebo | 0 | \(\le0.003\) | PASS |
| \(\kappa=0\) relative-Q90 placebo | 0 | \(\le0.010\) | PASS |
| \(\kappa=0\) coverage placebo | 0 | \(\le0.003\) | PASS |
| \(\beta=0\) policy changes | 0.17548 | \(\ge0.050\) | PASS |
| \(\beta=0\) hard-prevalence placebo | 0 | \(\le0.003\) | PASS |
| \(\beta=0\) relative-Q90 placebo | 0 | \(\le0.010\) | PASS |
| \(\beta=0\) coverage placebo | 0 | \(\le0.003\) | PASS |
| Negative-\(\beta\) hard-prevalence shift | 0.09482 | \(\ge0.010\) | PASS |
| Positive-\(\beta\) hard-prevalence shift | 0.11102 | \(\ge0.010\) | PASS |
| Negative-\(\beta\) relative Q90 shift | 0.8017% [0.7942, 0.8092] | \(\ge3.0\%\), CI excludes 0 | **FAIL magnitude; PASS CI** |
| Positive-\(\beta\) relative Q90 shift | 0.9340% [0.9266, 0.9415] | \(\ge3.0\%\), CI excludes 0 | **FAIL magnitude; PASS CI** |
| Negative-\(\beta\) coverage gain | 0.2969 pp [0.2944, 0.2994] | \(\ge1.5\) pp, CI excludes 0 | **FAIL magnitude; PASS CI** |
| Positive-\(\beta\) coverage loss | 0.3484 pp [0.3458, 0.3510] | \(\ge1.5\) pp, CI excludes 0 | **FAIL magnitude; PASS CI** |
| Primary late minimum prefix ESS/n | 0.21511 | \(\ge0.15\) | PASS |
| Primary late maximum incremental ratio | 1.46347 | \(\le10\) | PASS |
| Primary late maximum normalized weight share | 0.001955 | \(\le0.02\) | PASS |

正式结论是 `gate.status = fail`。它不是“完全没有 effect”：方向和 paired CI 都支持
一个小但非零的 orthogonal copula shift；失败原因是它远小于预先要求的 substantive
magnitude。Runner 因而写入
`optional_six_method_stage.authorized = false`，六方法 downstream comparison 没有
启动。这是正确的 confirmatory 行为，不能在看到结果后降低 3%/1.5 pp 门槛。

## 7. 三项研究合起来能支持什么

可以支持的最强、仍然诚实的叙述是：

1. 在标准 sequential causal assumptions 下，stage-\(t\) post-action score law 需要
   history ratios 与 current-action ratio 的完整 committed prefix；exact finite MDP
   给出结构性证据。
2. 在一个明确受控、same-kernel、prediction-responsive treatment benchmark 中，
   Standard CP 的 marginal coverage 会随 shift 方向显著漂移；SC-PCP 能在负 shift
   下实质改善 calibration，并呈现可解释的 coverage--width trade-off。
3. 这种改进不是全局支配。Strong negative shift 下 SC-PCP 仍有轻微 undercoverage；
   positive shift 下 Standard CP 更保守；MFCS/PRC/SPCI 各有不同 trade-off 与信息预算。
4. 把 coordinate-wise marginals 完全锁死、只让 treatment 改变 copula 的更严格
   benchmark 在冻结 v1 中未产生足够大的 practical effect，因此只保留为 NO-GO
   mechanism diagnostic，不进入主方法排名。

这些结果不能支持“universal SOTA”“所有真实 clinical data 都有强 performative
shift”“SC-PCP 有 exact finite-sample coverage”或“greedy committed-prefix 等价于
global schedule optimization”。自然数据上的 48 个唯一 setting--method cells
（按 RQ 展示时为 54 行，包含 6 行 \(\beta=1\) 复用）应与本文 controlled 结果
并列呈现，而不能用后者替换前者。

## 8. Formal commands

以下命令描述产生 artifacts 的 protocol；要复现同一 source hash，应先在独立目录
恢复第 2 节的 source snapshot。

```bash
conda run -n ucp python scripts/run_exact_finite_mdp.py \
  --output results/work/exact_finite_mdp_20260825 \
  --instances 500

conda run -n ucp python scripts/run_controlled_six_method_benchmark.py \
  --output-dir results/work/controlled_six_method_confirm20_20260825 \
  --devices cuda:0,cuda:1

conda run -n ucp python scripts/run_copula_mechanism.py \
  --config configs/copula_mechanism.yaml \
  --output-dir results/work/copula_mechanism_v1_20260825 \
  --devices cuda:0,cuda:1
```

旧 artifact 目录不是临时 scratch space。Fresh run 必须使用新的空 output root；只有
在 source、config、seed bank、device mapping 与 artifact provenance 完全一致时才可
追加 `--resume`。

## 9. 投稿图表与 source data

本页的 exact-MDP 与 controlled all-six artifacts 已被确定性渲染为：

- [`figure_exact_prefix_identification.pdf`](../results/paper_formal_mechanism_20260826/figure_exact_prefix_identification.pdf)：
  500 paired instances 的 4×4 population-identification heatmap；
- [`figure_controlled_signed_all_six.pdf`](../results/paper_formal_mechanism_20260826/figure_controlled_signed_all_six.pdf)：
  formal all-six seeds 下的 same-radius mechanism、WSC 和 width；
- [`table_controlled_signed_all_six.pdf`](../results/paper_formal_mechanism_20260826/table_controlled_signed_all_six.pdf)：
  5 个 signed strengths × 6 个 canonical methods 的完整表。

完整 editable SVG、600-dpi TIFF、PNG preview、source-data CSV、analysis JSON、QA
和 render manifest 位于
[`results/work/formal_mechanism_report_20260826`](../results/work/formal_mechanism_report_20260826)。
Renderer 只读本页绑定的 frozen artifacts；没有重跑 seed、rollout、模型或 bootstrap。
正式论文的完整图序与 caption boundary 见
[`figure_portfolio_20260826.md`](figure_portfolio_20260826.md)。
