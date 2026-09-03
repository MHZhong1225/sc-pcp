# SC-PCP 最终正文实验与图表方案

本文档记录已经完成的 paper suite，而不是待运行计划。正式输入与输出为：

- suite：`results/work/paper_marginal_final_20260822/`
- PDFs：`results/paper_marginal_final_20260822/`

本页只冻结 2026-08-22 six-method suite 及其五个 PDF。2026-08-24 完成的独立
signed controlled benchmark、held-out confirmation 和额外机制图见
[`experimental_evidence_20260824.md`](experimental_evidence_20260824.md)；它不回写或
重定义下列历史 suite artifacts。

2026-08-25 完成的 exact finite-MDP、controlled all-six 和 orthogonal copula
正式研究见 [`formal_experiments_20260825.md`](formal_experiments_20260825.md)。它们是
新增的隔离证据，不改变本页冻结的 2026-08-22 protocol。

2026-08-26 完成的 horizon×overlap、calibration-size convergence、propensity 与
strict-split 四项正式诊断见
[`formal_experiments_20260826.md`](formal_experiments_20260826.md)。它们同样不回写
本页 protocol 或六方法 paper suite。

跨三轮冻结工件的最终正文图序、表格入口和 source-data/QA bundles 统一见
[`figure_portfolio_20260826.md`](figure_portfolio_20260826.md)。这些新增图件只做
确定性后处理，不构成新的 experiment protocol。

suite 根目录、每个 study 和每个 seed 均有 `COMPLETE`；正式 renderer 会检查
manifest、精确 seed 集合和每个 seed 的六个方法，任何一项不完整都会 fail closed。

## 1. 研究问题与复用关系

| RQ | 问题 | 数据与 seeds | 正文产物 |
|---|---|---|---|
| RQ1 | 六种方法能否兼顾 per-step marginal coverage 与集合宽度？ | Synthetic tail-shift \(\beta=1\)，100 seeds；四个 clinical datasets，各 20 seeds | Table 1、Table 2 |
| RQ2 | 各方法随 sequential stage 如何变化？ | Synthetic strong feedback \(\beta=2\) 与 MIMIC-IV | Figure 1 |
| RQ3 | feedback strength 改变时结果是否稳定？ | \(\beta\in\{0,0.5,1,2\}\)，每个条件 100 seeds | Figure 2 |
| RQ4 | committed-prefix 选择在每个 stage 做了什么？ | RQ1 Synthetic 中 `paper.mechanism_seed=1000` | Figure 3 |

物理目录只需要 `rq1/` 和 `rq3/`：RQ2 复用 `rq3/beta_2` 与
`rq1/mimic_iv`，RQ3 的 \(\beta=1\) 复用 `rq1/synthetic`，RQ4 复用
`rq1/synthetic/seed_01000` 的 committed-prefix surfaces。不得为同一结果重复运行。

Clinical datasets 固定为 MIMIC-IV、MIMIC-CXR + IV/ED、eICU 和 INSPIRE。
MIMIC-CXR 的 horizon 为 6，其余主实验 horizon 为 12。

## 2. 正文方法与公平性边界

正文只比较六个方法：

1. `Standard CP`
2. `ACI`
3. `MFCS`
4. `SPCI`
5. `PRC`
6. `SC-PCP`

旧的截断权重版本、profile 版本、oracle reference 和其他内部变体只能作为消融或诊断，
不能进入主方法图表。

所有方法在每个 seed 使用 matched evaluation random stream，并在冻结经验环境中的
fresh target-policy rollouts 上计算同一组指标。ACI、SPCI 和 PRC 属于
`on_policy_adaptation`，每个 seed 使用 2,000 条 adaptation trajectories；Standard CP、
MFCS 和 SC-PCP 属于 `offline_logged_data`。这些信息条件必须在 caption 或实验设置中披露，
不能写成六个方法具有完全相同的数据访问预算。

Clinical 实验是固定 cohort 上的 repeated-split controlled evaluation，不是真实临床部署
或独立临床总体抽样证据。

## 3. 正文指标和区间

设 \(\widehat C_{s,t}\) 为 seed \(s\) 在 stage \(t\) 的 fresh coverage，
\(\mathcal S\) 为成功输出预测集的 seeds。正文主 coverage 指标为

\[
\boxed{
\widehat{\mathrm{WSC}}_{\mathrm{marg}}
=\min_t\frac1{|\mathcal S|}\sum_{s\in\mathcal S}\widehat C_{s,t}
}.
\]

它不是 `mean_seed(worst_coverage_s)`；seed-level
`worst_coverage_s=min_t C_{s,t}` 只作诊断。其余三个正文指标为

\[
\mathrm{MeanCov}
=\frac1{|\mathcal S|}\sum_s\frac1T\sum_t\widehat C_{s,t},
\]

\[
\mathrm{AverageNormalizedWidth}
=\frac1{|\mathcal S|}\sum_s W_s,
\qquad
\mathrm{SelectionRate}
=\frac{|\mathcal S|}{|\mathcal R|}.
\]

区间定义固定如下：

- WSC：以 seed 的完整 per-time coverage vector 为重采样单位，10,000 次
  percentile bootstrap 95% CI；
- MeanCov 与 width：selected seeds 上的 Student-\(t\) 95% CI；
- Selection Rate：全部预设 seeds 上的 Wilson 95% CI；
- Figure 1：每个 stage 的跨 seed pointwise Student-\(t\) 95% interval。

Coverage 和 width 是 conditional on successful selection；Selection Rate 使用全部 runs。
表注必须报告 `n_selected / n_runs`。详细公式见 `docs/evaluation_metrics.md`。

一个方法进入正文效率比较，当且仅当 WSC 点估计至少为 0.90 且 Selection Rate
至少为 95%。Table 1/2 只加粗其中 Average normalized width 最小的方法；CI 不被
事后改成新的 eligibility gate。

## 4. 五个正式 PDF

### Table 1：Synthetic main result

文件：`table_1_synthetic_main.pdf`

- 只使用 RQ1 Synthetic tail-shift、\(\beta=1\)、100 seeds；
- 六个方法逐行展示 WSC、MeanCov、Average normalized width、Selection Rate；
- 不得混入 RQ3 的其他 feedback levels。

### Table 2：Clinical main result

文件：`table_2_clinical_main.pdf`

- 四个 clinical datasets 分组展示；
- 每个数据集均使用相同六个方法和四个指标；
- 每个 dataset-method 有 20 个预设 repeated-split seeds。

### Figure 1：Per-step coverage and width

文件：`figure_1_per_step_coverage.pdf`

采用 2 行 × 2 列 quantitative grid：

- 行：Synthetic strong feedback \((\beta=2)\)、MIMIC-IV；
- 列：Per-step coverage、Per-step normalized width；
- 横轴：真实 sequential treatment stage \(t=0,\ldots,T-1\)；
- 每个 panel 有六条方法曲线和稀疏 pointwise 95% error bars；
- coverage 列显示 `Target = 0.90`，width 列不显示 target line；
- 全图只保留一个共享 legend。

这里的 per-step coverage 是多维 outcome 的 joint coverage，不是 pathwise coverage。

### Figure 2：Feedback-strength stress test

文件：`figure_2_feedback_stress.pdf`

固定为 2 行 × 2 列：

- (a) Marginal worst-step coverage；
- (b) Mean coverage；
- (c) Average normalized width；
- (d) Selection Rate。

横轴为 \(\beta\in\{0,0.5,1,2\}\)。WSC 使用与主表完全相同的
`min_t mean_seed coverage` 和 seed-vector bootstrap。若所有方法的 Selection Rate
相同，panel (d) 显示共同曲线与说明，而不是删除 panel 或虚构方法差异。

### Figure 3：Committed-prefix mechanism

文件：`figure_3_committed_prefix_mechanism.pdf`

renderer 从 RQ1 Synthetic config 读取 `paper.mechanism_seed`；该 seed 不在预设 seeds
中时直接失败。三个 panel 的横轴均为 \(t=0,\ldots,T-1\)：

- (a) 所选候选的 calibration IW coverage estimate、独立 fresh coverage 和 0.90 target；
- (b) 每个 stage 的 selected radius \(q_t\)；
- (c) 每个 stage 的 selected effective sample size。

该图解释 stagewise committed-prefix 选择，不是 baseline 排名或 oracle 恢复图。

## 5. 统一视觉与输出规范

| Method | Color | Line | Marker |
|---|---|---|---|
| Standard CP | `#7e8c9c` | long-dashed | circle |
| ACI | `#aa3831` | dash-dot | triangle-down |
| MFCS | `#02bec4` | dotted | diamond |
| SPCI | `#7a3d9d` | short-dashed | hexagon |
| PRC | `#448c27` | dash-dot-dot | triangle-right |
| SC-PCP | `#4394f8` | solid, thicker | star |

所有正式图表使用 Times New Roman，所有可见文字为 17 pt。最终发布目录只允许上述
五个 PDF，不输出 PNG、CSV、Markdown summary 或其他中间格式。

## 6. 完整性与结果解释

正式 renderer 必须核对：

1. suite manifest 的 protocol 为 `committed_prefix_marginal_scpcp`；
2. 五个数据集和 \(\beta\in\{0,0.5,1,2\}\) 全部存在；
3. 每个 study 的 seed 集合与 config 精确一致；
4. 每个完成 seed 恰有六个正文方法；
5. Figure 3 所需 grids、candidate coverage、selected indices、radii 和 ESS shapes 一致；
6. 输出目录仅含 PDF。

正文结论应围绕“是否达到 marginal per-step target，以及达到目标需要多大 width”展开。
MeanCov 用于识别整体过覆盖；ESS、endpoint、failure、clinical cost 和旧 volume 指标只作为
机制诊断或附录内容，不与六个方法同图排名。
