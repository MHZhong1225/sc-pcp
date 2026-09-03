# SC-PCP 投稿图表组合（2026-08-26）

本文档把冻结主实验、2026-08-25 formal studies 和 2026-08-26
theory/robustness studies 组织成一套可直接写进论文的图表叙事。所有图表均由
已有 COMPLETE artifacts 确定性后处理得到；本轮没有重跑模型、rollout 或科学
seed。最新 five-setting profile bundle 同时提供：(i) production/native 五设置的完整
逐阶段曲线；(ii) 一个 gated controlled-stress grid。后者只在通过前置 gates 的
setting 画科学曲线，NO-GO panels 只显示 gate card。\(\gamma=-4\) 逐阶段 interval
只用 artifact 中冻结的 bootstrap stream 确定性重放，不产生新的分析选择或科学
随机性。

Primary coverage scalar 始终是

\[
\operatorname{WSC}
=\min_t\frac1{|\mathcal S|}\sum_{s\in\mathcal S}C_{s,t}.
\]

WSC 用于判断 point eligibility；达到 `WSC >= 0.90` 且
`Selection >= 0.95` 后，再以 normalized width 比较效率。不能把最高 WSC 当作
最好，也不能用 MeanCov 替换 WSC。

## 1. 正文图表顺序

### Figure 1：问题与方法闭环

- PDF：[`figure_method_schematic.pdf`](../results/paper_method_schematic_20260826/figure_method_schematic.pdf)
- Work bundle：[`method_schematic_20260826`](../results/work/method_schematic_20260826)

图中将两个作用分开：过去半径 \(q_{0:t-1}\) 改变到达阶段 \(t\) 的 state
occupancy；当前候选半径 \(r\) 先改变 prediction set 和 current treatment policy，
再产生 post-action score。随后用包含 current action 的完整 likelihood prefix

\[
W_{i,t}(q_{0:t})=\prod_{j=0}^{t}
\frac{\pi_j^{q_j}(A_{i,j}\mid S_{i,j})}
     {\mu_j(A_{i,j}\mid S_{i,j})}
\]

估计 candidate coverage/width surface，选择当前 stage 最窄可行半径并 commit。
这是顺序 greedy calibration，不是 fixed point、全局 \(K^T\) optimizer 或
finite-sample certificate。

### Figure 2：Exact finite-MDP identification

- PDF：[`figure_exact_prefix_identification.pdf`](../results/paper_formal_mechanism_20260826/figure_exact_prefix_identification.pdf)
- Source/QA：[`formal_mechanism_report_20260826`](../results/work/formal_mechanism_report_20260826)

4×4 heatmap 使用 500 个 paired finite-MDP instances 的
`mean maximum absolute population bias`。M1 中 current-only 与 full-prefix 正确；
M2 中 history-only 与 full-prefix 正确；M3 同时存在 history 与 current-action
channels 时，四个结构 estimands 中只有 full-prefix identification-correct。
这里的 unweighted/history-only/current-only 是结构诊断，不是 baseline methods。

### Figure 3：五数据集 coverage--width Pareto

- PDF：[`figure_main_pareto.pdf`](../results/paper_main_suite_figures_20260826/figure_main_pareto.pdf)
- Source/QA：[`main_suite_figures_20260826`](../results/work/main_suite_figures_20260826)

每个 panel 同时画六个 canonical methods，以 WSC 为纵轴、mean normalized width
为横轴；绿色区域表示 point WSC 至少 0.90，金色圆环表示该 setting 中最窄的
point-eligible method。SC-PCP 位于 5/5 point-estimate Pareto frontiers，并在
Synthetic、MIMIC-IV、INSPIRE 上最窄；MIMIC-CXR 和 eICU 中 ACI 最窄。
这支持 `5/5 Pareto, 3/5 efficiency winner`，不支持 universal SOTA。

### Figure 4：Dataset-native gated controlled-stress grid

- PDF：[`figure_controlled_stress_grid.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_controlled_stress_grid.pdf)
- Source/QA：[`five_setting_stage_profiles_20260826`](../results/work/five_setting_stage_profiles_20260826)

五列的设置身份被写进 figure contract，不能互换：

1. Synthetic 是 frozen native \(\beta=2\) DGP，**不是** \(\gamma=-4\)；
2. MIMIC-IV 是 `controlled_clinical_extension_v2` 的 \(\gamma=-4\) six-method
   `CURVES` panel；
3. eICU、INSPIRE、MIMIC-CXR + IV/ED 是同一 clinical v2 的 K0 fidelity
   `NO-GO` gate cards，分别只有 12/20、13/20、10/20 seeds 通过，因而没有
   scientific coverage/width rows。

可画曲线的 panels 上排为逐阶段 \(C_t-0.90\)（percentage points），下排为 target
normalized width；阴影是 pointwise 95% intervals，不是 simultaneous certificate。
MIMIC-IV \(\gamma=-4\) 中，Standard CP WSC 为 86.36%，SC-PCP 为 90.09%
（paired +3.73 pp；width ratio 1.204），MFCS 为 91.89% 但更宽。SC-PCP 是
point-eligible，不代表其跨过 0.90 的 95% CI 已成立；其 WSC CI 为
[89.43%, 90.10%]。Gate card 是正式负结果，不能用 production curves、旧 MIMIC v1
或插值线填补。

### Figure 5：Controlled signed performative mechanism

- PDF：[`figure_controlled_signed_all_six.pdf`](../results/paper_formal_mechanism_20260826/figure_controlled_signed_all_six.pdf)

Panel a 给出 Standard CP 历史半径下的 same-radius late coverage gap 与 target-Q90
shift；panels b--c 给出六个 canonical methods 的 WSC 与 normalized width。
全部数值来自正式 `controlled_six_method_confirm20_20260825`，不能与较早 two-method
artifact 拼接。Figure 4 以 \(\gamma=-4\) 开场只是 presentation choice；本图仍
如实标记 \(\gamma=-2\) 为 primary、\(\gamma=-4\) 为 stress，并保留完整五点
signed curve。在 formal all-six 的 \(\gamma=-4\) cell，Standard CP WSC 为
86.37%，SC-PCP 为 89.83%（paired +3.46 pp），SC-PCP/Standard width ratio 为
1.202；MFCS 是唯一 point-eligible method。该 cell 支持 substantial correction
与 coverage--width trade-off，不支持 SC-PCP 已达到 90% 或 universal dominance。

2026-08-25 formal all-six 是 MIMIC v1 protocol-specific evidence。它的
\(\gamma=-4\) SC-PCP WSC 0.898277 仍然有效，但不得被 Figure 4 的 clinical v2
0.900887 覆盖、替换或拼接。旧单-panel
[`figure_controlled_stress_stage_profile.pdf`](../results/paper_controlled_stress_stage_profile_20260826/figure_controlled_stress_stage_profile.pdf)
保留为该 v1 的历史逐阶段渲染，不再承担“所有数据集”的展示角色。

### Figure 6：Horizon--overlap 与 calibration-size diagnostics

- PDF：[`figure_theory_diagnostics.pdf`](../results/paper_theorem_robustness_20260826/figure_theory_diagnostics.pdf)
- Source/QA：[`theorem_robustness_report_20260826`](../results/work/theorem_robustness_report_20260826)

上排是 horizon×nominal policy-TV 下的 signed coverage shortfall、selected-prefix
ESS 和 committed-surface error；下排是 \(n_{\rm cal}\) 增大时 fixed-grid surface
error、exact population WSC 和 width。图量化 overlap/horizon cost 与 surface
recovery；它不证明 universal \(n^{-1/2}\) rate、WSC 单调或收敛到恰好 0.90。

### Appendix figure：Production/native 五设置逐阶段 profiles

- PDF：[`figure_stagewise_profiles.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_stagewise_profiles.pdf)
- Source/QA：[`five_setting_stage_profiles_20260826`](../results/work/five_setting_stage_profiles_20260826)

该图完整保留 Synthetic、MIMIC-IV、MIMIC-CXR + IV/ED、eICU 和 INSPIRE 的
production/native stagewise coverage 与 width；每个 setting 都画六个 canonical
methods。它回答“主套件中每个数据集的逐阶段表现是什么”，不是 controlled
\(\gamma=-4\) replication。Synthetic 列使用 2026-08-22 RQ1 native main
\(\beta=1\)，四个 clinical 列使用同一 production-style suite。Figure 4 中另列的
Synthetic native \(\beta=2\) 是 separate stratum，不能与本图的 RQ1 Synthetic
混称。旧的三设置 compact 图
[`paper_main_suite_figures_20260826/figure_stagewise_profiles.pdf`](../results/paper_main_suite_figures_20260826/figure_stagewise_profiles.pdf)
仍是有效的有限子集渲染，但不再被描述成完整五设置图。

### Appendix figure：Robustness audits

- PDF：[`figure_robustness_audits.pdf`](../results/paper_theorem_robustness_20260826/figure_robustness_audits.pdf)

该图分别呈现 propensity denominator sensitivity、target-law drift 与 strict-split
paired differences。它不支持 double robustness、任意 misspecification robustness
或 strict-split equivalence。

## 2. 正文表格

### Table 1：Frozen five-setting comparison

当前正式 renderer 将逻辑上的 Table 1 分成 synthetic 与 clinical 两个单页 PDF：

- [`table_1_synthetic_main.pdf`](../results/paper_marginal_final_20260822/table_1_synthetic_main.pdf)
- [`table_2_clinical_main.pdf`](../results/paper_marginal_final_20260822/table_2_clinical_main.pdf)

列为 WSC、MeanCov、normalized width 和 Selection；粗体仅标记 point-eligible
methods 中最窄者。正式英文稿可在排版层合并为 Table 1a/1b，但不得重算或更换
eligibility 规则。

### Table 2：Controlled signed all-six comparison

- PDF：[`table_controlled_signed_all_six.pdf`](../results/paper_formal_mechanism_20260826/table_controlled_signed_all_six.pdf)

表格完整列出 5 个 \(\gamma\) × 6 个 canonical methods 的 WSC、MeanCov、width、
Selection 及 intervals。粗体规则与 Table 1 相同。\(\gamma=-4,-2,0,+2,+4\)
的最窄 eligible methods 分别为 MFCS、MFCS、PRC、SC-PCP、SC-PCP。

## 3. 可复现渲染入口

所有命令都要求新的空 output roots：

```bash
conda run -n ucp python tools/render_method_schematic.py \
  --work-output results/work/method_schematic_rerender \
  --paper-output results/paper_method_schematic_rerender

conda run -n ucp python tools/render_main_suite_figures.py \
  --input results/work/complete_baseline_results_20260824 \
  --work-output results/work/main_suite_figures_rerender \
  --paper-output results/paper_main_suite_figures_rerender

conda run -n ucp python tools/render_formal_mechanism_results.py \
  --exact-root results/work/exact_finite_mdp_20260825 \
  --controlled-root results/work/controlled_six_method_confirm20_20260825 \
  --work-output results/work/formal_mechanism_report_rerender \
  --paper-output results/paper_formal_mechanism_rerender

conda run -n ucp python tools/render_controlled_stress_stage_profile.py \
  --input-root results/work/controlled_six_method_confirm20_20260825 \
  --work-output results/work/controlled_stress_stage_profile_rerender \
  --paper-output results/paper_controlled_stress_stage_profile_rerender

conda run -n ucp python tools/render_five_setting_stage_profiles.py \
  --production-input results/work/complete_baseline_results_20260824 \
  --synthetic-input results/work/native_synthetic_beta2_contract_20260826 \
  --clinical-input results/work/controlled_clinical_extension_v2 \
  --work-output results/work/five_setting_stage_profiles_rerender \
  --paper-output results/paper_five_setting_stage_profiles_rerender

conda run -n ucp python tools/render_theorem_robustness_results.py \
  --work-output results/work/theorem_robustness_report_rerender \
  --paper-output results/paper_theorem_robustness_rerender
```

每个 work bundle 保存 editable SVG、600-dpi TIFF、PNG preview、source-data
CSV、analysis/contract、QA 与 SHA-256 manifest；对应 paper directory 只含 PDF。
Five-setting bundle 的机器可读入口是 `production_stage_profiles.csv`、
`setting_status.csv`、`stage_profiles.csv`、`method_summary.csv`、
`figure_contract.json`、`figure_qa.md`、`render_manifest.json` 和 `COMPLETE`。

## 4. Claim boundary

整套图表支持：prediction-radius-dependent longitudinal policy 下完整 action prefix
的 identification、SC-PCP 的渐近 per-step marginal calibration 目标、受控强 shift
下的 substantial correction，以及 overlap/horizon/nuisance 边界。它不支持
finite-sample、distribution-free、PAC、data-conditional、真实 ICU treatment-effect、
exact fixed-point、global schedule optimality 或 universal SOTA 主张。
