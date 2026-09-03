# SC-PCP 完整实验证据矩阵（2026-08-30）

本文档只索引已经冻结或正式终止的实验工件，不重算结果，不改变任何 protocol、
gate、seed bank、方法定义或结果身份。唯一 paper method 是 `SC-PCP`；主比较方法名固定为
`Standard CP`、`ACI`、`MFCS`、`SPCI`、`PRC` 和 `SC-PCP`。

全文主 coverage 指标固定为

\[
\operatorname{WSC}
=\min_t\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}C_{s,t},
\]

即 `min_t mean_seed(C_seed,t)`。它不能替换为
`mean_seed(min_t C_seed,t)` 或 MeanCov。MeanCov、normalized width、Selection Rate、
逐阶段 coverage/width、ESS 和 gate availability 均按各正式 artifact 的冻结定义解释。

## 1. 总览

| 证据模块 | 正式 root | 状态 | 设计与规模 | 核心指标 | 可支持的 claim | 不可越界点 |
|---|---|---|---|---|---|---|
| Native Synthetic signed-\(\gamma\) 六方法 | `results/work/native_synthetic_signed_gamma_six_method_science_v1_exact_replay_r1` | `COMPLETE`；administrative exact replay | 20 个预设 science seeds；\(T=12\)；\(\gamma\in\{-4,-2,0,2,4\}\)；每点六个 canonical methods | WSC、MeanCov、normalized width、Selection Rate、逐阶段 coverage/width、paired SC-PCP contrasts | 在 native Synthetic signed mechanism 中比较六方法的 coverage--width trade-off；\(\gamma=-4\) 是默认 confirmatory method-comparison endpoint | Exact replay 不是新的独立 replication；除 \(\gamma=-4\) 外的 signed points 只能作 descriptive curve，不能事后排名或声明 superiority |
| Dataset-native clinical signed-\(\gamma\)：MIMIC-IV | `results/work/controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1` | `COMPLETE` | 20 个预设 confirmation seeds，20/20 science eligible；\(T=12\)；五个 signed \(\gamma\)；六方法 | 同上，另含 support、K0 fidelity、donor-overlap 和信息预算记录 | 在通过前置 gate 的 MIMIC-IV controlled environment 内，比较六方法在 \(\gamma=-4\) 的表现，并展示完整 signed descriptive curve | 不是自然临床 shift、真实 treatment effect 或真实部署保证；不能与旧 v1/v2 数值拼接 |
| Dataset-native clinical signed-\(\gamma\)：eICU | `results/work/controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1` | `COMPLETE`，但一枚预设 seed 对所有方法 unavailable | 20 个预设 confirmation seeds，19/20 support-and-K0 eligible；\(T=12\)；五个 signed \(\gamma\)；六方法 | 同上；Selection Rate 的分母固定为全部 20 seeds | 在 eICU 自身冻结的 dataset-specific environment 内作 \(\gamma=-4\) dataset-within comparison；其余 signed points作描述性敏感性 | 不能删除 unavailable seed，也不能把分母改成 19；不能跨数据集 pooling 或形成统一排名 |
| Dataset-native clinical signed-\(\gamma\)：INSPIRE | `results/work/controlled_clinical_fidelity_v4_signed_gamma_science_publish_retry_r1` | `COMPLETE` | 20 个预设 confirmation seeds，20/20 science eligible；\(T=12\)；五个 signed \(\gamma\)；六方法 | 同 MIMIC-IV | 在 INSPIRE 自身冻结的 dataset-specific environment 内作 \(\gamma=-4\) dataset-within comparison；其余 points 作描述性敏感性 | 不能解释成自然 clinical performative effect、跨数据集 conjunction 或 universal superiority |
| Dataset-native clinical signed-\(\gamma\)：MIMIC-CXR + IV/ED | `results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation`；`results/work/controlled_clinical_fidelity_v6_mimic_cxr_development` | `TERMINAL PRE-COVERAGE NO-GO` | V5 fresh confirmation 为 20 个预设 seeds，support/structural 20/20，但 K0 18/20；V6 terminal development 两条冻结 lineage 未达到要求，`terminal_no_v7=true`；计划 horizon 为 6 | 只有 support/K0/fidelity/gate 状态；没有 WSC、MeanCov、width、Selection ranking 或逐阶段 coverage | 支持“该 controlled environment construction 在本协议下未通过前置 fidelity gate”的负结果与适用边界 | **没有任何 coverage science row。** 不得画曲线、排名、插值或使用 production、旧协议、其他数据集替代；不得把 NO-GO 写成 SC-PCP coverage failure |
| Production/native five-setting robustness | `results/work/paper_marginal_final_20260822`；聚合入口 `results/work/complete_baseline_results_20260824` | `COMPLETE` | Synthetic 100 seeds；MIMIC-IV、MIMIC-CXR + IV/ED、eICU、INSPIRE 各 20 seeds；Synthetic、MIMIC-IV、eICU 与 INSPIRE 为 \(T=12\)，MIMIC-CXR 为 \(T=6\)；六方法 | WSC、MeanCov、normalized width、Selection Rate、逐阶段 coverage/width | 描述冻结 production/native empirical environments 中的 coverage--width Pareto 与可用性；作为 signed-\(\gamma\) 主结果的 robustness supplement | **该 suite 没有 controlled \(\gamma\)。** 不得把它作为默认 signed-\(\gamma\) 主实验、填补 CXR NO-GO，或解释为自然 causal/performance feedback 强度 |
| Exact prefix identification | `results/work/exact_finite_mdp_20260825` | `COMPLETE` | 500 个 paired finite-MDP instances；8 states、3 actions、\(T=4\)、每阶段 7 个 radius candidates；M0--M3 四种机制；完整枚举 \(7^4\) schedules | population coverage-surface RMSE、maximum absolute bias、finite logged-sample sampling RMSE、ESS、greedy/global width regret | 支持 full committed prefix 在同时存在 history 与 current-action channels 时的 population identification；量化有限样本 overlap 代价 | 是 identification diagnostic，不是 finite-sample coverage theorem；不能把 greedy 称为一般全局最优器 |
| Horizon \(\times\) overlap | `results/work/horizon_overlap_v1` | `COMPLETE` | 200 个 paired M3 exact-MDP instances；\(T\in\{2,4,8,12,20\}\)；nominal one-step TV \(\in\{0,.025,.05,.10,.15\}\)；每 cell 3,000 calibration trajectories | WSC、width、availability、selected-prefix ESS/n、committed-surface sup error | 量化 horizon 与 policy divergence 增大时 full-prefix estimator 的 ESS 和 surface-estimation 代价 | 不是六 baseline 排名；不能声称在所有 horizon/overlap 上优于 Standard CP，也不能隐去低 ESS cells |
| Calibration-size convergence | `results/work/rq6_ncal_convergence_v1` | `COMPLETE` | 100 个固定 M3 problems；每 problem 20 个 independent logged resamples；\(n_{cal}\in\{250,500,1000,2000,5000,10000\}\)；\(T=4,K=7\) | fixed-grid surface sup error、canonical WSC、width、availability、rowwise target attainment、endpoint use、ESS | 支持 calibration sample 增大时 coverage-surface estimation error 下降，并展示 selector availability/attainment 行为 | 不能声称证明了精确 \(n^{-1/2}\) rate、WSC 单调或有限样本 nominal guarantee；rowwise attainment 不能替换 WSC |
| Propensity robustness | `results/work/propensity_robustness_v1` | `COMPLETE` | 100 个 paired exact-MDP instances；\(S=8,A=3,T=8,K=7\)；每 instance 5,000 nuisance 和 5,000 calibration trajectories；oracle、correctly specified、reduced-state misspecified 三 arms | propensity error、WSC、width、ESS、schedule agreement、stage discrepancy、target-law drift | 支持 correctly specified plug-in 近似 oracle，以及严重 misspecification 对 ESS、schedule 和后期 stage 的影响 | primary fixed-target-law 与 appendix end-to-end target-law drift 是不同 estimands；不能声称 arbitrary misspecification robustness 或 double robustness |
| Strict-split robustness | `results/work/strict_split_robustness_v1_20260826` | `COMPLETE` | Synthetic 100 frozen-suite seeds；MIMIC-IV 20 frozen-suite seeds；controlled \(\gamma=-2\) 20 个 fresh seeds；canonical 与 strict arm 共享 D_COT-frozen grid | availability、WSC、width、selected/candidate ESS、paired differences | 支持将 D_COT 从 selection sample 中移除后的 robustness comparison | 是 post-freeze audit，不授权将 strict arm 事后替换为 canonical SC-PCP；不存在 \(\gamma=-4\) strict-split evidence |
| Prefix 与 policy-coupling ablation | `results/work/controlled_prefix_ablations_confirm20_20260824` | `COMPLETE`，post-confirmatory explanatory study | 已打开的 20-seed controlled confirmation bank；五个 signed \(\gamma\)；full SC-PCP、without-current、current-only、frozen-policy Prefix-IW、one-step-coupled Prefix-IW 五 variants | WSC、normalized width、ESS、相对 full-prefix 或 frozen-policy 的 paired contrasts | 解释 history ratio、current-action ratio 和 deployment-policy coupling 各自的作用 | 不是新的 baseline 或独立 confirmation；不得根据该结果修改 canonical selector，也不得把 diagnostic variants 放入六方法主表 |

## 2. Default signed-\(\gamma\) 身份

当前默认主端点固定为 \(\gamma=-4\)。在每个通过完整前置 gate 的数据集内，
\(\gamma=-4\) 可以承担 confirmatory method comparison 和 dataset-within ranking。
\(\gamma\in\{-2,0,+2,+4\}\) 保留为完整的 signed control curve，但只能作
descriptive sensitivity；它们不产生新的 confirmatory endpoint、method-selection rule
或 superiority claim。

Native Synthetic 与 dataset-native clinical v4 是两个独立 protocol strata。即使它们
共同展示 \(\gamma=-4\)，也不能跨 stratum pooling、合并 bootstrap、比较 normalized
width 的绝对大小，或定义跨数据集统一赢家。Clinical v4 同样明确禁止跨 MIMIC-IV、
eICU 与 INSPIRE pooling/conjunction；每个数据集只在自身冻结的 environment 中解释。

## 3. Dataset-specific setting 与 gate/confirmation 身份

Clinical v4 采用 `per_dataset_independent` 决策范围。MIMIC-IV、eICU 和 INSPIRE 的
donor/state-transition 参数在 coverage-blind development 阶段分别选择，并在任何 fresh
confirmation 前冻结；它们不是把一个数据集的参数直接复制到所有数据集。只有同时满足
support、K0 fidelity 和后续 donor-overlap screen 的数据集/seed 才能生成 science rows。

eICU 的一枚预设 seed 在 support gate 失败，因此对全部六方法统一 unavailable。
所有 coverage/width summary 可以条件于成功且 gate-eligible 的 seeds，但 Selection Rate
的分母始终是 20 个预设 seeds；不得将其重定义为 19。

MIMIC-CXR 的 v5 参数同样先在 development 选择，再进入 fresh confirmation；confirmation
未达到预先冻结的 K0 19/20 门槛。V6 只进行 coverage-blind terminal development，未通过
其更严格的冻结 lineage gate，未打开 confirmation，也未消耗 planned formal science
bank。因而 CXR 当前身份是 terminal pre-coverage NO-GO，而不是待补的缺失结果。

## 4. 指标、区间与信息预算

主比较必须同时报告以下量：

- WSC：完整 seed-stage coverage vectors 上的主 coverage scalar；
- MeanCov：跨阶段平均 coverage，只作补充，不能替换 WSC；
- mean normalized width：仅在预先声明 eligibility 后解释效率；
- Selection Rate：以全部预设 runs 为分母；
- stage coverage 与 stage normalized width：用于定位 worst stage 和阶段异质性；
- selected/candidate ESS、endpoint/failure 与 gate availability：用于解释 overlap 和可用性。

冻结区间口径为：WSC 使用完整 seed vector 的 10,000 次 percentile bootstrap；MeanCov
和 width 使用 selected seeds 上的 two-sided Student-\(t\) interval；Selection Rate 使用
全部预设 runs 上的 Wilson interval；逐阶段 coverage/width 使用 pointwise Student-\(t\)
interval，除非具体 artifact 明确冻结了其他 audit band。

Signed-\(gamma\) 六方法的共同预算为每 seed 3,000 条 calibration trajectories，其中
前 1,000 条只冻结 grid，并用 20,000 条 matched fresh target-policy trajectories 评价。
`ACI`、`SPCI`、`PRC` 各另有 2,000 条 target-policy adaptation trajectories；
`Standard CP`、`MFCS`、`SC-PCP` 为零 target-feedback arms。该信息预算差异必须披露，
不能称六方法具有完全相同的数据访问条件。

## 5. 不可混用与负结果边界

1. Production/native suite 没有 controlled \(\gamma\)，只承担 robustness；不能填补任何
   signed-\(gamma\) gate failure。
2. Native Synthetic、clinical v4、较早 controlled v1/v2 均是 protocol-specific
   artifacts；不得挑选较好数值、拼接 seeds/CI 或称为同分布 replication。
3. MIMIC-CXR 当前没有 coverage、MeanCov、width、Selection ranking 或逐阶段曲线；
   NO-GO 只说明 environment-fidelity construction 未获授权进入 science。
4. Orthogonal copula gate 的正式 downstream six-method stage 未获授权；该负结果不能通过
   事后降低 magnitude threshold 升级。
5. Prefix ablations、oracle、COT、DR、history-only、current-only、profile、LCB 和
   ordered-IUT 均不得生成 canonical paper-method row。
6. 所有 evidence 只支持 SC-PCP 的 plug-in、渐近 per-step marginal calibration claim；
   不支持 finite-sample、distribution-free、PAC、data-conditional、episode-wise、真实
   clinical causal effect 或 universal-SOTA claim。

## 6. 机器可读汇总入口

最新无重跑、确定性汇总 bundle 为
`results/work/complete_coverage_reporting_v4_minimal_quantitative_20260830`。
该 bundle 的四个机器可读 source CSV 与此前冻结的
`complete_coverage_reporting_v2_signed_gamma_cxr_terminal_20260828` 逐字节一致；
变化仅限绘图层。定量曲线图只包含四个拥有合法 signed-\(\gamma\) science rows 的
数据集，CXR 仅在 scalar table 中保留六个 `NA` rows，不在 coverage/width 坐标系内
画空值、零值或替代曲线。
其中：

- `setting_status.csv` 冻结每个 setting 的 protocol、confirmatory/ranking 身份、gate
  eligibility、science availability 和 source hash；
- `coverage_scalar_summary.csv` 汇总 WSC、MeanCov、width、Selection Rate、interval、
  budget 和 eligibility；
- `coverage_stage_profiles.csv` 汇总逐阶段 coverage/width 与 pointwise intervals；
- `paired_scpcp_contrasts.csv` 保存允许范围内的 paired SC-PCP contrasts；
- `figure_contract.json`、`figure_qa.md` 与 `render_manifest.json` 记录 deterministic
  rendering、claim boundary 和完整性校验。

这些文件只消费已有 COMPLETE/terminal artifacts；它们不是新的 experiment protocol，
也不改变任何 science result 的原始身份。

同一组冻结诊断与消融的极简定量图入口为
`results/work/complete_diagnostics_minimal_text_20260830`，PDF-only 投稿目录为
`results/paper_complete_diagnostics_minimal_text_20260830`。该 bundle 只读取本矩阵中
exact identification、horizon \(\times\) overlap、calibration-size、propensity、
strict-split 和 prefix-ablation 的正式工件；没有运行 scientific RNG、重算 bootstrap
或补造缺失区间。图内只保留面板字母、坐标/刻度、矩阵标签和必要的短图例，完整解释
应放在 manuscript caption，而不是画在图中。
