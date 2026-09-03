# 实验数据总索引（2026-08-24）

本文件是当前工作区所有已保留实验工件的**入口和证据边界**。它不复制逐 seed
CSV/NPZ（原始数据仍保存在相应 study root），而是说明哪些结果可进入正文、哪些只
能作为机制诊断、哪些已经得到 NO-GO 结论。主 coverage 口径始终是

\[
\operatorname{WSC}_{\rm marg}=\min_t\frac{1}{S}\sum_{s=1}^S C_{s,t},
\]

而不是 `mean_seed(min_t C_seed,t)`。完整定义见
[`evaluation_metrics.md`](evaluation_metrics.md)。

2026-08-25 的 exact finite-MDP、controlled 六方法和 orthogonal copula 三项正式研究
见 [`formal_experiments_20260825.md`](formal_experiments_20260825.md)。2026-08-26
新增的 horizon×overlap、calibration-size、propensity 与 strict-split 四项正式研究见
[`formal_experiments_20260826.md`](formal_experiments_20260826.md)，它是当前最新的
theory/robustness 证据记录。该页也追加记录随后完成的 dataset-native controlled
clinical extension v2。不同 protocol 的数值不得拼接或相互替换。

**展示角色不等于 protocol 角色。** 在 controlled semi-synthetic 证据内部，正文
单 cell 默认展示 \(\gamma=-4\) 作为 hero stress case；冻结 primary 仍为
\(\gamma=-2\)，完整五点 signed curve 仍是权威结果，production-style suite 继续
单列。

## 1. 结论地图

| Evidence class | 可得结论 | 不可得结论 |
| --- | --- | --- |
| 正式 paper suite | 当前 committed-prefix Prefix-IW SC-PCP 在五个主设置的 coverage--width 点估计表现 | 有限样本、distribution-free 或真实临床部署保证 |
| 原始 production / clinical policy | 可观察到的 policy response 很弱时，Standard CP 往往已经足够 | 自然临床环境存在显著 performative-treatment coverage drift |
| 受控 signed semi-synthetic / tail-shift 诊断 | 在冻结且 source/target 同 kernel 的环境中，prediction-induced treatment shift 可造成可复现的 signed score-law 与 coverage drift，SC-PCP 可按方向校正 | 这是自然 ICU 干预的经验发现或临床因果效应 |
| COT、DR、旧 LCB、K401 诊断 | 各自的效率或估计恢复是否值得进入主方法 | 第二个 paper method 或正式 PAC certificate |

**当前唯一 paper method 是 SC-PCP：**在结构性单步 policy-ratio cap 下，使用不截断的
累计 committed-prefix importance product 做 Hájek calibration，并学习自由 stagewise
radii。这里的“不截断”只指累计校准权重，不是否认 target policy 定义中的单步 ratio
cap。COT、profile、LCB、ordered-IUT、oracle 和 DR 都不生成正式 `SC-PCP` 行。方法与保证范围见
[`final_method.md`](final_method.md)。

### 1.1 2026-08-25 formal bundle

| Study | Root | Decision |
| --- | --- | --- |
| Exact finite MDP | `results/work/exact_finite_mdp_20260825` | PASS structural audit；diagnostic only |
| Controlled six methods | `results/work/controlled_six_method_confirm20_20260825` | COMPLETE；trade-off result, not universal SOTA |
| Equal-marginal copula | `results/work/copula_mechanism_v1_20260825` | formal NO-GO；six-method stage blocked |

三项 artifacts 均绑定 source-tree SHA256
`7665dfbe2f40d379879c5f3128e9767ad3ea724119b0f106129271ceaa916643`。
正式源快照为 `results/work/formal_source_snapshot_7665dfbe_20260825.tar.gz`，其
SHA256 为 `2116b9929240d8a25092f3c9015c362957bc963532930e5e720dc7ec78b2ea0b`。
正式运行后的 resume-only maintenance patch 已改变活动源树 hash；不得声称旧
artifacts 与当前 tree bitwise 相同。

### 1.2 2026-08-26 theory/robustness bundle

| Study | Root | Decision |
| --- | --- | --- |
| Horizon×overlap | `results/work/horizon_overlap_v1` | COMPLETE；200 paired exact-MDP instances，diagnostic only |
| Calibration-size convergence | `results/work/rq6_ncal_convergence_v1` | COMPLETE；100 problems×20 logged resamples |
| Propensity robustness | `results/work/propensity_robustness_v1` | COMPLETE；fixed-target primary 与 end-to-end appendix 分层 |
| Strict split | `results/work/strict_split_robustness_v1_20260826` | COMPLETE；canonical method unchanged |

四项 artifacts 的运行时 source-tree SHA256 均为
`296569d628875de774cb5012004c345d624653c1f4ecd2d3b6ff02e292f99226`。
对应本地 recovery archive 为
`results/work/extension_source_snapshot_296569d6_20260826.tar.gz`，SHA256
`57f2195a11d802fef9af84dd7f61e8df8a1d81853c899b3c03f8cac293dfa314`。
完整表格、CI、claim boundary 和工程 caveats 统一见 2026-08-26 formal record。

### 1.3 2026-08-26 dataset-native controlled clinical extension v2

| Study | Root | Decision |
| --- | --- | --- |
| Four-clinical dataset-native gated stress | `results/work/controlled_clinical_extension_v2` | COMPLETE；MIMIC-IV `CURVES`，eICU/INSPIRE/MIMIC-CXR + IV/ED 为 K0 NO-GO |

四个 datasets 的 support gate 都通过 20/20。K0 logging-mixture fidelity 的通过数
依次为 MIMIC-IV 20/20、eICU 12/20、INSPIRE 13/20、MIMIC-CXR + IV/ED 10/20；
预设门槛是至少 19/20。因此只有 MIMIC-IV 继续通过 donor-overlap screen 并生成
scientific curves，其他三项明确保存 `scientific_rows_saved=false`。这不是缺失数据，
而是 frozen gate 的正式结果。

最终 root manifest SHA256 为
`06996fee1f6eeed861a06ff2802253bebda1eaddb8e0e84b5c6577c07d599db0`；
source-tree SHA256 为
`e929fd61e2671190cc2daf10df2ca8168fb1b9131e421321fe542d539a75259d`；
recovery snapshot SHA256 为
`e0191329e036d05caff7d4b72e661e0a05cef8fc0e0d12118c7809021b773f91`。
两次透明 engineering retry、failure archives 和 exact MIMIC-IV \(\gamma=-4\)
六方法表见 [`formal_experiments_20260826.md`](formal_experiments_20260826.md) 第 10 节。

## 2. 正式、可引用的完整 paper suite

- Study root：`results/work/paper_marginal_final_20260822`
- PDF output：`results/paper_marginal_final_20260822`
- 状态：`COMPLETE`；8 settings、480 seed artifacts、2,880 method rows、480 surfaces。
- 每个 seed 恰有六个方法：`Standard CP`、`ACI`、`MFCS`、`SPCI`、`PRC`、`SC-PCP`。
- 每个 setting 的 `config.yaml`、每个 seed 的 `metadata.json`、`records.csv`、
  `surfaces.npz` 和 `COMPLETE` 都保留在 study root 下。
- 冻结运行记录的 git revision 是 `f56e20fff16d106ff6aac8536ba39a4bd355ba84`，
  source hash 是 `7b00ef06d294b0cc2b1edb211d75aa9f262c4e891a95824d4734cdfb8ce0b14e`。
  当前工作树之后已继续演化；复现时应以这些 stored provenance fields 为准，不能把
  当前 dirty tree 直接宣称为 bitwise-identical rerun。

### 2.1 RQ1：主比较

| Setting | Seeds | Standard CP WSC | SC-PCP WSC | Standard width | SC-PCP width | 按点估计最窄的 target-eligible 方法 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Synthetic | 100 | 0.8993 | 0.9018 | 1.831 | 1.844 | SC-PCP |
| MIMIC-IV | 20 | 0.8983 | 0.9012 | 2.146 | 2.184 | SC-PCP |
| MIMIC-CXR | 20 | 0.9020 | 0.9040 | 4.749 | 4.789 | ACI |
| eICU | 20 | 0.9056 | 0.9081 | 2.117 | 2.153 | ACI |
| INSPIRE | 20 | 0.8984 | 0.9010 | 2.442 | 2.498 | SC-PCP |

完整六方法的 WSC、MeanCov、width 和限制性解释见
[`main_results_20260822.md`](main_results_20260822.md)。SC-PCP 相对最窄的
point-eligible baseline 在 Synthetic、MIMIC-IV、INSPIRE 的 paired geometric width
reduction 分别为 2.47%、2.62%、2.88%；它没有在全部五个 setting 统一胜出。

### 2.2 RQ3：已有 feedback coefficient sensitivity

| Synthetic coefficient \(\beta\) | Seeds | Standard CP WSC | SC-PCP WSC | Standard width | SC-PCP width |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 100 | 0.8992 | 0.9016 | 1.966 | 1.979 |
| 0.5 | 100 | 0.8994 | 0.9018 | 1.917 | 1.930 |
| 1.0 | 100 | 0.8993 | 0.9018 | 1.831 | 1.844 |
| 2.0 | 100 | 0.8994 | 0.9020 | 1.663 | 1.674 |

该实验支持的是数值稳定性；它**不能**被写成“feedback 越强，Standard CP 失败越多”或
“SC-PCP 优势随 feedback 单调扩大”。特别是现有 `beta=0` 并不是严格的
no-feedback negative control。

## 3. 受控 shift 与 Prefix-IW 的可用证据

### 3.1 Tail-shift problem-value oracle audit

- Root：`results/work/tail_shift_problem_value_20260821`
- 完整性：100 seeds、每 seed 50,000 fresh rollouts。
- 它是 oracle / problem-value diagnostic，**不是**正式六方法主表。

| Method | Marginal WSC | 95% paired-seed CI | Mean normalized width |
| --- | ---: | ---: | ---: |
| Standard CP | 0.898650 | [0.897534, 0.899359] | 1.82905 |
| Greedy Sequential Oracle | 0.901508 | [0.900620, 0.902090] | 1.84077 |
| Current Profiled Oracle | 0.914016 | [0.912132, 0.914854] | 1.90776 |

这证明该**受控 tail-shift 环境**存在足以使 Standard CP marginal WSC 低于 0.90 的
problem value；它不证明自然 clinical production policy 也有同样强度的 shift。

### 3.2 Prefix-IW 的独立 100-seed confirm

- Root：`results/work/marginal_prefix_iw_tail_shift_confirm100_20260821`
- 100 unseen seeds `1000..1099`、每 seed 3,000 calibration trajectories 和 50,000
  fresh rollouts。
- `Marginal Prefix-IW` WSC = 0.901736，simultaneous worst band
  [0.900330, 0.903146]；width = 1.84354。
- 同一 artifact 内 Standard CP WSC = 0.899203；Greedy Sequential Oracle WSC =
  0.901840。Prefix-IW / oracle 的 paired geometric width ratio = 0.999899。
- selected-prefix ESS 最小值为 2,704.47，最小 candidate ESS 为 2,660.02；没有
  endpoint 或 grid failure。

这是当前“Prefix-IW 是可靠 transport engine”的最直接实证支撑。但该 early
confirm 的 `confirm_go=false`：它没有满足当时用于“Standard 必须被显著否定”的
附加 gate；不能把它改写成更强的临床或有限样本证据。

### 3.3 Signed \(\gamma\) benchmark：development20 与 fresh confirm20

- Development root：
  `results/work/controlled_prefix_benchmark_development20_20260824`。
- Held-out confirmation root：
  `results/work/controlled_prefix_benchmark_confirm20_20260824`。
- 两个 root 都有 study-level `metadata.json`、`summary.json`、20 个逐 seed JSON 和
  原子 `COMPLETE` marker；protocol 均为
  `controlled_performative_prefix_benchmark_v1`。
- Development seeds 为 `12200,12202,...,12238`，只用于冻结 unchanged canonical
  selector 与 primary mechanism endpoint \(\gamma=-2\)；后续采用 \(\gamma=-4\)
  的展示约定不改变该 development decision。Confirmation 使用此前未打开且
  与 development 不相交的 `12400,12402,...,12438`。
- 两个角色的 experiment source hash 完全相同：
  `23403dc6d0282a4b0c22e8894a5e4dbd7f523737454e049f969080c14f3dee0d`。
- 每个 seed/\(\gamma\) 使用 3,000 calibration、1,000 grid 与 20,000 fresh reference
  trajectories；\(\gamma\in\{-4,-2,0,2,4\}\)。source 与 target 使用同一个
  \(K_\gamma\)，只把 policy 从 \(\mu\) 改为 radius-dependent \(\pi_q\)，因此
  same-radius gap 隔离的是 prediction-mediated policy change，而不是把两个环境 kernel
  混在一起。

Fresh confirmation 的主数值如下。`Late drift` 是 Standard CP 在同一历史半径下的
target-minus-source coverage gap；区间与 Q90/width 区间均按 complete seed 配对
bootstrap。

| \(\gamma\) | Standard source WSC | Standard target WSC | SC-PCP target WSC | Late drift, pp [95%] | Target Q90 shift [95%] | SC/Standard width [95%] | Min selected ESS/n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -4 | 0.898395 | 0.861487 | 0.901087 | -3.443 [-3.738, -3.204] | +19.41% [+17.44, +21.66] | 1.2250 [1.2016, 1.2516] | 0.0338 |
| -2 | 0.896982 | 0.879432 | 0.901060 | -1.775 [-1.915, -1.657] | +12.20% [+11.18, +13.40] | 1.1563 [1.1444, 1.1697] | 0.2154 |
| 0 | 0.897030 | 0.898360 | 0.899990 | +0.131 [+0.077, +0.186] | -0.92% [-1.36, -0.49] | 1.0064 [1.0022, 1.0105] | 0.6124 |
| +2 | 0.898040 | 0.906357 | 0.901310 | +0.831 [+0.720, +0.972] | -6.13% [-7.15, -5.36] | 0.9671 [0.9613, 0.9720] | 0.7073 |
| +4 | 0.898302 | 0.910190 | 0.901297 | +1.155 [+0.939, +1.447] | -8.67% [-11.00, -6.98] | 0.9527 [0.9408, 0.9622] | 0.5982 |

这条曲线体现的是**有符号的 performative-treatment shift**：负方向使 Standard CP
显著欠覆盖，正方向使其过保守；未改动的 SC-PCP 在完整曲线上保持约 0.90，并在正方向
缩窄集合，而不是只做 radius inflation。\(\gamma=-4\) 的最小 selected ESS 只有约
101/3,000，是必须同时披露的 overlap stress boundary。该 benchmark 使用真实 MIMIC-IV
covariates 与观测标准化 residual innovations，但 donor reweighting 是刻意构造、与
calibration 对齐的 semi-synthetic mechanism；它不能被解释成自然 ICU treatment
effect。development、confirmation、simultaneous stage audit、paper figure 与全部区间见
[`experimental_evidence_20260824.md`](experimental_evidence_20260824.md)。

### 3.4 2026-08-25 controlled all-six formal extension

- Root：`results/work/controlled_six_method_confirm20_20260825`；
- 20 个全新 seeds `91000,91010,...,91190`，五个 \(\gamma\) 单元，每个单元恰含
  六个 canonical methods；
- Standard CP、MFCS、SC-PCP 使用 0 条 target adaptation trajectories；ACI、SPCI、
  PRC 各自使用 2,000 条；每个 method/seed/\(\gamma\) 使用 20,000 条 fresh
  target-policy evaluation trajectories；
- 在 \(\gamma=-4,-2\) 下，SC-PCP 相对 Standard CP 的 WSC 分别提高 3.46 pp 和
  1.86 pp，但 SC-PCP 自身 WSC 为 0.898277 和 0.897367，仍略低于 0.90；
- MFCS 是两个负向单元中唯一 point-eligible 的方法；paired geometric ratio 对应
  其 width 分别比 SC-PCP 大约 15.6% 和 20.9%。正向单元中 SC-PCP 更窄，但
  Standard/MFCS/PRC coverage
  更高。该结果是 coverage--width trade-off，不是 universal SOTA。

正文若只展示一个 controlled cell，使用 formal all-six 的 \(\gamma=-4\) rows，并
同时报告 SC-PCP WSC `0.898277`、residual undercoverage、width 和 MFCS。不得把 earlier
two-method 的 SC-PCP `0.901087` 或其 ESS 拼接到该 hero ranking 中。
该默认展示 slice 的逐阶段 coverage--width 图见
[`figure_controlled_stress_stage_profile.pdf`](../results/paper_controlled_stress_stage_profile_20260826/figure_controlled_stress_stage_profile.pdf)。

完整 30 行主表、25 个 paired comparisons、区间、预算、source snapshot 和
claim boundary 见 [`formal_experiments_20260825.md`](formal_experiments_20260825.md)。
下文 2026-08-24 two-method confirm 仍是有效的历史机制工件，但其约 0.9011 的
SC-PCP WSC 不得冒充本次 all-six formal 数值。

### 3.5 Dataset-native controlled clinical extension v2

- Root：`results/work/controlled_clinical_extension_v2`；
- 四个 clinical datasets 各 20 个预设 seeds，patient-disjoint
  \(D_{\rm pred}/D_{\rm fidelity}/D_{\rm env}=40/20/40\)；
- 每个数据集使用自身 action ontology、horizon 与 donor environment；禁止用现存
  MIMIC v1 或其他数据集替换；
- Support gate 均为 20/20；K0 fidelity 仅 MIMIC-IV 达到 20/20 并越过 19/20
  门槛，eICU/INSPIRE/MIMIC-CXR + IV/ED 分别为 12/20、13/20、10/20 NO-GO；
- MIMIC-IV 进一步通过 \(\gamma=-4\) q-mid/q-high donor-overlap screen，并保存
  五点 signed curve；其他三个 settings 没有 scientific coverage rows。

MIMIC-IV confirmatory \(\gamma=-4\) 中，Standard CP、MFCS、SC-PCP 的 WSC/width
分别为 `0.863580/4.20938`、`0.918937/5.98445`、`0.900887/5.06708`。
SC-PCP 相对 Standard CP 的 paired WSC improvement 为 3.731 pp，width ratio 为
1.204；相对 MFCS 的 WSC 低 1.805 pp，但 width ratio 为 0.852。SC-PCP 的 WSC
CI `[0.89431,0.90105]` 跨过 0.90；只能说 point-eligible，不能说 95% coverage
已经被证明。

这项 v2 与 2026-08-25 MIMIC v1 使用不同 split、gates、seeds 与 source contract。
旧 v1 的 SC-PCP \(\gamma=-4\) WSC `0.898277` 保留为 protocol-specific result，
不得被 v2 的 `0.900887` 覆盖或挑选替换。完整六方法表、gates、两次 engineering
retry 与 provenance hashes 见
[`formal_experiments_20260826.md`](formal_experiments_20260826.md) 第 10 节。

## 4. Transport-estimator 诊断：均为 NO-GO，不进入主方法

在进入 estimator NO-GO 记录前，当前 controlled benchmark 还保留了一组只作机制解释的
post-confirmatory ablation：

- Root：`results/work/controlled_prefix_ablations_confirm20_20260824`；
- 20 seeds \(\times\) 5 个 \(\gamma\) \(\times\) 5 个方法，共 500 条记录，全部 selected；
- canonical 行逐 seed/\(\gamma\) 与 parent confirm artifact 完全复现；
- 在 \(\gamma=-2\) 下，Full SC-PCP WSC 为 0.901060，删除 current ratio 后为
  0.887890，current-only 为 0.888130；
- fixed-policy Prefix-IW 为 0.897907，而把相同 radii 用于其诱导的新 policy 后降到
  0.890217；\(\gamma=0\) 的 fixed-vs-coupled placebo gap 区间跨 0。

该 artifact 只验证 current action、历史 prefix 和 policy coupling 的作用，不用于重新选择
或修改 canonical selector。完整 signed 表、paired intervals 和 ESS 解释见
[`experimental_evidence_20260824.md`](experimental_evidence_20260824.md)。

### 4.1 Fixed-schedule learned COT recovery

- Development root：`results/work/fixed_schedule_cot_probe_dev5_20260824`（5 seeds）。
- Independent replication root：`results/work/fixed_schedule_cot_probe_replication20_20260824`（20 fixed seeds）。
- 每个 gamma / seed 使用 3,000 calibration trajectories 和 10,000 fresh target
  reference rollouts；目标是 recovery error，不是最终 coverage ranking。

| \(\gamma\) | Prefix CDF sup error | COT CDF sup error | Prefix Q90 abs. error | COT Q90 abs. error | Prefix min ESS/n | COT min ESS/n |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.011555 | 0.011648 | 0.035325 | 0.036334 | 0.578 | 0.671 |
| -2 | 0.012889 | 0.013944 | 0.049948 | 0.051945 | 0.264 | 0.314 |
| -3 | 0.014058 | 0.017376 | 0.059165 | 0.064000 | 0.182 | 0.221 |
| -4 | 0.014777 | 0.019134 | 0.063587 | 0.071693 | 0.163 | 0.085 |

COT 在一部分条件下提高 ESS，但没有更小且稳定的 CDF/quantile error；强 shift 下甚至
出现更差 ESS。结论：**COT transport estimator NO-GO**。

### 4.2 Sequential doubly robust (DR) recovery

- Root：`results/work/sequential_dr_probe_dev20_20260824`
- 20 seeds，与上节 COT replication 共用 frozen schedules/reference。

| \(\gamma\) | Prefix Q90 error | DR Q90 error | DR / Prefix ratio | 该 ratio 的 95% upper | DR CDF error - Prefix CDF error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.036025 | 0.036447 | 1.0117 | 1.0332 | -0.000006 |
| -2 | 0.051657 | 0.051702 | 1.0009 | 1.0326 | +0.000004 |
| -3 | 0.060051 | 0.062455 | 1.0400 | 1.0776 | +0.000275 |
| -4 | 0.066778 | 0.066616 | 0.9976 | 1.0517 | +0.000215 |

冻结 gate 为 `NO_GO`。DR 不应进入 fresh confirmation、clinical 扩展或 paper method
selection。

## 5. 历史 / 退役路线：保留以便审计，不与 SC-PCP 混用

| Study | Root | 完成度 | 结论 | 使用边界 |
| --- | --- | --- | --- | --- |
| Old A--E conservatism decomposition | `results/work/conservatism_decomposition_standard_fresh_20260821` | 100 complete seeds | 旧 profile/LCB pipeline 相比 greedy oracle 的 width ratio E/A = 1.067262；主要 realized overhead 来自 profiled bridge 与 LCB guard | 仅解释为什么退役旧路径；详细数值见 `conservatism_decomposition_20260823.md` |
| PAC K401 grid refinement | `results/work/pac_grid_refinement_20seed_20260823` | 20 complete seeds | K401/K101 width ratio = 0.998117，仅 0.1883% 改善，低于冻结 0.5% gate | **NO-GO**；不能恢复旧 PAC claim |
| Phase-0 standard/tail oracle replay | `results/work/phase0_tail_shift_confirm_1000_1100_20260821` | 100 complete seeds | profile-vs-free-stagewise oracle surfaces | Oracle/context diagnostic；不进入六方法表 |

`archive/` 中的 `failed_*`、`obsolete_*`、早期 render 和 preflight 目录均是明确退役的
历史记录；它们不构成正文或当前方法的证据。

## 6. 如何读取与复现

1. **正文结果**：只从 `paper_marginal_final_20260822` 读取；运行
   `tools/render_paper_results.py` 的 validation path，不手工拼接不同 study 的 CSV。
2. **机制或 estimator 诊断**：只在各自 root 内读取 `summary.json` 与 per-seed data；
   不把 oracle/COT/DR 行插入 canonical six-method comparison。
3. **新实验**：必须使用新的空 output root；不能覆盖 frozen roots，也不能以已有
   seed data 改参数后重新写 summary。
4. **报告受控 shift**：只从上节两组原子 artifacts 与其 renderer source data 读取；
   必须同时报告 source/target 同一 kernel、policy overlap/ESS、fresh target evaluation、
   完整 signed/strength curve 和 semi-synthetic claim boundary。
5. **理论与鲁棒性诊断**：从四个 2026-08-26 roots 读取，并以
   [`formal_experiments_20260826.md`](formal_experiments_20260826.md) 的 estimand
   边界解释；不得把 current/history ablations 当 baseline，不得合并 propensity
   primary 与 end-to-end appendix，也不得根据 strict-split 结果事后换方法。
6. **投稿图表**：使用
   [`figure_portfolio_20260826.md`](figure_portfolio_20260826.md) 中逐项绑定的
   deterministic render bundles。Paper directories 只含 PDF；SVG/TIFF/PNG、source
   CSV、analysis、QA 与 hash manifest 保留在对应 `results/work` 目录。渲染不得触发
   科学 seed、rollout 或模型；\(\gamma=-4\) stage-profile renderer 只按正式
   artifact 已冻结的 bootstrap seed 确定性重放 pointwise bands。新的 five-setting
   bundle 将 production/native 五设置曲线与 gated controlled grid 分成两个 PDF；
   NO-GO panel 不得出现伪造或替代曲线。

## 7. 当前证据边界与下一步

- 正式 production six-method suite、tail-shift/Prefix-IW audits、旧 signed-\(\gamma\)
  development/confirm，以及新的 controlled all-six formal extension 都有可审计
  artifact；各协议必须分开引用，不能跨 root 挑选数值。
- COT、DR、K401 都已有完整 NO-GO artifact，后续不应继续调参或恢复为主方法。
- current-action ratio、full prefix 与 fixed-versus-coupled policy 的解释性 ablation，
  以及 exact finite-MDP M0--M3 结构诊断均已完成；当前重点是独立 proof review 和
  closest-work 定位，而不是继续调 selector。
- Horizon×overlap、calibration-size convergence、propensity sensitivity 与 strict
  split 已全部完成并有 formal artifact。它们显示 overlap/horizon curse、surface
  error decay、propensity misspecification 的 ESS/late-stage 代价，以及 strict split
  的近似行为；不提供 finite-sample/PAC、exact rate、double robustness 或 equivalence。
- Dataset-native clinical v2 已完成：MIMIC-IV 通过 gates 并保存 curves；eICU、
  INSPIRE、MIMIC-CXR + IV/ED 在 K0 fidelity 被正式判 NO-GO，且没有 scientific
  coverage rows。该 negative evidence 不能用 production/native 或旧 v1 artifacts
  补齐，也不能据此做跨数据集 conjunction。
- difficulty 不由 conformity-score rank 定义的 equal-marginal copula benchmark 已按
  冻结协议完成并判 NO-GO：方向性 effect 非零，但 Q90/coverage 幅度未达到 3%/1.5 pp
  门槛。不得依据正式结果降低门槛或重调 DGP。
- \(\gamma=-4\) 必须作为低 overlap stress endpoint 连同 ESS 披露；不能删除 seeds、
  事后 clip 累计权重或加入 \(\gamma\)-specific margin 来美化结果。
