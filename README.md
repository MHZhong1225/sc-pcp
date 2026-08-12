# SC-PCP：逐时间步 performative coverage

本仓库只保留 **per-step SC-PCP** 这一套方法。它为顺序治疗中的每一个决策时点选择同一个 prediction-set radius
`q`，目标为

\[
\forall t\in\{0,\ldots,T-1\},\qquad
\Pr_{\tau\sim P_{\hat q}}\{Y_{t+1}\in C_{\hat q}(S_t,A_t)\}\ge 1-\alpha.
\]

完整的实验/认证语义见 [`docs/per_step_protocol.md`](docs/per_step_protocol.md)。

代码默认使用 `ucp` 环境和两张 GPU（`cuda:0,cuda:1`）。最终实验只使用
`scripts/run_per_step.py`、`scripts/run_per_step_study.py` 与
`scripts/plot_per_step.py` 三个入口。

## 方法实现

给定冻结的异方差 outcome model
\((\hat\mu(s,a),\hat\sigma(s,a))\)，主版本使用二维 normalized-max score：

\[
s_t=\max_{j\in\{1,2\}}
\frac{|Y_{t+1,j}-\hat\mu_j(S_t,A_t)|}{\hat\sigma_j(S_t,A_t)+10^{-6}}.
\]

最终 prediction set 固定为 box。其临床 worst-case cost 是

\[
J_q(s,a)=w_D\hat\mu_D+w_T\hat\mu_T+c(a)
+q(w_D\hat\sigma_D+w_T\hat\sigma_T).
\]

部署策略采用 behavior-anchored exponential tilt：

\[
\pi_q(a\mid s)\propto \mu_{\rm ref}(a\mid s)
\exp\{-\eta J_q(s,a)\},
\]

并严格投影到 `pi/mu_ref <= policy_ratio_cap`。在 synthetic/tabular oracle
实验中 `mu_ref` 和 OPE 分母是已知的 logging policy；临床日志中二者均为只在
`D_beh` 上训练、温度校准后的 propensity model。

clinical predictor / propensity path 使用两层 GRU 编码过去四个 **dynamic**
state windows，并将 latest normalized static covariates 与 action embedding 在
head 前单独融合；static values 不会被当作重复的 temporal signal。

真实数据的动态 state 采用跨数据集固定、带 missingness flag 的 schema：MAP、HR、
RR、SpO2、SBP、DBP、temperature，以及 creatinine、lactate、pH、bicarbonate、
hemoglobin、platelets、sodium、potassium；另含截至当前决策时点的累计 fluid / pressor
exposure。可用来源还加入 urine output、respiratory-support/ventilation、EtCO2、FiO2、
PEEP、tidal/minute ventilation 与 blood loss；在不提供相应通道的数据集中保持固定默认值
和 missing flag。每个已完成 interval 对各连续动态量保留 `last / mean / min / max` 与
missing flag。入院年龄、性别、来源、病区/ICU 类型等只作为 admission-time static context。
一个 bin 内的新观测只写入下一状态 `S_(t+1)`；其中后半窗单独定义 response/outcome，
不会被当前 `S_t` 或当前 action 使用。缺测值用固定临床参考值和独立 missing flag 表示，
不用未来值回填。

原始临床事件先经过按变量预先固定的生理可行范围筛选；稀有 static categorical levels
在 patient split 之前统一折叠为 `__other__`，神经网络输入的标准化值再限制于
`[-10,10]`。这些规则不查看 outcome label、action prevalence 或 `D_cert`，用于阻止单位
错误、录入异常和训练集中未见类别制造数值爆炸。

对每个候选 radius 和每个时点，COT 学习

\[
\rho_t^q(s)=d_t^q(s)/d_t^\mu(s),\qquad
\omega_t^q(s,a)=\rho_t^q(s)\pi_q(a\mid s)/\mu(a\mid s),
\]

并按时点递归拟合
\(\rho_{t+1}^q(S_{t+1})\approx
E[\rho_t^q(S_t)\pi_q(A_t\mid S_t)/\mu(A_t\mid S_t)\mid S_{t+1}]\)。
每个非初始时点一个正的、q-conditioned ratio head，从 identity ratio 初始化；训练时对每个候选
`q` 分别施加归一化项，并在独立的 `D_COT` 训练部分做 mean-one calibration。诊断记录的是
留出部分上的 `max_q |mean(rho)-1|`。

严格证书分支在 `D_cert` 上计算原始 Horvitz--Thompson 量

\[
\widehat F_{q_D,t}(q_M)=n^{-1}\sum_i
\widehat\omega_{it}^{q_D}\mathbf1\{s_{it}\le q_M\},
\]

没有外部 ratio-error bound 时，实用分支改用概率保持的 Hájek 估计

\[
\widetilde F_{q,t}(m)=
\frac{\sum_i\widehat\omega_{it}^{q}\mathbf1\{s_{it}\le m\}}
{\sum_i\widehat\omega_{it}^{q}},
\]

使 CDF 始终位于 `[0,1]`，并消除每个 `(q,t)` 权重整体尺度误差。由此导出
`K × T × K` DCov surface、其 `min_t` heatmap 以及 performative diagonal。候选 grid
固定为 `D_COT` score 的 101 个 0.50--0.999 分位点；不会
查看 `D_cert` 后再改变 grid，也不会假设 coverage 对 `q` 单调或使用 binary search。

临床数据的患者级别角色固定为：

| role | 比例 | 用途 |
|---|---:|---|
| `D_pred` | 40% | 训练并冻结 outcome model / score shape |
| `D_beh` | 15% | 历史 propensity model |
| `D_COT` | 15% | 固定 q-grid、训练 COT |
| `D_cert` | 15% | DCov、LCB 与最终 radius |
| `D_env` | 15% | 仅构建临床 Track A 的 empirical MDP |

synthetic/tabular 的 logging policy 与生成环境都已知，不需要估计 `D_beh` 或 `D_env`，因此
使用 `D_pred/D_COT/D_cert = 40/20/40%`。不拟合无用 propensity model，并把更多独立样本
用于最终 coverage 下界。对所有数据集，无需拟合 COT 的 baseline 可使用
`D_COT ∪ D_cert` 作为 calibration budget，避免在数据量上偏袒主方法。

## 认证的含义

对于权重上界 `B` 和外部提供的 simultaneous
\(L_1\) ratio-error bound \(\varepsilon_{q,t}\)，实现的是

\[
L_t(q)=\big[\widehat F_{q,t}(q)-B
\sqrt{\log(KT/\delta_{\rm samp})/(2n_{\rm eff})}-\varepsilon_{q,t}\big]_+.
\]

独立单位是患者而不是住院行。若患者 \(c\) 有 \(m_c\) 条 trajectory，则
\(n_{\rm eff}=N^2/\sum_c m_c^2\)；没有重复患者时退化为通常的 `n`。

程序枚举整个 grid，返回最小满足 `min_t L_t(q) >= 1-alpha` 的候选；不存在时
返回 `UNCERTIFIED`。

若采用 `ratio_bound_source: declared`，外部 ratio bound 必须以至少
`1-ratio_delta` 的概率同时成立，且配置要求 `0 < ratio_delta < delta`；程序用
`delta_samp = delta-ratio_delta` 构造 sampling margin，再用 union bound 合并两类失败
概率。`ratio_delta=0` 只保留给内部 exact-tabular oracle，不可拿来标记神经估计器。

这一区分是刻意的：神经 COT 的训练 loss、validation loss、normalization error
**不是** ratio-error theorem bound。因此默认配置为
`ratio_bound_source: none`，输出只会被标记为 `practical bootstrap LCB`，绝不称为 formal
certificate。只有下列三种情形会写 `CERTIFIED`：

- `ratio_bound_source: declared`：用户提供了独立、同时成立且以 `1-ratio_delta` 覆盖的
  L1 误差界（包含任何 propensity / truncation 误差）；记录标为
  `assumption_based_ratio_bound`；
- exact tabular oracle path：已知占用比，内部诊断标为 `oracle_ratio_bound`。
- finite-MDP learned-COT validation：只在已知完整 transition/occupancy 的环境中，内部枚举
  capped learned COT weight 与真 state-action ratio 的 population L1 discrepancy；诊断标为
  `tabular_exact_l1_oracle_bound`。它包含 cap bias，只验证证书逻辑，不作为另一种方法输出。

`configs/per_step_tabular_validation.yaml` 是 finite-MDP theorem plumbing 与重复验证设置。
它会输出 neural COT 相对于 exact state-action weights 的 held-out
empirical L1 error，以及 COT 和 prefix-IW 相对于 exact target CDF 的 held-out CDF
error；并额外输出枚举得到的 population capped-weight L1 bound。prefix-IW 的
trajectory-prefix weight 与 marginal
state-action weight 并非同一个随机变量，因此不把两者的权重差误称为
ratio-estimation error。exact ratio 只作为内部 theorem diagnostic，绝不作为 clinical
offline competitor。临床及连续 synthetic COT 的正式保证不能由网络拟合残差替代。

## 数据与临床轨道

原始数据实际位于 `/home/ubuntu/zmh/dataset`（不是仓库下的 `dataset/`）。每个
clinical config 会建立一次由 `cohort_seed` 固定的 raw cohort/cache；实验 seed 只改变
patient split 与模型随机性，因此多 seed 不会反复扫描大表。

| config | task | interval / T | action | outcome |
|---|---|---:|---|---|
| `per_step_mimic_iv.yaml` | ICU hemodynamic | 4h / 12 | curated IV fluid × pressor levels | hypotension, tachycardia burden |
| `per_step_eicu.yaml` | ICU hemodynamic | 4h / 12 | fluid × pressor levels | hypotension, tachycardia burden |
| `per_step_mimic_cxr.yaml` | ED→ICU multimodal respiratory | 6h / 6 | none / non-invasive / invasive support | hypoxemia, tachypnea burden |
| `per_step_inspire.yaml` | intraoperative hemodynamic | 10min / 12 | none / fluid-only / vasopressor-containing | hypotension, hypertension burden |

治疗和结果严格按窗口分开：前半窗定义 treatment exposure，后半窗生成 response。
MIMIC-IV fluid 使用审查过的 IV crystalloid/colloid item set，而不是 free-text
`fluid` 关键词；MIMIC 持续 vasopressor 按 interval overlap、CXR respiratory support
按 LOCF 处理。eICU fluid 只读取 I/O cell-level 明确的 crystalloid/colloid volume，并排除
oral/enteral intake、line flush 与 medication IVPB，不使用混合来源的累计 `intaketotal`。
低频 action 的 original-to-model mapping 会写入 seed metadata；单一治疗
action 不会被静默改写成 “none”。

MIMIC-CXR 使用 ED-to-ICU cohort，并选取 ICU 前 2h 到后 6h 的 index CXR；若图像在
ICU 入科之后，episode 的 time zero 推迟到该图像时间，因而所有治疗均在图像可用后发生。
DenseNet-121 只使用 `D_pred ∩` 官方 MIMIC-CXR `train` split 的 CheXpert labels
fine-tune，然后冻结并产生 256-d embedding。缺失的本地 JPG 会在 cohort 构建时剔除。
仓库不再提供 label-proxy 图像路径，MIMIC-CXR 结果只能来自真实 DenseNet-121 encoder。
原始 conventional oxygen 与 HFNC/NIV 在 patient split 前固定合并为 non-invasive support，
与 invasive ventilation 分开；raw-to-model mapping 会写入每个 seed 的 metadata。

INSPIRE 原始 `pressor+fluid` 在固定的前 5min action window 中约占 0.2%，不满足
预先声明的 2% support 门槛。因此实现会在划分 patient roles **之前**固定为三类
`none / fluid-only / vasopressor-containing`，并将 raw combined cell 归入最后一类；
seed metadata 仍保留 raw-to-model action mapping。这样不会因 seed 改变治疗语义。

每个真实数据集均输出两条严格分开的轨道：

- **Track A — `empirical_environment`**：仅用 `D_env` 建立冻结 PCA-32 + action-specific
  kNN transition library，fresh rollout 下的 coverage 是该 empirical MDP 的 oracle
  deployment metric；GRU history 由当前 rollout 自身 shift，不拼接 donor trajectory
  的旧 history。
- **Track B — `logged_data`**：只在原始 logged source trajectories 上报告 estimate、
  LCB、ESS、policy KL、最大 policy ratio，并给出 logged state-action 上的 prediction-set
  volume、观测 cost/utility，以及冻结 outcome model 在 logged states 上对 target action 的
  model-based cost/utility。后两类指标分别标注为 descriptive 与 model-based；都不是观察到的
  target-policy deployment coverage 或因果 policy-value 估计。

## Baselines

| record label | 信息条件 | 说明 |
|---|---|---|
| `Historical CP` | offline | 每时点 split-CP quantile 的保守最大值 |
| `MFCS-style (depth=3)` | offline | scalar-score finite-depth feedback adapter |
| `IW-SC-PCP` | offline | trajectory prefix-IW ablation；保存未截断 variance / cap-hit diagnostics |
| `SC-PCP` | offline | 最终方法：COT + probability-preserving calibration + coverage LCB；名称不随证书状态改变 |
| `ACI-style online` / `MultiDimSPCI-style online` / `Repeated recalibration` / `PRC-MaxTime-style online (grid-adapted)` | on-policy | Track A 中允许额外 deployment trajectories，数量单独记录 |

`baselines/` 中 vendored MFCS、MultiDimSPCI 与 PRC 实现并非可直接运行于 2-D
logged clinical trajectories。为避免虚假 “native baseline” 声称，仓库中的对应
adapters 均显式保留 `-style` 或 `-online` 名称；raw observational Track B 不把 native
on-policy ACI/MultiDimSPCI/PRC 写成可用。
关于三个 vendored repository 为什么不能在当前任务中诚实地称为 native upstream
reproduction，见 [baseline compatibility audit](docs/native_baseline_audit.md)。

`samples.online_rollouts` 是每个 online baseline 的**总** target-policy budget（主配置为
2,000）；三个 adaptation rounds 会把这同一预算尽量均匀地分成三份，而不会把它误当作
每轮 2,000。

没有 external ratio bound 时，程序仍保存 raw-HT sampling-only 下界，但它明确标为
`raw_ht_sampling_only_no_transport_bound`，未知 ratio error 写作 NA，绝不伪装成 0。
实际选择使用 2,000 次 patient-cluster bootstrap 的 Hájek 估计。它直接对完整冻结的
`(q,t)` family 做 studentized max-t；这避免直接 bootstrap
非光滑的 `min_t`，并同时保护数据驱动的 q 选择和所有时点约束。全成功或零方差 cell 另用
Kish-ESS one-sided simultaneous Wilson 下界，禁止出现“40/40 成功所以 LCB=1”。record label 为
`practical_hajek_cluster_bootstrap_max_t_wilson_lcb`，且
`certificate_formal=False`；它是 practical sampling guard，不控制 COT/propensity/截断的
transport bias，因此不是 theorem certificate。
若用户提供 `declared` bound，它只授予 COT formal route；prefix-IW 需要自己的误差界，
当前实现仍保持 practical-only。

每个 Track A record 仍保留
`pathwise_coverage = P(\forall t: score_t \le q)` 作为审计指标，但不再为它拟合或输出另一套
方法，也不进入最终主表。它不能套用 per-step 的 `1-alpha` 目标线。

## 主结果产物

正式主实验统一写入 `results/final/main/`。四个临床任务各运行 20 个 patient-split seeds；
每个配置显式限制最多 60,000 个候选 episode，使用 50,000 条独立 rollout 评价每个最终规则。
旧的单 seed、80% 目标、缩小模型或 proxy 图像结果不进入本仓库的结果体系。

当前冻结实现（`source_tree_sha256 = 65a10a938d80d7d69231464adf469bd51ce4382a3af5f62884116ded3f006be1`）
已经完成四个临床任务共 80 个 seeds、synthetic 200 个 seeds，以及 5 个 exact-tabular
validation seeds。统一主表见
[`results/final/main_figures/main_results.md`](results/final/main_figures/main_results.md)，机器可读版本见
[`results/final/main_figures/main_results.csv`](results/final/main_figures/main_results.csv)。

绘图器会同时导出：

- `main_results.csv` / `main_results.md`：完整主比较，不只摘录 SC-PCP；
- offline panel：Historical CP、MFCS-style、IW-SC-PCP、SC-PCP；
- online-with-adaptation-data panel：ACI-style、MultiDimSPCI-style、Repeated recalibration、
  PRC-MaxTime-style，并强制显示额外 target-policy trajectory budget；
- synthetic/tabular 的 MC oracle 只放在独立 reference panel，不伪装成同信息条件 competitor。

表中报告 `mean ± s.e.`，fresh-target 达标率把 abstention 计为失败；formal certificate rate
单列，不能与 empirical target-hit rate 混称 CertRate。Track B logged-data diagnostics 始终
单独报告，不与 Track A fresh-deployment coverage 混表。

## 运行

先进入环境：

```bash
conda activate ucp
```

主 synthetic experiment（两张卡自动按 seed 调度）：

```bash
conda run -n ucp python scripts/run_per_step.py \
  --config configs/per_step_synthetic.yaml \
  --devices cuda:0,cuda:1 \
  --output-dir results/final/main/synthetic
```

四个正式临床主实验（配置内已固定 20 seeds）：

```bash
conda run -n ucp python scripts/run_per_step.py \
  --config configs/per_step_mimic_iv.yaml --devices cuda:0,cuda:1 \
  --output-dir results/final/main/mimic_iv

conda run -n ucp python scripts/run_per_step.py \
  --config configs/per_step_eicu.yaml --devices cuda:0,cuda:1 \
  --output-dir results/final/main/eicu

conda run -n ucp python scripts/run_per_step.py \
  --config configs/per_step_inspire.yaml --devices cuda:0,cuda:1 \
  --output-dir results/final/main/inspire

conda run -n ucp python scripts/run_per_step.py \
  --config configs/per_step_mimic_cxr.yaml --devices cuda:0,cuda:1 \
  --output-dir results/final/main/mimic_cxr
```

做预设单因素研究（feedback、horizon、policy tilt、sample size、ratio cap、alpha）：

```bash
conda run -n ucp python scripts/run_per_step_study.py \
  --config configs/per_step_synthetic.yaml \
  --study feedback --values 0,0.5,1,1.5,2 \
  --devices cuda:0,cuda:1 --seeds 0:20
```

主 performativity matrix 是固定的 \(4\times4\) \(\beta\times\eta\) factorial：

```bash
conda run -n ucp python scripts/run_per_step_study.py \
  --config configs/per_step_synthetic.yaml \
  --study factorial \
  --feedback-values 0,0.5,1,2 \
  --policy-tilt-values 0.25,0.5,1,2 \
  --devices cuda:0,cuda:1 --seeds 0:20
```

`factorial`（别名 `feedback_policy`）只改变 synthetic/tabular environment 的
`feedback_strength=\beta` 与 policy 的 `tilt=\eta`；manifest 会逐格写入这两个轴、
horizon、样本量和 overlap cap，便于审计其它设定未随之变化。

`ratio_cap` study 会同步设置 COT 的确定性 LCB weight bound 为
`rho_cap × policy_ratio_cap`；因此它检验的是真实 overlap--certificate trade-off，
而不是把所有设置固定在最宽松的基础 `B`。

画 Track A/Track B 分离的表和图：

```bash
conda run -n ucp python scripts/plot_per_step.py \
  --input results/final/main \
  --output results/final/main_figures
```

产物包括每 seed 的 `records.csv`、`surfaces.npz`、`metadata.json` 和最后原子发布的
`COMPLETE`；study 根目录另有 `study_status.json` 与最终 `COMPLETE`。绘图器只读取完整 seed，
不会把中断任务或旧结果混入汇总。当前目录没有 Git metadata，因此每个 study/seed 还记录
active Python source 的 `source_tree_sha256`，保证结果可对应到确切代码快照。`surfaces.npz`
保存 DCov、diagonal、LCB、sampling/ratio margins、ESS、COT/IW pre-cap variance、cap-hit rate；tabular 运行还
保存 exact L1/CDF diagnostics。绘图脚本导出 `min_t DCov` heatmap、逐时点 coverage、
coverage–volume tradeoff、LCB/ESS/weight diagnostics 和两条轨道的 summary CSV。对 study
输入还会额外写出：

主逐时 coverage 与 coverage–volume 图只包含 `selection_estimand=per_step` 的方法。
每张图同时导出可编辑 SVG、矢量 PDF 和 300-dpi PNG，图内标明跨 seed 的误差定义与实际
seed 数。

- `track_a_certification_summary.csv`：每个 method/setting 的 fresh-target 达标率、
  abstention 和 formal-certificate rate；达标率的分母是所有 split seed，abstention
  不会被静默剔除；
- `track_a_factorial_summary.csv` 及 `track_a_factorial_worst_coverage.*`：主
  \(\beta\times\eta\) WorstCov matrix（`A=` 标记 abstention）；
- `cot_iw_surface_diagnostics.csv`、`cot_iw_horizon_summary.csv` 及
  `cot_vs_prefix_iw_horizon_diagnostics.*`：在冻结 per-step q-grid 上比较 COT 与
  prefix-IW 的 min-time ESS、raw variance、相对 MC-oracle 的 selected-q error，且仅在
  exact tabular 环境可用时加入 CDF-error panel。

DCov heatmap 会标记 Historical CP、Repeated recalibration、SC-PCP 等**标量 per-step**
selection；非恒定 stagewise controller 不会被错误投影到该 \((q_D,q_M)\) 图中。

## 验证

```bash
conda run -n ucp pytest -q
```

测试覆盖 cap-preserving policy projection、非单调 grid selection、无 ratio bound 不可
称 formal certificate、patient-cluster bootstrap、全成功边界 guard、cluster-aware formal
margin/ESS、tied q-grid、GRU empirical-history shift、patient split 与
direct-action merge semantics，以及临床 `state→action→response` 对齐、逐观测 burden、
lab label 排歧和 eICU IV-fluid/urine cell 过滤。
