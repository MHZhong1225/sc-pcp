# MIMIC-CXR 稀疏尾部环境支持研究协议

**协议 ID：** `mimic_cxr_environment_support_v1_20260903`  
**状态：** prospective、post-failure、coverage-blind；经一次 prelaunch integrity amendment 后重新冻结  
**适用数据集：** MIMIC-CXR + MIMIC-IV/ED，固定 horizon \(T=6\)  
**日期：** 2026-09-03

## 1. 研究目的与身份

本研究回答一个窄而明确的问题：在不改变 MIMIC-CXR controlled environment
的状态转移与 outcome bridge 定义时，将更多患者分配给环境经验库，能否为极少见的
重度低氧结局提供足够的经验支持，并通过原有的 pre-coverage fidelity 与 overlap
检查。

这是一次 **post-failure environment-support study**，不是 V5 或 V6 的续跑、修补、
重解释或所谓 V7。它只检验一个在任何新 coverage 计算前冻结的患者角色分配：

\[
(D_{\rm pred},D_{\rm fidelity},D_{\rm env})=(20\%,20\%,60\%).
\]

此前曾考虑同时比较 \(40/20/40\)、\(30/20/50\) 和 \(20/20/60\)。该想法在本协议
启动前被明确放弃，因为不同 role splits 会改变 fidelity patient identity，随后按
fidelity 指标选择 split 容易形成不可比的 post-failure selector。因而：

- \(40/20/40\) 只保留为既有 V5/V6 负结果的历史锚点，不在本研究重跑；
- \(30/20/50\) 不执行；
- \(20/20/60\) 是唯一新设计，不存在 split selector、候选排序或胜者选择。

## 2. 为什么只改变环境支持

V5 fresh confirmation 的 support 与 structural invariants 均为 20/20，但 K0 fidelity
只有 18/20。两枚失败 seed 都由 outcome 0 的 signed-residual Wasserstein-1 距离
触发：

- seed 119120：stored stage 1，0.295658；
- seed 119180：stored stage 3，0.320077；
- 冻结阈值为 0.25。

Outcome 0 是 post-action hypoxaemia burden，

\[
Y_0=\operatorname{mean}\{(92-\mathrm{SpO}_2)_+/10\}.
\]

Coverage-blind raw-data audit 显示，相关 action-stage cells 的 \(Y_0>1\) 事件每阶段
只有 0--2 条，跨阶段严重尾部集中在四位患者。失败 split 将这些患者分入
\(D_{\rm fidelity}\)，而相应 \(D_{\rm env}\) 没有同量级尾部。V6 增加 stagewise
action-by-SpO\(_2\) mean terms 后仍未消除失败，说明继续调节 bridge 均值参数并不是
本研究的问题。

本协议因此只检验预先指定的支持假设：把 \(D_{\rm env}\) 从 40% 增至 60%，可能提高
稀疏 outcome innovations 进入经验环境库的概率。该假设没有保证一定通过；
\(D_{\rm pred}\) 缩小至 20% 也可能降低 predictor 或 propensity 的稳定性，所有后果均由
冻结 gate 如实判定。

## 3. 永久保留的旧负结果

以下 artifact 是不可覆盖、不可删除、不可改名为 exploratory、也不可被本研究成功与否
替代的正式历史证据：

- `results/work/controlled_clinical_fidelity_v5_mimic_cxr_confirmation`：
  `CONFIRMATION_COMPLETE_NO_GO`，support/structure 20/20，K0 18/20；
- `results/work/controlled_clinical_fidelity_v6_mimic_cxr_development`：
  `DEVELOPMENT_NO_GO`，没有打开 confirmation，没有 coverage science。

若本协议最终通过，论文仍须同时说明 V5/V6 在 \(40/20/40\) lineage 下失败；新结果只能
表述为“扩大环境经验库后的新支持研究通过”，不能表述为旧失败被纠正、撤销或属于运行
错误。若本协议失败，该失败同样永久保留。

## 4. 固定数据与角色分配

### 4.1 固定 cohort 与任务

本研究沿用冻结的 MIMIC-CXR v17 cohort、特征、action ontology、outcome 定义、
历史长度与 \(T=6\)。不得根据本协议的 development 或 confirmation 结果：

- 删除严重患者或任何 seed；
- 对 outcome 作 clipping、winsorisation 或重新缩放；
- 改变完整轨迹纳入条件；
- 按 outcome、patient ID、action 或尾部状态作分层分配；
- 缩短 horizon 或重新合并 action；
- 更换 encoder、predictor、propensity 或 score 定义。

### 4.2 唯一的 \(20/20/60\) split

每个 seed 在 patient level 对排序后的唯一患者 ID 作一次冻结随机排列。令患者数为
\(N_p\)，按现有确定性取整规则依序分配：

1. 前 20% 为 \(D_{\rm pred}\)；
2. 接下来的 20% 为 \(D_{\rm fidelity}\)；
3. 其余约 60% 为 \(D_{\rm env}\)。

三组 patient IDs 必须两两不交。一个患者的完整 trajectory 只能属于一个角色。
每个 seed 必须保存三组 patient-ID hash、实际患者数、episode 数和 split audit。

\(D_{\rm pred}\) 重新拟合该 seed 的 CXR encoder、outcome predictor、behaviour propensity
及其余冻结 nuisance components；\(D_{\rm env}\) 重新拟合该 seed 的 controlled
environment；\(D_{\rm fidelity}\) 只执行 support/K0 检查，不能参与拟合或选择。

CXR encoder 的实现边界也在正式启动前完成审计和合同一致性修正。其训练行严格等于
`D_pred patient IDs ∩ official MIMIC-CXR train mask`；`predictor_fraction=0.20` 使用与
三角色划分完全相同的、由该 seed 驱动的 sorted-unique-patient permutation。因此
\(D_{\rm fidelity}\) 与 \(D_{\rm env}\) 的患者绝不进入 encoder training。该修正只使实现
符合已冻结的 \(20/20/60\) 设计，不引入新的科学参数，也没有使用 coverage、MeanCov、
width 或 selection 证据。

## 5. 固定 environment construction

本研究复用 **V5 B02 的结构合同**，而不是复用任何旧 seed 的已拟合系数：

`B02_pooled_successor_bridge_stage_one_hot`。

具体冻结为：

- C13 state kernel：raw geometry、full-cell sentinel \(k=10{,}000\)、uniform donor
  weighting、bandwidth 2.0；
- state transition：ridge residual；
- outcome residual：raw、joint two-outcome donor innovation；
- ridge：sample-normalised、intercept 不惩罚、\(\lambda=10^{-3}\)；
- outcome bridge：16 个冻结 successor clinical coordinates、current-action one-hot、
  stage one-hot，跨 stage pooled fit；
- bridge 与 residual library 仅使用该 seed 的 \(D_{\rm env}\)；
- K0 使用 \(\gamma=0\)。

不得增加新的 bridge features、interaction、candidate、regularisation grid、tail model、
state kernel 或 donor rule。唯一设计变化是 role split 从旧锚点 \(40/20/40\) 改为
\(20/20/60\)。

## 6. 冻结 seed banks 与随机流

### 6.1 两块 coverage-blind development verification

- Block A：631000, 631010, ..., 631190，共 20 seeds；
- Block B：631200, 631210, ..., 631390，共 20 seeds。

两块数据只验证唯一的 \(20/20/60\) 设计，不参与候选选择。冻结本协议前，以下
coverage-blind support/K0 pilot 已被查看：

- Block A：631000、631010、631020、631030、631040；
- Block B：631200、631210、631220、631230、631240。

因此 `development_is_scientifically_fresh=false`：这 40 个 seeds 不能称为独立、全新或
prospective development validation。上述 10 个 pilot 必须原样留在各自预设 block 中，
不得删除、替换、降权或单列后从 gate 分母移除；第 8 节的 40-seed aggregate gate 仍对
全体预设 seeds 一次性执行。冻结前没有查看任何 method-level coverage、MeanCov、width
或 selection 结果。两块 development 与旧 V5/V6 lineage 使用不同的 split/RNG identity，
但仍来自同一冻结患者 cohort，绝不是外部患者队列验证。

### 6.2 Fresh operational confirmation

- Confirmation：633000, 633010, ..., 633190，共 20 seeds；
- science summary bootstrap seed：63300019。

Confirmation bank 在 development gate 写出 `GO` 前不得运行或查看。它与两个
development blocks 的 split/RNG identity 必须无碰撞；这里的 fresh 仅指预先封存且尚未
查看的 RNG/split confirmation，并不表示新患者。由于仍来自同一冻结 cohort，不能称为
independent-patient 或 external-cohort confirmation。

### 6.3 唯一一次 prelaunch integrity amendment

原设计于 `2026-09-03T05:51:54Z` 冻结。正式运行前的 RNG-only 审计发现，原封存
confirmation bank `632000, 632010, ..., 632190` 中前 10 个 seed 的 outcome-model
流 `seed+1`，分别与 development Block B 的 CXR-encoder 流 `seed+701` 使用同一 RNG ID：

`632001, 632011, 632021, 632031, 632041, 632051, 632061, 632071, 632081, 632091`。

这是随机流身份冲突，不是 support、K0 或方法性能结果。发现时 development、confirmation
与 science 三个正式 output root 均不存在；原 632xxx confirmation 没有正式运行，任何
confirmation support/K0 row 均未打开，coverage、MeanCov、width 与 selection 也均未
计算或查看。原 bank（连同 bootstrap seed `63200019`）因此作为一个整体标记为
**unconsumed and invalidated**，此后不得恢复使用或与新 bank 混合。

在不查看性能结果的前提下，confirmation RNG identity 一次性替换为第 6.2 节的 633xxx
bank 与 bootstrap seed `63300019`。预先审计结果为：

- 20 个新 base seeds 对应的 341 个完整声明流内部 341/341 唯一；
- 与审计时 7,947 个历史 artifact/source RNG IDs 的碰撞为 0；
- development support/K0 实际会执行的 200 个流与 confirmation precoverage 的 100 个流
  之间碰撞为 0；
- bootstrap seed `63300019` 与历史 IDs 及 development 实际执行流的碰撞均为 0。

完整 confirmation mapping SHA-256 为
`c31fcc88d50e9a5b9d6ed94be7b6864b79d25c6c475b10af7f7291ddce22b6cb`；development
实际 mapping 与 confirmation precoverage mapping 的 SHA-256 分别为
`7d5140bd3a2e42399edb394ee3167583e4bae6cde2a1f5768bddc4278cae9108` 与
`1ffe184c6deba56207e2949a37c55d9fd845da5f93e65282cd4fe0a289a58bd3`。

同一次 prelaunch integrity amendment 还记录了第 4.2 节所述 encoder training-scope
实现审计与合同一致性修正；它没有改变科学设计。除 confirmation RNG identity 外，没有
修改 split、患者、B02、support/K0/overlap gate、science 参数、预算、停止规则或 endpoint。
这不是性能调参。协议于 `2026-09-03T06:11:42Z` 重新冻结；这是唯一允许并已记录的
post-freeze 修改，此后不再允许进一步变更。

### 6.4 Provenance 要求

正式启动前必须冻结并校验：配置 bytes/hash、source-tree hash、三个 seed sets、全部派生
RNG stream mapping、旧 V5/V6 parent hashes、B02 contract hash、raw-cache identity 与
本协议 hash。所有新阶段使用空 output root；`--resume` 只允许恢复完全相同的配置、源码、
seeds 和 provenance，任何不匹配均 fail closed。

## 7. 原 gate 定义保持不变

### 7.1 Support gate

每个 seed 的 \(D_{\rm env}\) 在每个 stage/action cell 至少需要 20 位 unique donor
patients。不得合并 cell、降低阈值或把 episode 数当作 unique patient 数。

### 7.2 K0 fidelity gate

每个 seed 使用 16 个 systematic logging-mixture one-step replays。以下阈值与旧协议
完全相同：

| Metric | Per-seed threshold |
|---|---:|
| maximum score KS | ≤ 0.10 |
| maximum signed-residual W1 | ≤ 0.25 |
| maximum successor-mean W1 | ≤ 0.25 |
| maximum successor-q95 W1 | ≤ 0.50 |

所有 exact structural invariants 必须通过。Action-stratified、outcome-coordinate 与
raw-tail summaries 可以保存作解释，但始终是 non-gating diagnostics。

定义 seed \(s\) 的 numeric ratio：

\[
\rho_s=\max\left\{
\frac{\mathrm{KS}_s}{0.10},
\frac{\mathrm{W1}^{\rm residual}_s}{0.25},
\frac{\mathrm{W1}^{\rm mean}_s}{0.25},
\frac{\mathrm{W1}^{\rm q95}_s}{0.50}
\right\}.
\]

结构失败或任一数值非有限时令 \(\rho_s=\infty\)；numeric pass 当且仅当
\(\rho_s\le1\)。

### 7.3 Donor-overlap screen

只有 development 与 fresh confirmation 均通过后，才允许在 confirmation 的
gate-eligible seeds 上运行 overlap probes。冻结检查点为 \(\gamma=-4\) 下的 q-mid 与
q-high/max-response 两个 3,000-trajectory probes：

- patient-aggregated local ESS 的 1% quantile ≥ 10；
- median ESS fraction ≥ 0.25；
- maximum donor probability ≤ 0.25。

Overlap 通过只是一项 empirical interpretation screen，不证明 positivity，也不产生
coverage guarantee。

## 8. Development verification gate

Block A 与 Block B 全部 40 个预设 seeds 都必须完成；不允许 seed replacement 或 deletion。
唯一设计进入 confirmation 的充要条件为：

1. 每块 support-and-numeric-K0 joint pass count 均至少 19/20；
2. 两块合计 support-and-numeric-K0 joint pass count 至少 39/40；
3. structural invariants 为 40/40；
4. 40 个 \(\rho_s\) 的 pooled linear 95th percentile 不高于 0.95；
5. support artifacts 完整，且没有实现、split、RNG 或 provenance invalidity。

每块的 pass count、q95、mean、完整 seed-ratio vector 都必须报告。Pooled q95 是唯一
gating q95；不得在看到结果后改用 nearest、lower 或其他 quantile convention。

若任何条件失败，状态写为 `DEVELOPMENT_SUPPORT_NO_GO`，立即停止。不得打开 633xxx
confirmation bank，不得增加 split、bridge 或第三块 development 数据。

## 9. Fresh confirmation 与 science unlock

若 development 为 `GO`，则在任何 confirmation 数据打开前再次冻结 \(20/20/60\)、B02、
阈值、seed bank、RNG mapping 和全部下游 science 设置。Fresh confirmation 必须满足：

- support pass count ≥ 19/20；
- numeric K0 pass count ≥ 19/20；
- support-and-K0 joint eligible count ≥ 19/20；
- 所有 20 个 K0 rows 的 exact structural invariants 均通过；
- 不得删除或替换 unavailable/failed seed。

若 confirmation 失败，写出 `CONFIRMATION_SUPPORT_NO_GO` 并永久停止；不得运行 overlap、
coverage、width 或 method selection。

若 confirmation 通过，则运行第 7.3 节的 overlap screen。至少 19/20 个预设 seeds 必须
同时通过 support、K0 与 overlap，才能写出不可变的 `SCIENCE_UNLOCK.json`。该文件必须
绑定所有 parent hashes、eligible seed identities、science config 与 source snapshot。
在它存在且校验通过前，任何代码路径都不得计算或保存 coverage、MeanCov、width、
Selection Rate 或逐阶段 method result。

Overlap 不通过时状态为 `OVERLAP_NO_GO`；可以保留 donor-only diagnostics，但不得生成
confirmatory 或 descriptive coverage curves。

## 10. 冻结的 signed-γ science

只有 `SCIENCE_UNLOCK.json` 有效时才启动六种 canonical methods：

`Standard CP`、`ACI`、`MFCS`、`SPCI`、`PRC`、`SC-PCP`。

冻结 γ grid 为

\[
\{-4,-2,0,2,4\}.
\]

\(\gamma=-4\) 是唯一 primary method-comparison endpoint。其余四点只能作 descriptive
signed sensitivity curve；不得从五点中事后选择更好看的主结果。Target-policy 单步
ratio cap 固定为 3，SC-PCP committed-prefix calibration weights 不截断。

每个 science seed 的信息预算固定为：

- 3,000 calibration trajectories，其中前 1,000 条只冻结 radius grid；
- 20,000 matched fresh target-policy evaluation trajectories；
- ACI、SPCI、PRC 各自另有 2,000 条 target-policy adaptation trajectories；
- Standard CP、MFCS、SC-PCP 不使用 target feedback。

不得根据任何 coverage/width 结果修改 split、B02、grid、γ、trajectory budget、方法参数
或 gate。

## 11. 主指标与报告规则

设 \(\mathcal R\) 是全部 20 个预设 confirmation seeds，\(\mathcal S_m\subseteq
\mathcal R\) 是方法 \(m\) 成功输出完整逐阶段半径并具有 fresh evaluation vector 的
seeds。对 seed \(s\)、stage \(t\)，令 \(\widehat C_{s,t}\) 为 fresh target-policy
joint coverage。主 coverage 指标固定为

\[
\boxed{
\widehat{\mathrm{WSC}}_m
=\min_{0\le t<T}
\frac{1}{|\mathcal S_m|}
\sum_{s\in\mathcal S_m}\widehat C_{s,t}
}.
\]

不得替换为

\[
\frac{1}{|\mathcal S_m|}\sum_s\min_t\widehat C_{s,t}
\]

或 MeanCov。MeanCov 必须作为补充指标同时报告：

\[
\widehat{\mathrm{MeanCov}}_m
=\frac{1}{|\mathcal S_m|T}
\sum_{s\in\mathcal S_m}\sum_{t=0}^{T-1}\widehat C_{s,t}.
\]

同时报告 mean normalized width、逐阶段 coverage/width、selected/candidate ESS、failure
stage 与

\[
\widehat{\mathrm{SelectionRate}}_m
=\frac{|\mathcal S_m|}{20}.
\]

Selection Rate 的分母永远是 20 个预设 confirmation seeds，不能改为 gate-eligible 或
selected seeds。Point eligibility 固定为 WSC ≥ 0.90 且 Selection Rate ≥ 0.95；
它不是 95% CI certification rule。

WSC 使用 10,000 次 complete-seed-vector percentile bootstrap；MeanCov 与 width 使用
selected seeds 上的 two-sided Student-\(t\) interval；Selection Rate 使用 20-run Wilson
interval。表和图必须显示 `n_selected / 20` 及 gate availability，不能用条件均值隐藏
abstention。

## 12. 不可更改的停止规则

1. **Development failure：** 任一第 8 节条件不满足即永久停止，不打开 confirmation。
2. **Confirmation failure：** 任一第 9 节 confirmation 条件不满足即永久 pre-coverage
   NO-GO；不允许第三次 bridge/split repair。
3. **Overlap failure：** 不写 science unlock，不生成任何 coverage/width 曲线。
4. **Implementation invalidity：** structural、split、RNG、hash 或 manifest 错误不计为
   scientific pass。只有不改变设计和随机身份的公开 administrative exact replay 才可考虑。
5. **无事后放宽：** 不得修改 0.10/0.25/0.25/0.50、19/20、39/40、q95 0.95、ESS 或
   donor-probability thresholds。
6. **无删 seed：** 除第 6.3 节透明记录的、未消耗 prelaunch RNG 冲突修正外，不能删除
   尾部患者、失败 seed、unavailable seed，不能再换 seed bank 或反复重跑直至通过。
7. **无隐式候选：** 不能回头加入 \(30/20/50\)、重跑 \(40/20/40\)、改变 B02，或把
   predictor/environment 参数当作新 selector。
8. **Science 一经打开即封存：** coverage 计算开始后，任何失败、低 selection 或不理想
   width 都必须保留，不能再修改该协议。

## 13. 允许与不允许的论文结论

若全部 gate 通过，本研究最多支持：在冻结 MIMIC-CXR cohort、\(20/20/60\) patient split、
B02 controlled environment 和经验 overlap screen 下，开展 \(\gamma=-4\) 的 dataset-within
六方法比较是可操作的。

它不支持：

- V5/V6 原负结果无效；
- 自然临床 performative effect 或 causal treatment effect；
- 外部医院泛化或独立患者验证；
- finite-sample、distribution-free、PAC、data-conditional 或 episode-wise coverage；
- SC-PCP 在所有数据集、所有 γ 或所有 baselines 上统一最优。

若任一 gate 不通过，正确结论仍是 environment-support NO-GO，而不是 SC-PCP coverage
失败，因为 method-level coverage science 从未获准打开。
