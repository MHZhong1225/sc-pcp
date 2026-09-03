# SC-PCP 完整主实验结果（2026-08-22）

本页记录最终 committed-prefix marginal SC-PCP 的冻结主实验，避免把早期 profiled/LCB 结果与最终方法混用。

2026-08-24 完成的 signed controlled benchmark 是独立的 post-freeze 机制证据，见
[`experimental_evidence_20260824.md`](experimental_evidence_20260824.md)。它不改变本页
冻结 six-method suite 的任何数值、eligibility 判定或历史 provenance。

2026-08-25 的 exact finite-MDP、controlled all-six 和 orthogonal copula 正式结果见
[`formal_experiments_20260825.md`](formal_experiments_20260825.md)。它们同样是隔离的
post-freeze studies；本页仍只负责 2026-08-22 production-style suite。

2026-08-26 的 horizon×overlap、calibration-size、propensity 与 strict-split 正式
诊断见 [`formal_experiments_20260826.md`](formal_experiments_20260826.md)。这些结果
界定方法的统计与 nuisance 边界，不修改本页 frozen suite。

本页数值的新增 Pareto 与逐阶段图分别为
[`figure_main_pareto.pdf`](../results/paper_main_suite_figures_20260826/figure_main_pareto.pdf)
和
[`figure_stagewise_profiles.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_stagewise_profiles.pdf)。
后者包含全部五个 production/native RQ1 settings；较早的
`paper_main_suite_figures_20260826/figure_stagewise_profiles.pdf` 只包含三个 setting，
保留为 compact historical render。
它们只做确定性后处理，不改变任何 seed、interval 或 eligibility 判定。

## 完整性与统计口径

- 原始套件：`results/work/paper_marginal_final_20260822`
- 正文 PDF：`results/paper_marginal_final_20260822`
- 设置数：8；seed 工件：480；主结果行：2,880。
- 每个 seed 恰好包含 `Standard CP`、`ACI`、`MFCS`、`SPCI`、`PRC`、`SC-PCP` 六种方法。
- Synthetic 与 RQ3 每个设置 100 seeds；四个 clinical datasets 各 20 seeds。
- 所有方法 Selection Rate 均为 100%；SC-PCP 没有 endpoint 或 selection failure。
- WSC 指 \(\min_t |\mathcal S|^{-1}\sum_s C_{s,t}\)，不是 \(|\mathcal S|^{-1}\sum_s\min_t C_{s,t}\)。
- `Target` 是本文声明的点估计 eligibility：WSC ≥ 0.90 且 Selection Rate ≥ 95%。
  当前工作区保留的是完整运行工件，而非可独立核验的外部预注册时间戳；它不等同于
  95% coverage certification。

## RQ1 主结果

| Dataset | Method | WSC | MeanCov | Avg. norm. width | Target |
|---|---|---:|---:|---:|:---:|
| Synthetic | Standard CP | 0.8993 | 0.9001 | 1.831 | No |
|  | ACI | 0.8996 | 0.9002 | 1.832 | No |
|  | MFCS | 0.9138 | 0.9151 | 1.907 | Yes |
|  | SPCI | 0.8983 | 0.8998 | 1.831 | No |
|  | PRC | 0.9106 | 0.9119 | 1.890 | Yes |
|  | **SC-PCP** | **0.9018** | **0.9027** | **1.844** | **Yes** |
| MIMIC-IV | Standard CP | 0.8983 | 0.9107 | 2.146 | No |
|  | ACI | 0.8988 | 0.9087 | 2.111 | No |
|  | MFCS | 0.9079 | 0.9180 | 2.291 | Yes |
|  | SPCI | 0.8971 | 0.9007 | 2.015 | No |
|  | PRC | 0.9043 | 0.9158 | 2.243 | Yes |
|  | **SC-PCP** | **0.9012** | **0.9128** | **2.184** | **Yes** |
| MIMIC-CXR | Standard CP | 0.9020 | 0.9067 | 4.749 | Yes |
|  | **ACI** | **0.9001** | **0.9013** | **4.646** | **Yes** |
|  | MFCS | 0.9132 | 0.9183 | 5.057 | Yes |
|  | SPCI | 0.8963 | 0.8994 | 4.613 | No |
|  | PRC | 0.9123 | 0.9176 | 5.040 | Yes |
|  | SC-PCP | 0.9040 | 0.9083 | 4.789 | Yes |
| eICU | Standard CP | 0.9056 | 0.9169 | 2.117 | Yes |
|  | **ACI** | **0.9037** | **0.9108** | **2.034** | **Yes** |
|  | MFCS | 0.9207 | 0.9280 | 2.316 | Yes |
|  | SPCI | 0.8974 | 0.8998 | 1.915 | No |
|  | PRC | 0.9167 | 0.9258 | 2.273 | Yes |
|  | SC-PCP | 0.9081 | 0.9189 | 2.153 | Yes |
| INSPIRE | Standard CP | 0.8984 | 0.9114 | 2.442 | No |
|  | ACI | 0.8980 | 0.9098 | 2.404 | No |
|  | MFCS | 0.9040 | 0.9171 | 2.604 | Yes |
|  | SPCI | 0.8980 | 0.9022 | 2.305 | No |
|  | PRC | 0.9031 | 0.9162 | 2.573 | Yes |
|  | **SC-PCP** | **0.9010** | **0.9133** | **2.498** | **Yes** |

加粗方法是在该数据集上满足 target 的方法中 width 最小者。CI、完整精度和逐阶段曲线见最终 PDF。

## 结果应该怎样解释

SC-PCP 在 Synthetic、MIMIC-IV 和 INSPIRE 上是最窄的 target-eligible 方法；相对各自最强的 **point-eligible** baseline，paired geometric width 分别降低 2.47%、2.62% 和 2.88%。在 MIMIC-CXR 和 eICU 上，ACI 更高效，SC-PCP 分别宽 2.97% 和 5.84%。因此结论应是“SC-PCP 在五个数据集上均处于 coverage–width Pareto frontier，并在三个数据集上取得最佳 target-level efficiency”，而不是“在所有数据集统一胜出”。这里的 eligibility 始终指上文声明的 WSC 与 Selection Rate 点估计规则，不等同于 95% coverage certification。

相对 Standard CP，SC-PCP 的 width overhead 为 0.67%（Synthetic）、1.81%（MIMIC-IV）、0.84%（MIMIC-CXR）、1.67%（eICU）和 2.32%（INSPIRE）。早期方法只在单独的 **practical bootstrap/LCB audit** 中评估；它不是正式 PAC certificate，而且没有与当前方法保留同场景、同 seeds 的 paired old-to-new comparison。因此这里不直接量化当前方法相对早期版本的 width 改进。从主实验点估计看，SC-PCP 将 Synthetic、MIMIC-IV 和 INSPIRE 的 marginal worst stage 推到 0.90 之上；在这三个由 SC-PCP 跨过点阈值的 settings 中，只有 Synthetic 的 95% bootstrap interval 完全高于 0.90，MIMIC-IV 和 INSPIRE 仍跨过 0.90。eICU 的 SC-PCP interval 也完全高于 0.90。

Standard CP 的 MeanCov 可以约为或高于 0.90，同时 WSC 仍低于 0.90；原因是高覆盖阶段能够补偿低覆盖阶段。per-step 目标不允许这种跨阶段补偿，这正是同时报告 WSC 与 MeanCov 的必要性。

## RQ3 feedback stress

| Feedback \(\beta\) | Standard WSC | SC-PCP WSC | Standard width | SC-PCP width | Width overhead |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.8992 | 0.9016 | 1.966 | 1.979 | 0.70% |
| 0.5 | 0.8994 | 0.9018 | 1.917 | 1.930 | 0.69% |
| 1.0 | 0.8993 | 0.9018 | 1.831 | 1.844 | 0.67% |
| 2.0 | 0.8994 | 0.9020 | 1.663 | 1.674 | 0.66% |

RQ3 支持的是对所列连续动力学系数的数值稳定性：SC-PCP 始终把 marginal WSC 保持在约 0.902，Selection Rate 为 100%，且相对 Standard CP 的额外 width 维持在约 0.7%。该实现的 \(\beta=0\) 仍保留 action-to-difficulty 路径，因此不能把这张表解释为严格的 no-to-strong feedback 曲线，也不支持“feedback 越强，SC-PCP 相对优势越大”。

## 限制与下一步

- Clinical 只有 20 seeds；MIMIC-IV、MIMIC-CXR、INSPIRE 的 SC-PCP WSC 95% bootstrap interval 仍跨过 0.90。它们只能表述为 WSC 点估计达到目标，不能表述为 95% 证据已经证明 coverage；正文必须保留这个不确定性。
- Synthetic logging propensity 已知；clinical propensity 是拟合的，因此 clinical 结论是 model-dependent controlled-environment evidence。
- ACI、SPCI、PRC 各自额外获得 2,000 条 target-policy adaptation trajectories；Standard CP、MFCS 和 SC-PCP 是 offline 方法。主表比较最终性能，但信息预算并不相同。
- 当前首要优化方向不是再降低 calibration target，而是改善 clinical propensity/transport 质量，或在全新的 development/confirmation protocol 下研究 weak-shift shrinkage。不能在已看过的主结果上事后调阈值。
