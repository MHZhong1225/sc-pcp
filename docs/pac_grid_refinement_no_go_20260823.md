# K=401 置信路径效率优化：20-seed NO-GO（2026-08-23）

本诊断只研究已经退役的 `profile + learned COT + practical bootstrap/LCB + ordered-IUT` 路径，不改变当前 committed-prefix marginal SC-PCP。问题是：在保持 coverage target、nominal confidence level、数据角色和 selector 逻辑不变时，把 scalar certification grid 从 101 个点加密到 401 个点，能否显著回收离散化造成的 width。

## 冻结协议

- Standard synthetic，seeds `0..19`。
- Target coverage 固定为 0.90；`delta=0.05`。
- K101 与 K401 共用同一个 D_COT-frozen profile、同一个 fitted COT、D_cert、2,000 次 patient-cluster bootstrap、widest-to-narrowest fixed sequence 和 50,000-rollout fresh CRN stream。
- K401 在每对旧 grid knots 之间插入 3 个 deterministic geometric knots；所有旧 knots 被逐值保留。
- K101 weights、point estimates、LCBs、estimated widths 和 selection 独立重算，不能由 K401 surface 切片冒充 parity。
- 10,000 次 paired seed bootstrap 汇总 width ratio 与 coverage difference。

在查看 20-seed 结果前锁定的 production gate 是：20/20 paired available；K101 parity 通过；K401 marginal WSC 不低于 0.90；paired WSC loss 的 95% lower endpoint 不低于 -0.002；width ratio 不高于 0.995 且 one-sided 95% upper bound 小于 1；ESS、cap-hit、target、delta 和 certificate identity 均通过。0.5% 的 materiality threshold 用来补偿 grid evaluation 约 4 倍的候选维度、内存与计算。

## 最终结果

Study root 为 [`results/work/pac_grid_refinement_20seed_20260823`](../results/work/pac_grid_refinement_20seed_20260823)，状态为 `COMPLETE`，20/20 seed artifacts 完整；active source hash 与 stored source hash 同为 `d89b99c693ee7bc41c55745cf22cfdc3d0b8bfefd8d2330e0f739faeb80b2c78`。

| Metric | K=101 | K=401 | Paired change |
|---|---:|---:|---:|
| Geometric mean normalized width | 1.952659 | 1.948983 | ratio 0.998117 |
| Width-ratio 95% CI | — | — | [0.997470, 0.998747] |
| Width-ratio one-sided 95% upper | — | — | 0.998657 |
| Marginal WSC | 0.923103 | 0.922370 | -0.000733 |
| Paired WSC-difference 95% CI | — | — | [-0.000976, -0.000481] |
| Mean coverage | 0.926953 | 0.926279 | -0.000674 |
| Selection available | 20/20 | 20/20 | unchanged |
| Per-seed strict fresh target met | 20/20 | 20/20 | unchanged |

机制与完整性检查：

- K401 在 20/20 seeds 都更窄，没有 dense-wider seed。
- Independent K101 weight、point 和 width parity 的最大误差为 0；LCB 最大误差为 `4.17e-7`，低于冻结的 `1e-6` float32 tolerance；selection replay 为 20/20。
- Minimum dense patient-cluster ESS 为 1945.37，maximum cap-hit rate 为 0。
- K101 与 K401 的 certificate label 相同，且都明确为 `formal=false` 的 practical patient-cluster bootstrap LCB；加密 grid 没有把它升级为 theorem-level PAC certificate。

## 判定

K401 的 width 改善是统计上稳定的，但只有

\[
1-0.998117=0.1883\%.
\]

它没有达到冻结的 0.5% production gate，且略低于早期理论筛选使用的 0.2% point-improvement threshold。因此总 gate 为 **NO-GO**。不把 K401 合并到主方法，不运行 K801，不通过降低 target 或增大 delta 来救结果，也不扩到 clinical/main suite。

旧 A--E 分解中 D→E 的 LCB width ratio 是 1.025412，对应 log-overhead 0.025095；K401 回收的 log-width 约为 `-log(0.998117)=0.001884`，只相当于旧 guard overhead 的约 7.5%，以及旧 A→E 总 log-overhead 的约 2.9%。这确认 common-grid discretization 不是旧方法保守性的主要来源。

K401 的 WSC 下降约 0.073 个百分点，同时仍显著高于 0.90。这不是 coverage failure，却再次说明加密 grid 只是让 selected radius 更靠近 certificate boundary，并没有消除有限样本 LCB margin。真正的 fixed-confidence 瓶颈仍是 certificate uncertainty 与 profiled-family结构，而不是 101-point grid。

## 边界与保留理由

该 diagnostic code 保留在 [`src/scpcp/pac_grid_refinement.py`](../src/scpcp/pac_grid_refinement.py)、[`src/scpcp/pac_grid_study.py`](../src/scpcp/pac_grid_study.py) 和 [`scripts/run_pac_grid_study.py`](../scripts/run_pac_grid_study.py)，仅用于复现 NO-GO，不是第二个 paper method，也不在生产 `run_seed` 调用链中。保留它而不是删掉，是为了让完整 artifact、source hash 和否定性结论仍可审计；唯一生产方法名仍然是 `SC-PCP`。

辅助旧 E replay audit 中，20/20 selected base indices 与冻结 E 相同，但重新训练 GPU pilot COT 后的 schedules 不是 bitwise identical，最大绝对差为 0.00273。因此该外部 replay 没有被用作核心 K101/K401 gate；核心结论只依赖同一 run 内独立重算的 K101 comparator。旧路径的 A--E 结果继续由 [`docs/conservatism_decomposition_20260823.md`](conservatism_decomposition_20260823.md) 及其冻结 artifact 独立承担。

机器可读结果：[`summary.json`](../results/work/pac_grid_refinement_20seed_20260823/summary.json) 和 [`summary.csv`](../results/work/pac_grid_refinement_20seed_20260823/summary.csv)。
