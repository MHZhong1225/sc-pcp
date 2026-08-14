# SC-PCP：逐阶段 Performative Coverage

本仓库实现一套正式方法：**profiled-scale ordered-IUT SC-PCP**。它面向 prediction set 会反过来改变治疗策略、后续状态与结局分布的顺序决策问题，并要求最终部署策略诱导的每个阶段都达到覆盖目标：

\[
\min_{0\le t<T}
P_{P_{\widehat s}}
\{Y_{t+1}\in C_{\widehat s,t}(S_t,A_t)\}
\ge 1-\alpha.
\]

旧的 shared-scalar radius、full-grid max-\(t\) selector 和 \(K\times T\times K\) DCov surface 不属于主方法，也不进入正式结果。完整定义见 [`docs/final_method.md`](docs/final_method.md)，冻结的实验协议见 [`docs/per_step_protocol.md`](docs/per_step_protocol.md)。

## 方法概览

SC-PCP 不直接搜索一个任意的 \(T\) 维半径向量。它先在独立的 \(D_{\rm COT}\) 上，从逐阶段 normalized-max scores 得到正的阶段轮廓

\[
b=(b_0,\ldots,b_{T-1}),
\qquad
\left(\prod_t b_t\right)^{1/T}=1,
\]

然后只选择一个全局尺度 \(s\)，部署半径为

\[
\boxed{q_t(s)=s b_t.}
\]

这保留了阶段难度差异，同时把最终选择限制为一个预先冻结的一维候选族。每个候选尺度都完成以下闭环：

1. 用实际阶段半径 \(s b_t\) 构造 prediction set 和 behavior-anchored target policy；
2. 用 scale-conditioned COT 学习 target 与 logging policy 下的逐阶段 state-action marginal occupancy ratio；
3. 在 untouched \(D_{\rm cert}\) 上估计每个阶段的 performative coverage lower bound；
4. 对同一候选的全部阶段做 intersection--union test（IUT）；
5. 按最宽到最窄的固定序列检验尺度，并在第一个失败处停止；
6. 只在已经通过检验的 prefix 内选择 estimated normalized width 最小的候选。

如果没有候选通过，方法返回 `UNCERTIFIED_ORDERED_IUT` 并 abstain，不用最大集合兜底。Stage multiplicity 由 IUT 处理，candidate multiplicity 由 fixed sequence 处理；主方法不使用 full-grid max-\(t\) correction，也不假设 coverage 随尺度单调。

## Evidence boundary

仓库明确区分两类证据：

- **Practical branch（连续状态与临床主实验）**：对患者级 cluster 做 2,000 次重采样，构造逐候选、逐阶段的 Hájek marginal lower bounds，再执行 ordered-IUT。该结果是 frozen transported estimate 的 sampling guard；它不控制 learned COT、fitted propensity 或 clipping 的 transport error，因此 `certificate_formal=false`。
- **Formal branch（有限 MDP 验证或外部有效误差界）**：使用 bounded raw-HT lower bounds，并扣除覆盖整个冻结 candidate-stage family 的 simultaneous transport-error bound。可完全枚举的 tabular MDP 用 population \(L_1\) discrepancy 验证这一 premise；证据仍写入同一个 `SC-PCP` record，不再生成第二条 scalar/max-\(t\) audit row。

因此，clinical coverage 是在冻结的受控评估环境中的模型化实验结果，不是对真实患者实施算法所得的临床覆盖保证。

## Patient-level data roles

所有数据角色都先按患者划分，同一患者不会跨角色：

| Role | 用途 |
| --- | --- |
| \(D_{\rm pred}\) | 拟合并冻结 outcome mean/scale model 与 behavior nuisance；两者分别读取结局与动作标签 |
| \(D_{\rm COT}\) | 冻结 stage profile 和 scale grid；训练并校准 COT |
| \(D_{\rm cert}\) | 计算 coverage evidence、候选宽度并选择最终尺度 |
| \(D_{\rm env}\) | 只建立 clinical controlled evaluator，不参与选择 |

临床数据采用 40/15/30/15 的 \(D_{\rm pred}/D_{\rm COT}/D_{\rm cert}/D_{\rm env}\) 划分，不再为 propensity 单独牺牲一个 data split。Behavior nuisance 在 \(D_{\rm pred}\) 的动作标签上拟合，并在内部患者留出集上完成 decision-time calibration；outcome model 只读取同一角色中的结局标签。临床 Track A evaluator 是 **leave-one-patient-out conditional-residual-bootstrap controlled evaluation**：donor search 排除当前患者，并把 donor residual 在当前 query 的预测均值与尺度上重新组合。它只用于冻结后的评估，不能被解释为观察到的 target-policy deployment 或真实世界因果效果。

## Paper comparison

每个完整 paper seed 都必须包含六个方法行：SC-PCP 和五个对照方法。

| Paper label | 角色 |
| --- | --- |
| `Standard CP` | 固定历史分布的逐阶段 split conformal baseline |
| `ACI stagewise adaptation` | 使用额外 on-policy trajectories 的阶段式在线适应 baseline |
| `MFCS task-adapted` | 适配本任务 score 与冻结 profile family 的有限深度 feedback baseline |
| `MultiDimSPCI task-adapted` | 使用额外 on-policy trajectories 的多结局在线适应 baseline |
| `PRC grid-adapted` | 在相同冻结 profile-scale grid 上运行的 on-policy PRC adapter |
| `SC-PCP` | 本文方法：marginal COT + profiled global scale + ordered-IUT |

`baselines/` 中的上游代码用于核对算法来源；无法直接处理本任务的二维 logged clinical trajectories 时，paper record 明确使用 `task-adapted` 或 `grid-adapted` 名称，不声称是未经修改的 native reproduction。主表同时展示六个方法，不只报告 SC-PCP；online baselines 的额外 adaptation budget 与 offline 方法的信息条件分开标注。

## 完整 `paper_v2` 实验

默认环境为 `ucp`，默认使用两张 GPU。下面的入口运行全部预设 RQ1 与 RQ3 作业；RQ2 和 RQ4 复用 manifest 中指定的冻结产物。它不是 smoke run：synthetic 主实验为 100 seeds，四个 clinical 数据集各 20 seeds，feedback stress 的每个额外 \(\beta\) 设置也使用 100 seeds。

```bash
conda run -n ucp python scripts/run_paper_suite.py \
  --sections rq1,rq3 \
  --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire \
  --devices cuda:0,cuda:1 \
  --output-root results/work/paper_v2
```

runner 会拒绝写入非空 output root，并且只有全部预设 seed 完成后才写 `COMPLETE`。完整运行结束后生成论文主表和图：

```bash
conda run -n ucp python tools/render_paper_results.py \
  --input results/work/paper_v2 \
  --output results/paper_v2
```

renderer 会 fail closed：缺少数据集、seed、六方法记录或 `COMPLETE` 标记时不会生成正式 PDF。输出包括 synthetic 主表、clinical 主表、逐阶段 coverage、feedback stress 和 self-consistent diagonal 机制图。

### Exact finite-MDP theorem validation

Tabular validation 使用同一 profiled ordered-IUT SC-PCP schema，配置为完整 200 seeds 和每个已选择 seed 50,000 条 fresh rollout：

```bash
conda run -n ucp python scripts/run_per_step.py \
  --config configs/per_step_tabular_validation.yaml \
  --devices cuda:0,cuda:1 \
  --workers-per-device 2 \
  --output-dir results/work/profiled_ordered_tabular_validation

conda run -n ucp python tools/summarize_tabular_validation.py \
  --input-root results/work/profiled_ordered_tabular_validation \
  --expected-seeds 0:200 \
  --output results/paper_v2/tabular_theorem_validation
```

旧的 scalar/max-\(t\) validation artifacts 会被 summarizer 明确拒绝，必须用当前唯一方法的配置重跑。

## 验证

```bash
conda run -n ucp pytest -q
```

核心实现位于 `src/scpcp/`；单数据集调度入口为 `scripts/run_per_step.py`，完整论文入口为 `scripts/run_paper_suite.py`，最终论文汇总入口为 `tools/render_paper_results.py`。
