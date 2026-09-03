# SC-PCP 方法与完整 baseline 实验结果（2026-08-24）

本文档给出当前论文可以直接使用的方法说明与完整 baseline 结果。这里的
**baseline** 只指五个独立比较方法：`Standard CP`、`ACI`、`MFCS`、`SPCI` 和
`PRC`；正文方法为 `SC-PCP`。任何删除 current ratio、只保留 current ratio、
fixed-policy 等组件实验都属于 ablation，**不进入本文的 baseline 表**。Oracle、
COT 和 DR 诊断也不进入 baseline 排名。

完整机器可读结果位于
[`results/work/complete_baseline_results_20260824`](../results/work/complete_baseline_results_20260824)：

- `rq1_all_baselines.csv`：5 个主 setting × 6 个方法，共 30 行；
- `rq3_all_baselines.csv`：4 个 feedback setting × 6 个方法，共 24 行；
- `per_stage_all_baselines.csv`：全部方法的逐阶段 coverage、width 与 pointwise 95% CI，共 612 行；
- `metadata.json`：统计口径与 frozen-suite provenance。

这些文件由冻结工件
[`paper_marginal_final_20260822`](../results/work/paper_marginal_final_20260822)
重新校验并确定性导出。源套件包含 8 个 settings、480 个完整 seed 工件、2,880
条 canonical method records；每个 seed 恰好有六个方法。

2026-08-25 完成的 controlled all-six、exact finite-MDP 和 orthogonal copula
正式研究见 [`formal_experiments_20260825.md`](formal_experiments_20260825.md)。
该文档是强受控 shift 下 baseline 排名的最新来源；不得用下文较早 two-method
confirm 的数值替换新的 all-six 结果。

2026-08-26 完成的 horizon×overlap、calibration-size、propensity 与 strict-split
正式诊断见 [`formal_experiments_20260826.md`](formal_experiments_20260826.md)。
它们回答 statistical boundary 与 robustness 问题，不新增 baseline row，也不改变
本页六方法定义或任何 2026-08-22 数值。

本页冻结 CSV 的新增确定性图件为
[`figure_main_pareto.pdf`](../results/paper_main_suite_figures_20260826/figure_main_pareto.pdf)
与
[`figure_stagewise_profiles.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_stagewise_profiles.pdf)；
后者包含全部五个 production/native RQ1 settings。对应 editable exports、source
CSV 与 QA 分别位于
[`results/work/main_suite_figures_20260826`](../results/work/main_suite_figures_20260826)
和
[`results/work/five_setting_stage_profiles_20260826`](../results/work/five_setting_stage_profiles_20260826)。
完整投稿图序见 [`figure_portfolio_20260826.md`](figure_portfolio_20260826.md)。

## 1. 问题定义

考虑 \(T\) 阶段的顺序治疗过程。患者在阶段 \(t\) 到达状态 \(S_t\)，策略选择
动作 \(A_t\)，随后观测二维下一步结局 \(Y_{t+1}\)。历史数据由 logging policy
\(\mu_t(a\mid s)\) 产生。我们必须只用这些历史轨迹，在部署前选择一条阶段半径

\[
q=(q_0,\ldots,q_{T-1}).
\]

困难在于半径不只控制 prediction set 的大小，还进入治疗策略。因此，部署法则是

\[
q\longrightarrow \pi^q
\longrightarrow P_q
\longrightarrow \text{future score law},
\]

而不是一个固定 test distribution 下的普通 split-conformal 问题。目标是在由最终
半径自身诱导的 trajectory law \(P_{\widehat q}\) 下，对每个固定阶段达到

\[
\Pr_{\tau\sim P_{\widehat q}}
\{Y_{t+1}\in C_{\widehat q_t}(S_t,A_t)\}
\ge 1-\alpha,
\qquad t=0,\ldots,T-1.
\]

本文取 \(\alpha=0.10\)。

## 2. 共同 prediction score 与 prediction set

Outcome model 只在独立的 \(D_{\rm pred}\) 上拟合并冻结。它输出每个动作下的
二维 conditional mean 和 scale：

\[
\widehat\mu(s,a),\qquad \widehat\sigma(s,a).
\]

使用 normalized maximum nonconformity score

\[
R_t=
\max_{j\in\{1,2\}}
\frac{|Y_{t+1,j}-\widehat\mu_j(S_t,A_t)|}
{\widehat\sigma_j(S_t,A_t)}.
\]

半径 \(q_t\) 产生 axis-aligned joint prediction box

\[
C_{q_t}(s,a)=
\left\{y:
|y_j-\widehat\mu_j(s,a)|
\le q_t\widehat\sigma_j(s,a),\ j=1,2
\right\}.
\]

因此 \(R_t\le q_t\) 等价于二维结局同时落入 prediction box。正文 coverage 是
per-step joint outcome coverage，不是整条 episode 全部阶段同时命中的 pathwise
coverage。

## 3. Prediction-set-dependent treatment policy

所有方法部署时共享同一个 behavior-anchored target policy。对状态 \(s\)、动作
\(a\) 和候选半径 \(q\)，prediction box 内的 worst-case clinical cost 为

\[
J_q(s,a)=
w^\top\widehat\mu(s,a)+c(a)
+q\,w^\top\widehat\sigma(s,a).
\]

令

\[
d_q(s,a)=J_q(s,a)-\min_{a'}J_q(s,a'),
\qquad
u_q(s,a)=\max\{\exp[-(\eta/\tau)d_q(s,a)],10^{-12}\}.
\]

实现没有直接对 exponential tilt 归一化，而是投影到单步 density-ratio 上界。
对每个 \((s,q)\) 求 \(z(s,q)>0\)，使

\[
\sum_a\mu_{\rm ref}(a\mid s)
\min\left\{\frac{u_q(s,a)}{z(s,q)},c\right\}=1,
\]

并定义

\[
\boxed{
\pi^q(a\mid s)=
\mu_{\rm ref}(a\mid s)
\min\left\{\frac{u_q(s,a)}{z(s,q)},c\right\}.
}
\]

主设置为 \(\eta=\tau=1\)、单步 ratio cap \(c=10\)、propensity floor
0.01。这里的 cap 是 **target-policy 定义的一部分**，不是 calibration 后对累计
importance weights 的 clipping。

## 4. SC-PCP：causal committed-prefix transport

### 4.1 为什么必须使用完整 prefix

到达阶段 \(t\) 时，过去半径 \(q_{0:t-1}\) 已经改变了此前动作和当前状态分布；
当前候选 \(r\) 又改变 \(A_t\)，而 \(R_t\) 在 \(A_t\) 之后产生。因此 stage-\(t\)
score event 的 likelihood ratio 必须同时包含历史动作与当前动作：

\[
\boxed{
W_{i,t}(r;q_{0:t-1})=
\prod_{h=0}^{t-1}
\frac{\pi_h^{q_h}(A_{ih}\mid S_{ih})}
{\mu_h(A_{ih}\mid S_{ih})}
\cdot
\frac{\pi_t^{r}(A_{it}\mid S_{it})}
{\mu_t(A_{it}\mid S_{it})}.
}
\]

当前动作 ratio 不能省略；未来动作 ratio 则不能提前加入。在 consistency、
sequential exchangeability、nonanticipation 和 positivity 下，trajectory density
中的 transition/outcome kernels 相消，得到

\[
C_t(q_{0:t})
=\Pr_{P_q}(R_t\le q_t)
=\mathbb E_\mu
\left[W_t(q_t;q_{0:t-1})\mathbf 1\{R_t\le q_t\}\right].
\]

这就是 committed-prefix identification。

### 4.2 Hájek coverage 与 width surfaces

令

\[
D_{\rm cal}=D_{\rm COT}\cup D_{\rm cert}.
\]

旧名称只表示数据角色；最终方法不训练 COT，也不把 \(D_{\rm cert}\) 当成 PAC
certificate set。对当前候选 \(r\)，SC-PCP 使用 self-normalized Hájek estimate

\[
\widehat C_t(r;q_{0:t-1})=
\frac{
\sum_{i\in D_{\rm cal}}W_{i,t}(r;q_{0:t-1})
\mathbf 1\{R_{it}\le r\}}
{\sum_{i\in D_{\rm cal}}W_{i,t}(r;q_{0:t-1})}.
\]

令 \(s_{Y,j}\) 是 \(D_{\rm pred}\) 中第 \(j\) 个 outcome 的标准差，并定义

\[
B_{it}=\frac12\sum_{j=1}^2
\frac{2\widehat\sigma_j(S_{it},A_{it})}{s_{Y,j}}.
\]

对应的 target-policy normalized width estimate 是

\[
\widehat{\mathcal W}_t(r;q_{0:t-1})=
\frac{\sum_iW_{i,t}(r;q_{0:t-1})\,rB_{it}}
{\sum_iW_{i,t}(r;q_{0:t-1})}.
\]

实现用 float64 保存 raw cumulative log weights。每个 candidate column 只减去本列
最大 log weight 后再 exponentiate；共同倍数从 Hájek 分子、分母和 ESS 中消去。
所以准确表述是：

> uncapped cumulative committed-prefix weights under a structurally ratio-capped target policy.

### 4.3 Free stagewise greedy selector

\(D_{\rm COT}\) 单独把每阶段 score 的 0.50--0.999 empirical quantiles 冻结为
\(K=101\) 个候选：

\[
\mathcal G_t=\{r_{t1},\ldots,r_{tK}\}.
\]

给定已承诺的 \(\widehat q_{0:t-1}\)，可行集合为

\[
\mathcal A_t(\widehat q_{0:t-1})=
\left\{r\in\mathcal G_t:
\widehat C_t(r;\widehat q_{0:t-1})\ge0.90
\right\}.
\]

选择规则是

\[
\boxed{
\widehat q_t\in
\arg\min_{r\in\mathcal A_t(\widehat q_{0:t-1})}
\widehat{\mathcal W}_t(r;\widehat q_{0:t-1}).
}
\]

代码扫描全部 101 个候选，不假定 coverage 或 width 随 radius 单调。选中后提交
\(\widehat q_t\) 及其 raw log-prefix weight，再进入 \(t+1\)。若某阶段没有可行
candidate，返回 unavailable；不会用最大半径兜底。

最终输出是自由的 \(T\) 维 schedule，而不是 profile family \(q_t=s b_t\)。若有
\(N\) 条 calibration trajectories 和 \(A\) 个动作，candidate evaluation 复杂度是

\[
O(NTKA).
\]

“每阶段扫描 \(K\) 个候选”只描述本 greedy rule 的工作量；它不是把同一个全局
\(K^T\) schedule optimization objective 精确约化为 \(TK\)。算法也不声称全局
最小化所有未来阶段的总 width。

完整执行顺序可以概括为：

1. 在 patient-disjoint \(D_{\rm pred}\) 上拟合并冻结 outcome model；clinical
   setting 也在该 role 的动作标签上拟合并冻结 logging propensity nuisance。
2. 在 \(D_{\rm COT}\) 和 \(D_{\rm cert}\) 上计算 normalized maximum scores。
3. 只用 \(D_{\rm COT}\) 为每个 stage 构造 101-point empirical-quantile grid。
4. 令 \(D_{\rm cal}=D_{\rm COT}\cup D_{\rm cert}\)，并把每条轨迹的初始 raw
   log-prefix weight 设为 0。
5. 在 stage \(t\)，对每个 \(r\in\mathcal G_t\) 把 current-action log ratio
   加到已提交 prefix，计算 Hájek coverage、normalized width 与 ESS。
6. 在 estimated coverage 不低于 0.90 的候选中选择 estimated width 最小者；
   无可行候选则返回 unavailable。
7. 提交该 radius 及对应的 raw cumulative log weight，继续下一 stage。
8. 完整 schedule 冻结后，只在独立 fresh target-policy rollout 或 frozen clinical
   evaluator 上报告 deployment coverage 与 width。

### 4.4 理论保证与严格边界

条件于独立 \(D_{\rm pred}\) 拟合并冻结的 outcome/behavior nuisances，假设：

1. calibration patients 独立，且 patient roles 不重叠；\(T\) 固定；
2. consistency、no interference、sequential exchangeability、nonanticipation；
3. 所有 candidate policies 可访问的 history-action 上有 causal positivity；
4. synthetic logging propensity 已知；若使用 fitted propensity，则要求独立拟合且
   uniformly ratio-consistent；
5. 在完整紧 prefix-radius 类上，weighted numerator、denominator 和 width surfaces
   uniformly Glivenko--Cantelli，Hájek denominator uniformly 远离 0；
6. 完整 schedule 的 availability probability 趋于 1。

定义 uniform surface error

\[
\Delta_n=
\max_{t<T}\sup_{q_{0:t}}
|\widehat C_{n,t}(q_{0:t})-C_t(q_{0:t})|.
\]

上述条件给出 \(\Delta_n=o_p(1)\)。算法所选候选经验上可行，所以无论 width argmin
和随机 grid 怎样依赖 calibration data，都有

\[
C_t(\widehat q_{0:t})
\ge \widehat C_{n,t}(\widehat q_{0:t})-\Delta_n
\ge1-\alpha-\Delta_n.
\]

固定有限 \(T\) 下，

\[
\boxed{
\min_{t<T}C_t(\widehat q_{0:t})
\ge1-\alpha-o_p(1).
}
\]

这个 post-selection argument 依赖完整参数类上的 uniform convergence，不需要唯一
或稳定的 population selector，也不要求 exact fixed point。由于 \(D_{\rm COT}\)
既定义 grid 又进入 \(D_{\rm cal}\)，仅仅说“每次只有 101 个候选”不足以证明结果；
必须保留上述 continuum-uniform 条件。

SC-PCP **不是** finite-sample、distribution-free、PAC、data-conditional 或 exact
weighted conformal procedure。当前代码没有 weighted test-point mass、\((n+1)\)
rank correction 或针对 calibration-selected policy 的 weighted-exchangeability construction。
最准确的定义是：

\[
\boxed{
\text{conformal score family}
+\text{committed-prefix Hájek marginal calibration}.
}
\]

Canonical selector 实现在
[src/scpcp/marginal_prefix.py](../src/scpcp/marginal_prefix.py)，生产集成在
[src/scpcp/experiment.py](../src/scpcp/experiment.py)。

## 5. Baselines：独立比较方法，不是 ablations

### 5.1 共同接口与信息预算

五个 baselines 与 SC-PCP 共享 patient split、outcome model、score、rectangular set、
radius-dependent policy 和 50,000 条 fresh evaluation trajectories。主要信息预算如下。

| Method | 类型 | Target-policy adaptation trajectories | Rounds | Fresh evaluation |
|---|---|---:|---:|---:|
| Standard CP | offline | 0 | 0 | 50,000 |
| ACI | online adapter | 2,000 | 3 | 50,000 |
| MFCS | offline | 0 | 0 | 50,000 |
| SPCI | online adapter | 2,000 | 3 | 50,000 |
| PRC | online adapter | 2,000 | 3 | 50,000 |
| SC-PCP | offline | 0 | 0 | 50,000 |

ACI、SPCI、PRC 各自独立获得 2,000 条 target-policy trajectories，按
667/667/666 分成三轮；并非三者共享 2,000 条。因此主表比较最终系统表现，但
不是严格 equal-information estimator comparison。

### 5.2 Standard CP

Standard CP 忽略 deployment-induced shift。对 \(D_{\rm cal}\) 中 stage-\(t\) 的
\(n\) 个 scores，使用 split-conformal rank

\[
k=\min\{n,\lceil(n+1)(1-\alpha)\rceil\},
\qquad q_t^{\rm CP}=R_{t,(k)}.
\]

它输出自由的 stagewise vector，不使用 target feedback、profile 或 101-point grid。

### 5.3 ACI

ACI 为每阶段维护 history \(H_t^{(r)}\) 和 miscoverage controller
\(\alpha_t^{(r)}\)：

\[
q_t^{(r)}=Q_{1-\alpha_t^{(r)}}(H_t^{(r)}),
\]

\[
\alpha_t^{(r+1)}=
\operatorname{clip}_{[0.001,0.999]}
\left[\alpha_t^{(r)}+\gamma(\alpha-e_t^{(r)})\right].
\]

初始 history 使用完整 \(D_{\rm cal}\)，此后每阶段最多保留最近 10,000 个 scores。
主设置为 \(\gamma=0.01\)、3 rounds、2,000 target trajectories。它是 patient-batch、
stagewise adapter，不直接继承原始逐样本 ACI 的长期频率 theorem。

### 5.4 MFCS

MFCS 使用 \(D_{\rm COT}\) 冻结的 profile-scale family \(q_t(s)=s b_t\) 和 101-point
scale grid。对候选 \(s_k\)，使用 depth \(d=3\)、cap \(B=40\) 的有限历史权重

\[
W_{it}^{s_k,d}=
\min\left\{B,
\prod_{r=\max(0,t-d+1)}^t
\frac{\pi_{s_kb_r,r}(A_{ir}\mid S_{ir})}
{\mu_r(A_{ir}\mid S_{ir})}
\right\}.
\]

Coverage estimator 是不 self-normalize 的 raw Horvitz--Thompson mean：

\[
\widehat C_{k,t}^{\rm MFCS}=\frac1n\sum_i
W_{it}^{s_k,d}\mathbf1\{R_{it}\le s_kb_t\}.
\]

它选择第一个 worst-stage estimate 达到 0.90 的 scale，且不使用 target-policy
adaptation trajectories。

### 5.5 SPCI

SPCI adapter 对每阶段维护最近 1,000 个 normalized maximum scores：

\[
q_t^{(r)}=Q_{0.90}(H_t^{(r)}),
\qquad
H_t^{(r+1)}=\operatorname{last}_{1000}
\left(H_t^{(r)}\cup\{R_{it}^{(r)}\}_i\right).
\]

初始化只取 \(D_{\rm cal}\) 每阶段最近的至多 1,000 个 scores，而不是完整
\(D_{\rm cal}\)。主设置为 3 rounds、2,000 target trajectories。它保持共同的
rectangular prediction set，因此是 task-aligned adapter，不是上游 ellipsoidal
MultiDimSPCI 的原样 reproduction。

### 5.6 PRC

PRC 同样使用 frozen profile-scale family。初始 scale 是能包住 Standard CP
stagewise radii 的最小值：

\[
s^{(0)}=\max_t\frac{q_t^{\rm CP}}{b_t}.
\]

每轮在 target batch 上评估 frozen scale grid；候选需同时满足

\[
\min_t\widehat C_{k,t}^{(r)}-m_r\ge1-\alpha,
\qquad
m_r=\sqrt{\frac{\log(KT/\delta)}{2n_r}},
\]

以及 \(|s_k-s^{(r)}|\le h_{\max}\)。在通过者中选择最小 scale；没有通过者就保持
当前 scale。主设置为 \(K=101,\delta=0.05,h_{\max}=0.35\)、3 rounds、2,000
target trajectories。它是 frozen-grid sequential adapter，不直接继承要求 scalar
monotone risk 的原生 PRC theorem。

Baseline adapters 的实际实现集中在
[src/scpcp/baselines.py](../src/scpcp/baselines.py)；完整接口与预算说明另见
[docs/baselines_and_settings.md](baselines_and_settings.md)。

## 6. 评价协议与统计量

### 6.1 数据与运行规模

| Setting | Horizon | Seeds | Calibration/evaluation 说明 |
|---|---:|---:|---|
| Synthetic main | 12 | 100 | 5,000 logged；50,000 fresh/method/seed |
| MIMIC-IV | 12 | 20 | patient split；50,000 frozen-environment fresh/method/seed |
| MIMIC-CXR + IV/ED | 6 | 20 | patient split；50,000 frozen-environment fresh/method/seed |
| eICU | 12 | 20 | patient split；50,000 frozen-environment fresh/method/seed |
| INSPIRE | 12 | 20 | patient split；50,000 frozen-environment fresh/method/seed |
| RQ3 synthetic \(\beta\in\{0,.5,1,2\}\) | 12 | 100 each | \(\beta=1\) 复用 Synthetic main |

Synthetic patient roles 为 40/20/40 的
\(D_{\rm pred}/D_{\rm COT}/D_{\rm cert}\)。Clinical 为 40/15/30/15 的
\(D_{\rm pred}/D_{\rm COT}/D_{\rm cert}/D_{\rm env}\)。所有划分均按 patient ID。

### 6.2 主指标

正文 coverage 指标严格为

\[
\boxed{
\widehat{\rm WSC}_{\rm marg}
=\min_t\frac1{|\mathcal S|}\sum_{s\in\mathcal S}\widehat C_{s,t}.
}
\]

它不是 \(|\mathcal S|^{-1}\sum_s\min_t\widehat C_{s,t}\)。MeanCov 是 seed 与阶段
上的总体均值；Average normalized width 是每坐标宽度按 \(D_{\rm pred}\) outcome
scale 归一化后的 patient-stage 平均。

WSC 的 95% CI 使用 10,000 次 complete-seed-vector percentile bootstrap；每次
重采完整的 \(T\)-vector，再重新取 \(\min_t\)。MeanCov 和 width 使用跨 seed 的
Student-\(t\) interval。Selection Rate 使用全部预定 runs 的 Wilson interval。

Point eligibility 规则为

\[
\widehat{\rm WSC}_{\rm marg}\ge0.90
\quad\text{且}\quad
\widehat{\rm SelectionRate}\ge0.95.
\]

这是声明的点估计排名规则，不是 95% coverage certificate。

## 7. RQ1：完整六方法主结果

表中每项为 point estimate `[95% CI]`。所有方法均完成全部 runs；Synthetic 的
Selection Rate 为 100/100，Wilson 95% CI 为 [96.30%, 100%]；每个 clinical
setting 为 20/20，Wilson 95% CI 为 [83.89%, 100%]。

| Setting | Method | WSC [95% CI] | MeanCov [95% CI] | Width [95% CI] | Eligible |
|---|---|---:|---:|---:|:---:|
| Synthetic | Standard CP | 0.8993 [0.8981, 0.8996] | 0.9001 [0.8998, 0.9004] | 1.831 [1.828, 1.835] | No |
| Synthetic | ACI | 0.8996 [0.8987, 0.8998] | 0.9002 [0.9000, 0.9004] | 1.832 [1.828, 1.835] | No |
| Synthetic | MFCS | 0.9138 [0.9118, 0.9144] | 0.9151 [0.9143, 0.9159] | 1.907 [1.902, 1.913] | Yes |
| Synthetic | SPCI | 0.8983 [0.8961, 0.8987] | 0.8998 [0.8993, 0.9004] | 1.831 [1.827, 1.835] | No |
| Synthetic | PRC | 0.9106 [0.9086, 0.9111] | 0.9119 [0.9111, 0.9126] | 1.890 [1.886, 1.895] | Yes |
| Synthetic | **SC-PCP** | **0.9018 [0.9006, 0.9020]** | 0.9027 [0.9024, 0.9030] | **1.844 [1.840, 1.847]** | **Yes** |
| MIMIC-IV | Standard CP | 0.8983 [0.8952, 0.9014] | 0.9107 [0.9092, 0.9122] | 2.146 [2.113, 2.178] | No |
| MIMIC-IV | ACI | 0.8988 [0.8966, 0.9011] | 0.9087 [0.9075, 0.9100] | 2.111 [2.081, 2.142] | No |
| MIMIC-IV | MFCS | 0.9079 [0.9040, 0.9114] | 0.9180 [0.9163, 0.9197] | 2.291 [2.257, 2.326] | Yes |
| MIMIC-IV | SPCI | 0.8971 [0.8926, 0.8988] | 0.9007 [0.8992, 0.9022] | 2.015 [1.970, 2.059] | No |
| MIMIC-IV | PRC | 0.9043 [0.9004, 0.9081] | 0.9158 [0.9141, 0.9174] | 2.243 [2.210, 2.277] | Yes |
| MIMIC-IV | **SC-PCP** | **0.9012 [0.8984, 0.9041]** | 0.9128 [0.9113, 0.9143] | **2.184 [2.153, 2.216]** | **Yes** |
| MIMIC-CXR + IV/ED | Standard CP | 0.9020 [0.8933, 0.9074] | 0.9067 [0.9015, 0.9120] | 4.749 [4.348, 5.149] | Yes |
| MIMIC-CXR + IV/ED | **ACI** | **0.9001 [0.8965, 0.9013]** | 0.9013 [0.8992, 0.9033] | **4.646 [4.264, 5.029]** | **Yes** |
| MIMIC-CXR + IV/ED | MFCS | 0.9132 [0.9032, 0.9188] | 0.9183 [0.9122, 0.9244] | 5.057 [4.630, 5.484] | Yes |
| MIMIC-CXR + IV/ED | SPCI | 0.8963 [0.8911, 0.8982] | 0.8994 [0.8974, 0.9015] | 4.613 [4.244, 4.982] | No |
| MIMIC-CXR + IV/ED | PRC | 0.9123 [0.9025, 0.9174] | 0.9176 [0.9126, 0.9227] | 5.040 [4.608, 5.471] | Yes |
| MIMIC-CXR + IV/ED | SC-PCP | 0.9040 [0.8956, 0.9089] | 0.9083 [0.9030, 0.9137] | 4.789 [4.384, 5.193] | Yes |
| eICU | Standard CP | 0.9056 [0.9001, 0.9105] | 0.9169 [0.9140, 0.9198] | 2.117 [2.073, 2.161] | Yes |
| eICU | **ACI** | **0.9037 [0.9011, 0.9061]** | 0.9108 [0.9090, 0.9127] | **2.034 [1.990, 2.078]** | **Yes** |
| eICU | MFCS | 0.9207 [0.9143, 0.9241] | 0.9280 [0.9247, 0.9313] | 2.316 [2.255, 2.376] | Yes |
| eICU | SPCI | 0.8974 [0.8915, 0.8977] | 0.8998 [0.8984, 0.9012] | 1.915 [1.870, 1.961] | No |
| eICU | PRC | 0.9167 [0.9102, 0.9217] | 0.9258 [0.9226, 0.9291] | 2.273 [2.222, 2.325] | Yes |
| eICU | SC-PCP | 0.9081 [0.9026, 0.9127] | 0.9189 [0.9161, 0.9218] | 2.153 [2.107, 2.198] | Yes |
| INSPIRE | Standard CP | 0.8984 [0.8957, 0.9005] | 0.9114 [0.9106, 0.9123] | 2.442 [2.414, 2.470] | No |
| INSPIRE | ACI | 0.8980 [0.8954, 0.8999] | 0.9098 [0.9091, 0.9105] | 2.404 [2.377, 2.431] | No |
| INSPIRE | MFCS | 0.9040 [0.9012, 0.9066] | 0.9171 [0.9158, 0.9183] | 2.604 [2.563, 2.645] | Yes |
| INSPIRE | SPCI | 0.8980 [0.8926, 0.8988] | 0.9022 [0.9010, 0.9033] | 2.305 [2.265, 2.345] | No |
| INSPIRE | PRC | 0.9031 [0.9005, 0.9055] | 0.9162 [0.9149, 0.9174] | 2.573 [2.536, 2.609] | Yes |
| INSPIRE | **SC-PCP** | **0.9010 [0.8985, 0.9028]** | 0.9133 [0.9125, 0.9141] | **2.498 [2.472, 2.524]** | **Yes** |

### 7.1 RQ1 的准确结论

1. SC-PCP 在 5/5 settings 的 WSC 点估计达到 0.90，且 180/180 RQ1 runs 均成功输出。
2. 在声明的 point-eligibility 规则下，SC-PCP 是 Synthetic、MIMIC-IV、INSPIRE
   上最窄的 eligible method；相对最窄 eligible baseline 的 paired geometric width
   reduction 分别为 2.47%、2.62%、2.88%。
3. MIMIC-CXR 和 eICU 上 ACI 更窄；SC-PCP 分别比 ACI 宽 2.97% 和 5.84%。因此
   不能写“SC-PCP 在所有数据集都 SOTA”。
4. 相对 Standard CP，SC-PCP 的 width overhead 分别为 0.67%、1.81%、0.84%、
   1.67%、2.32%。
5. Synthetic 与 eICU 的 SC-PCP WSC interval 完全高于 0.90；MIMIC-IV、
   MIMIC-CXR、INSPIRE 的 interval 跨过 0.90。这三项只能说点估计达标，不能说
   95% 证据已证明 coverage。
6. MFCS 和 PRC 的 WSC 通常更高，但同时明显更宽；这体现其保守性，不应把高于
   0.90 越多理解为越好。

## 8. RQ3：全部 baselines 的 feedback-coefficient sensitivity

\(\beta=1\) 复用 RQ1 Synthetic；其余三档各有 100 seeds。所有 24 个 method-setting
cells 的 Selection Rate 均为 100/100。

| Setting | Method | WSC [95% CI] | MeanCov [95% CI] | Width [95% CI] | Eligible |
|---|---|---:|---:|---:|:---:|
| \(\beta=0\) | Standard CP | 0.8992 [0.8981, 0.8995] | 0.9000 [0.8997, 0.9003] | 1.966 [1.961, 1.970] | No |
| \(\beta=0\) | ACI | 0.8995 [0.8986, 0.8997] | 0.9001 [0.8998, 0.9004] | 1.966 [1.962, 1.970] | No |
| \(\beta=0\) | MFCS | 0.9136 [0.9116, 0.9143] | 0.9148 [0.9139, 0.9158] | 2.047 [2.040, 2.054] | Yes |
| \(\beta=0\) | SPCI | 0.8985 [0.8960, 0.8989] | 0.9000 [0.8993, 0.9006] | 1.967 [1.962, 1.971] | No |
| \(\beta=0\) | PRC | 0.9110 [0.9090, 0.9117] | 0.9123 [0.9114, 0.9131] | 2.032 [2.026, 2.039] | Yes |
| \(\beta=0\) | **SC-PCP** | **0.9016 [0.9005, 0.9020]** | 0.9027 [0.9024, 0.9030] | **1.979 [1.975, 1.984]** | **Yes** |
| \(\beta=0.5\) | Standard CP | 0.8994 [0.8981, 0.8996] | 0.9001 [0.8997, 0.9004] | 1.917 [1.913, 1.921] | No |
| \(\beta=0.5\) | ACI | 0.8996 [0.8986, 0.8997] | 0.9001 [0.8999, 0.9004] | 1.917 [1.913, 1.921] | No |
| \(\beta=0.5\) | MFCS | 0.9140 [0.9120, 0.9146] | 0.9154 [0.9146, 0.9161] | 1.998 [1.993, 2.004] | Yes |
| \(\beta=0.5\) | SPCI | 0.8983 [0.8962, 0.8988] | 0.9000 [0.8994, 0.9006] | 1.918 [1.913, 1.922] | No |
| \(\beta=0.5\) | PRC | 0.9114 [0.9094, 0.9120] | 0.9128 [0.9121, 0.9135] | 1.984 [1.979, 1.990] | Yes |
| \(\beta=0.5\) | **SC-PCP** | **0.9018 [0.9006, 0.9021]** | 0.9027 [0.9024, 0.9030] | **1.930 [1.926, 1.934]** | **Yes** |
| \(\beta=1\) | Standard CP | 0.8993 [0.8981, 0.8996] | 0.9001 [0.8998, 0.9004] | 1.831 [1.828, 1.835] | No |
| \(\beta=1\) | ACI | 0.8996 [0.8987, 0.8998] | 0.9002 [0.9000, 0.9004] | 1.832 [1.828, 1.835] | No |
| \(\beta=1\) | MFCS | 0.9138 [0.9118, 0.9144] | 0.9151 [0.9143, 0.9159] | 1.907 [1.902, 1.913] | Yes |
| \(\beta=1\) | SPCI | 0.8983 [0.8961, 0.8987] | 0.8998 [0.8993, 0.9004] | 1.831 [1.827, 1.835] | No |
| \(\beta=1\) | PRC | 0.9106 [0.9086, 0.9111] | 0.9119 [0.9111, 0.9126] | 1.890 [1.886, 1.895] | Yes |
| \(\beta=1\) | **SC-PCP** | **0.9018 [0.9006, 0.9020]** | 0.9027 [0.9024, 0.9030] | **1.844 [1.840, 1.847]** | **Yes** |
| \(\beta=2\) | Standard CP | 0.8994 [0.8982, 0.8996] | 0.9001 [0.8997, 0.9004] | 1.663 [1.660, 1.666] | No |
| \(\beta=2\) | ACI | 0.8995 [0.8986, 0.8997] | 0.9002 [0.8999, 0.9004] | 1.663 [1.661, 1.666] | No |
| \(\beta=2\) | MFCS | 0.9133 [0.9113, 0.9139] | 0.9146 [0.9138, 0.9155] | 1.729 [1.724, 1.734] | Yes |
| \(\beta=2\) | SPCI | 0.8976 [0.8957, 0.8987] | 0.8999 [0.8994, 0.9005] | 1.664 [1.660, 1.667] | No |
| \(\beta=2\) | PRC | 0.9105 [0.9085, 0.9111] | 0.9119 [0.9112, 0.9127] | 1.716 [1.712, 1.721] | Yes |
| \(\beta=2\) | **SC-PCP** | **0.9020 [0.9006, 0.9021]** | 0.9027 [0.9023, 0.9030] | **1.674 [1.671, 1.677]** | **Yes** |

RQ3 的正确解释是：SC-PCP 在四个系数下都保持约 0.902 的 WSC，并且是每档
最窄的 point-eligible 方法。相对 Standard CP 的 width overhead 始终约 0.7%。
但是当前 simulator 的 \(\beta=0\) 仍保留 action-to-difficulty 通道，所以这不是严格
的 no-feedback-to-strong-feedback experiment；它只支持对所列连续动力学系数的
数值稳定性，也不支持“SC-PCP 优势随 \(\beta\) 增大”。

## 9. 受控 signed benchmark：机制确认与后续 all-six extension

Frozen six-method suite 中 production-style policy response 很弱，因此不能用它证明
强 performative-treatment shift。独立的 held-out controlled benchmark 使用同一
kernel 下的 source logging policy 与 radius-dependent target policy，固定相同半径
比较 target 与 source coverage。它明确展示了以下链条：

\[
q\rightarrow\pi_q\rightarrow
\text{treatment/trajectory law}\rightarrow
\text{score Q90}\rightarrow\text{coverage drift}.
\]

下表是 2026-08-24 **two-method mechanism confirm** 的关键结果；原始工件为
[`controlled_prefix_benchmark_confirm20_20260824`](../results/work/controlled_prefix_benchmark_confirm20_20260824)。

| \(\gamma\) | Policy TV | Standard same-radius late gap | Score Q90 shift | Standard target WSC | SC-PCP target WSC | SC/Standard width |
|---:|---:|---:|---:|---:|---:|---:|
| -4 | 11.89% | -3.443 pp [-3.738, -3.204] | +19.41% [+17.44, +21.66] | 0.8615 | 0.9011 | 1.2250 [1.2016, 1.2516] |
| -2 | 8.75% | -1.775 pp [-1.915, -1.657] | +12.20% [+11.18, +13.40] | 0.8794 | 0.9011 | 1.1563 [1.1444, 1.1697] |
| 0 | 4.20% | +0.131 pp [+0.077, +0.186] | -0.92% [-1.36, -0.49] | 0.8984 | 0.9000 | 1.0064 [1.0022, 1.0105] |
| +2 | 2.13% | +0.831 pp [+0.720, +0.972] | -6.13% [-7.15, -5.36] | 0.9064 | 0.9013 | 0.9671 [0.9613, 0.9720] |
| +4 | 1.87% | +1.155 pp [+0.939, +1.447] | -8.67% [-11.00, -6.98] | 0.9102 | 0.9013 | 0.9527 [0.9408, 0.9622] |

这项结果确实体现了 performative-treatment shift：在负向机制下，Standard CP
使用相同 historical radius 时损失 1.8--3.4 个 coverage points；在正向机制下
变得保守。SC-PCP 不是单纯加宽：负 shift 时扩大集合，正 shift 时比 Standard
窄 3.3%--4.7%，同时 WSC 保持在约 0.90。

该 benchmark 是 calibration-aligned semi-synthetic mechanism confirmation，不是
自然 ICU treatment-effect evidence。随后完成的 fresh all-six artifact
`results/work/controlled_six_method_confirm20_20260825` 已在同类受控 signed 环境中
加入全部五个 baseline。其 \(\gamma=-4,-2\) SC-PCP WSC 为 0.898277 和
0.897367，相对 Standard CP 提高 3.46 pp 和 1.86 pp，但仍略低于 0.90；MFCS
coverage 更高但明显更宽。完整 30 行主表、25 个 paired comparisons 与信息预算见
[`formal_experiments_20260825.md`](formal_experiments_20260825.md)。因此可以写强 shift
下的完整 trade-off，不能写 SC-PCP 在所有强度、所有 baseline 上 universal SOTA。

## 10. 当前最稳妥的论文结论

可以支持：

1. SC-PCP 解决的是 prediction radius 决定 sequential target policy 时的离线
   committed-prefix marginal calibration，而不是固定 target policy 的普通 CP。
2. 在完整 five-setting baseline suite 中，SC-PCP 5/5 point-WSC 达标、5/5 位于
   point-estimate coverage--width Pareto frontier，并在 3/5 settings 取得最小
   eligible width。
3. 在独立 controlled signed benchmark 中，performative-treatment mechanism 会让
   Standard CP 产生大幅双向 coverage drift；SC-PCP 在负 shift 下实质纠偏并在正
   shift 下缩窄 width，但 fresh all-six 负向 WSC 仍有约 0.17--0.26 pp residual
   undercoverage。
4. SC-PCP 完全离线；它与获得 2,000 条 target feedback 的 ACI/SPCI/PRC 比较时，
   仍需披露信息预算差异。

不能支持：

1. universal SOTA 或“每个数据集都优于所有 baseline”；
2. finite-sample、distribution-free、PAC 或 exact weighted-conformal validity；
3. 自然 ICU 数据中存在强 performative causal treatment effect；
4. 用较早 two-method artifact 的约 0.9011 替换 fresh all-six 的 0.8983/0.8974，
   或据此宣称强-shift universal SOTA；
5. 把任意 ablation、oracle、COT 或 DR diagnostic 混入 baseline 主表。
