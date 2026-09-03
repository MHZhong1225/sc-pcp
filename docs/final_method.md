# 最终方法：Committed-prefix Marginal SC-PCP

**规范名称。** SC-PCP 是 **uncapped committed-prefix importance-weighted
marginal calibration with free stagewise radii**。这里的每个修饰词都有明确含义：
`uncapped` 指累计 trajectory likelihood-ratio product 在估计端不做终端截断；
`committed-prefix` 指过去阶段一经选择便固定，并决定当前候选对应的部署分布；
`importance-weighted marginal calibration` 指用 logging trajectories 估计 target-policy
下每个固定阶段的边际 coverage；`free stagewise radii` 指
\(q_0,\ldots,q_{T-1}\) 分别选择，而不是共享一个 global scale。

本文档定义仓库中唯一的 SC-PCP 主方法。它面向这样一类顺序预测问题：预测集不仅要覆盖下一步结局，还会进入决策规则，进而改变动作、后续状态和未来待预测数据的分布。

我们的目标不是让所有阶段的平均 coverage 看起来接近 0.90，而是对每个固定阶段 \(t\) 都满足

\[
\Pr_{\tau\sim P_{\widehat q}}
\{Y_{t+1}\in C_{\widehat q_t}(S_t,A_t)\}
\ge 1-\alpha,
\qquad t=0,\ldots,T-1.
\]

最终方法直接学习自由的阶段半径

\[
\widehat q=(\widehat q_0,\ldots,\widehat q_{T-1}),
\]

不再把它限制成 \(q_t=s b_t\)，也不再用 coverage LCB、COT 或 ordered-IUT 选择一个 global scale。那些实现只保留在诊断代码中，不生成正文的 `SC-PCP` 结果。

### 一眼看懂方法在做什么

难点并非“从历史 residual 中取哪个分位数”，而是**候选预测集会改变动作，动作又会改变之后看到的数据**。因此，每个候选半径都对应一个候选部署 policy，也对应一个不同的 target score distribution。SC-PCP 在每个阶段重复下面的闭环：

1. 从 logging policy 产生的完整患者轨迹中读取当前 states、actions 和 scores；
2. 把候选半径 \(r\) 放入决策规则，得到候选诱导的当前 target policy；
3. 将已经承诺的历史 action ratios 与候选的**当前 action ratio**相乘，把当前 score event transport 到该候选的 target law；
4. 在同一组累计权重下估计 target-policy coverage 和 normalized width；
5. 在 coverage 达到 \(1-\alpha\) 的候选中选择 width 最小者；
6. 提交该半径和对应的未截断 log-prefix，再进入下一阶段；
7. 如果任何阶段没有可行候选，则明确返回 unavailable，而不是用最大半径掩盖失败。

这条流程同时说明了方法的两个关键边界。第一，它是按因果时间顺序构造 schedule 的 greedy rule，不声称求解全局 \(K^T\) schedule optimization。第二，方法只在能够输出完整 schedule 时返回预测集；availability 是结果的一部分，不能从汇总中删除。

### \(\gamma\) 属于实验环境，不属于方法

带符号反馈实验中的 \(\gamma\) 控制预测集通过环境反馈影响后续状态或结局的方向与强度。它是 data-generating/deployment environment 的参数，**不是** SC-PCP 的校准超参数，也不是算法选择的阶段半径。正文实验以 \(\gamma=-4\) 为默认、预先指定的主反馈设置，其他 \(\gamma\) 只用于敏感性分析；改变 \(\gamma\) 会改变评价环境，却不改变下面的 SC-PCP 定义和选择规则。

## 1. 为什么问题不是一个普通 quantile

在第 \(t\) 个阶段，冻结的 outcome model 给出二维均值和尺度

\[
\widehat\mu(S_t,A_t),\qquad \widehat\sigma(S_t,A_t),
\]

并定义 normalized max score

\[
R_t=
\max_{j\in\{1,2\}}
\frac{|Y_{t+1,j}-\widehat\mu_j(S_t,A_t)|}
{\widehat\sigma_j(S_t,A_t)}.
\]

半径 \(q_t\) 对应 prediction box

\[
C_{q_t}(s,a)=
\left\{y:
|y_j-\widehat\mu_j(s,a)|
\le q_t\widehat\sigma_j(s,a),\ j=1,2
\right\}.
\]

如果半径只影响集合大小，普通 split CP 的阶段 quantile 已经足够。本任务的关键在于，集合还改变行为锚定的 target policy。对于线性临床代价，策略使用 prediction box 内的 worst-case cost：

\[
J_{q_t}(s,a)=
w^\top\widehat\mu(s,a)+c(a)
+q_t\,w^\top\widehat\sigma(s,a),
\]

代码并不是直接归一化 exponential tilt，而是把它投影到预先指定的单步 density-ratio 上界。令

\[
d_q(s,a)=J_q(s,a)-\min_{a'}J_q(s,a'),
\qquad
u_q(s,a)=\max\{\exp[-\kappa d_q(s,a)],10^{-12}\},
\]

并令 \(c\ge1\) 为 `policy_ratio_cap`。实现对每个 \((s,q)\) 求一个 \(z(s,q)>0\)，使

\[
\sum_a\mu_{\rm ref}(a\mid s)
\min\left\{\frac{u_q(s,a)}{z(s,q)},c\right\}=1,
\]

然后定义

\[
\boxed{
\pi^q(a\mid s)=
\mu_{\rm ref}(a\mid s)
\min\left\{\frac{u_q(s,a)}{z(s,q)},c\right\}.
}
\]

因此，policy ratio cap 是部署策略定义的一部分，不是校准结束后对 importance weights 的 clipping。当 policy anchor 与 transport denominator 是同一个真实或一致估计的 logging policy 时，单步比满足

\[
0<\pi^q(a\mid s)/\mu(a\mid s)\le c.
\]

因此，改变 \(q_t\) 会改变当前动作分布；此前半径还会通过此前动作改变当前状态分布。我们真正需要校准的是

\[
F_t(r;q_{0:t-1})
=
\Pr_{P_{(q_{0:t-1},r)}}(R_t\le r),
\]

而不是 logging distribution 下的
\(\Pr_\mu(R_t\le r)\)。这也是为什么各阶段不能被当成彼此独立的 weighted quantile 问题。

## 2. Causal committed prefix

历史数据由 logging policy
\(\mu_t(a\mid s)\) 产生。给定已经承诺的前缀
\(q_{0:t-1}\) 和第 \(t\) 阶段的候选半径 \(r\)，第 \(i\) 条轨迹在当前 score event 上的 causal likelihood ratio 是

\[
W_{i,t}(r;q_{0:t-1})
=
\prod_{h=0}^{t-1}
\frac{\pi_h^{q_h}(A_{ih}\mid S_{ih})}
{\mu_h(A_{ih}\mid S_{ih})}
\cdot
\frac{\pi_t^{r}(A_{it}\mid S_{it})}
{\mu_t(A_{it}\mid S_{it})}.
\]

最后一个当前动作比不能省略：\(R_t\) 在 \(A_t\) 之后观测，当前动作已经影响 \(Y_{t+1}\)。在 consistency、sequential exchangeability 和 positivity 下，

\[
\mathbb E_\mu\!\left[
W_t(r;q_{0:t-1})\mathbf 1\{R_t\le r\}
\right]
=F_t(r;q_{0:t-1}).
\]

这给出一个自然的顺序结构。到达阶段 \(t\) 时，过去的半径已经确定，未来半径不会影响当前 score；因此，本文定义的 greedy committed-prefix rule 只扫描当前阶段的候选。这里利用的是当前 event 的 nonanticipation，并不是把同一个全局 \(K^T\) schedule-optimization problem 等价地约化为 \(TK\) 个候选。选中 \(q_t\) 后，把对应的**未截断** log likelihood ratio 加入 prefix，再进入阶段 \(t+1\)。

这种 committed-prefix 处理有两个目的。第一，它尊重顺序决策的因果方向；第二，它避免先独立校准 \(T\) 个阶段、再假装这些独立解共同诱导同一条部署分布。

## 3. Target-policy coverage 与 width

令 \(D_{\mathrm{cal}}=D_{\mathrm{COT}}\cup D_{\mathrm{cert}}\)。这两个旧名称只表示固定的数据角色；最终方法不再把 \(D_{\mathrm{cert}}\) 用作独立 PAC certification set。对阶段 \(t\) 的候选 \(r\)，我们用 Hájek self-normalization 估计 target-policy coverage：

\[
\widehat F_t(r;q_{0:t-1})
=
\frac{
\sum_{i\in D_{\mathrm{cal}}}
W_{i,t}(r;q_{0:t-1})
\mathbf1\{R_{it}\le r\}
}{
\sum_{i\in D_{\mathrm{cal}}}
W_{i,t}(r;q_{0:t-1})
}.
\]

实现始终保留 raw cumulative log weights。对每一个 candidate column，只减去该列的最大 log weight 后再指数化：

\[
\widetilde W_{i,t}(r)
=
\exp\{\log W_{i,t}(r)-\max_j\log W_{j,t}(r)\}.
\]

共同缩放同时从 Hájek 分子和分母中消去，也不改变 ESS；它只防止数值溢出，不是 weight clipping。最终主方法对**累计 committed-prefix product** 不做 terminal clipping；但 target policy 本身有上面定义的单步 ratio cap。准确表述是：

> uncapped cumulative committed-prefix weights under a structurally ratio-capped target policy.

这一设计不会在估计阶段把 target law 换成一个事后截断 law，但 horizon 增长时累计权重仍可能快速退化。

对同一个候选，normalized width 的估计为

\[
\widehat{\mathcal W}_t(r;q_{0:t-1})
=
\frac{\sum_i W_{i,t}(r;q_{0:t-1})\,B_{it}(r)}
{\sum_i W_{i,t}(r;q_{0:t-1})},
\]

其中

\[
B_{it}(r)=
r\cdot
\frac{1}{2}
\sum_{j=1}^2
\frac{2\widehat\sigma_j(S_{it},A_{it})}
{\operatorname{SD}_{D_{\rm pred}}(Y_j)}.
\]

这使不同 outcome 量纲在同一数据集内可比较，也让 selection 直接优化正文所报告的效率量。

## 4. 每阶段选择规则

每个阶段使用一个 \(K=101\) 的候选 grid

\[
\mathcal G_t=\{r_{t,1},\ldots,r_{t,K}\}.
\]

Grid 是 \(D_{\mathrm{COT}}\) 阶段 scores 的 empirical quantiles，概率范围固定为 \([0.50,0.999]\)。\(D_{\mathrm{COT}}\) 随后也进入 \(D_{\mathrm{cal}}\)，所以 grid 并不独立于 selection evidence。渐近论证必须依赖完整紧半径类上的 uniform convergence，不能把这 101 个随机候选误写成独立预冻结的有限类。对于已经承诺的 prefix，定义可行集合

\[
\mathcal A_t(q_{0:t-1})
=
\left\{r\in\mathcal G_t:
\widehat F_t(r;q_{0:t-1})\ge1-\alpha
\right\}.
\]

随后选择

\[
\boxed{
\widehat q_t
\in
\arg\min_{r\in\mathcal A_t(\widehat q_{0:t-1})}
\widehat{\mathcal W}_t(r;\widehat q_{0:t-1}).
}
\]

我们扫描整个 grid，而不是在第一次 crossing 处停止。原因很简单：半径会改变 policy 和访问到的 state distribution，所以 target-policy coverage 和 width 都不必随 \(r\) 单调。若有多个 width 完全相同的候选，代码按 grid 顺序取第一个；若没有候选达到 target，方法返回 unavailable，不用最大半径兜底。

选中
\(\widehat q_t\) 后，将对应 raw log weight 提交给下一阶段。若轨迹数为 \(N\)、动作数为 \(A\)、每阶段候选数为 \(K\)，candidate evaluation 的时间复杂度为 \(O(NTKA)\)，另加 outcome-scale inference；“\(TK\)”只表示被评价的 candidate columns 数量。该 greedy rule 没有求解一个全局 \(K^T\) objective，也不与全局 schedule optimization 等价。

## 5. 数据角色与完整算法

所有划分先按 patient ID 完成，同一患者不跨角色。

| Role | 最终用途 |
|---|---|
| \(D_{\rm pred}\) | 拟合并冻结 outcome mean/scale model；logging policy 未知时也在这里拟合并校准 behavior nuisance |
| \(D_{\rm COT}\) | 冻结每阶段 101-point candidate grid；也作为 \(D_{\rm cal}\) 的一部分 |
| \(D_{\rm cert}\) | 只作为 \(D_{\rm cal}\) 的另一部分，不再构造 LCB 或 PAC certificate |
| \(D_{\rm env}\) | 只建立 clinical frozen controlled evaluator，不参与 calibration |

Synthetic 使用 40/20/40 的 \(D_{\rm pred}/D_{\rm COT}/D_{\rm cert}\) 划分。Clinical 使用 40/15/30/15 的 \(D_{\rm pred}/D_{\rm COT}/D_{\rm cert}/D_{\rm env}\) 划分。Standard CP 也使用相同的 \(D_{\rm cal}\)，避免 SC-PCP 的改善来自更多 calibration trajectories。

~~~text
Algorithm: Committed-prefix Marginal SC-PCP

Inputs:
  patient-level D_pred, D_COT, D_cert
  target 1-alpha and K-point stage grids
  frozen outcome model, logging policy, target-policy rule

1. Fit and freeze the outcome model on D_pred.
2. If the logging policy is unknown, fit and freeze its calibrated nuisance
   model using the prespecified D_pred action labels.
3. Compute normalized max scores on D_COT and D_cert.
4. Build one K-point empirical-quantile grid per stage from D_COT only.
5. Concatenate D_cal = D_COT union D_cert and initialize every trajectory's
   raw log-prefix weight to zero.
6. For t = 0,...,T-1:
     a. For every r in G_t, append the current action log ratio to the
        already committed raw prefix.
     b. Compute the log-stabilized Hajek coverage, normalized width, and ESS.
     c. Among candidates with estimated coverage >= 1-alpha, choose the one
        with minimum estimated normalized width.
     d. If no candidate is feasible, return unavailable and the failure stage.
     e. Commit the selected radius and its uncapped raw log-prefix weight.
7. Return q_hat=(q_hat_0,...,q_hat_{T-1}).
8. Evaluate the frozen schedule only on the independent target-policy rollout
   stream or frozen clinical controlled evaluator.
~~~

## 6. 保证边界

这里必须把统计目标说准确。最终 SC-PCP 是一个 **plug-in、渐近的 per-step marginal calibration procedure**，不是 finite-sample distribution-free split CP，也不是 95% PAC/data-conditional certificate。

对于随机 calibration data \(D\)，算法输出
\(\widehat q(D)\)。正文的 marginal per-step estimand 是

\[
\overline C_t
=
\mathbb E_D\left[
\Pr_{\tau\sim P_{\widehat q(D)}}
\{R_t\le\widehat q_t(D)\}
\right],
\]

主 coverage 指标是

\[
\boxed{
\operatorname{MarginalWSC}=\min_t\overline C_t.
}
\]

### 6.1 假设

下面的 claim 条件于只用独立 \(D_{\rm pred}\) 拟合并冻结的 nuisance \(\widehat\eta\)，包括 outcome model、score scale 和 fitted behavior propensity。

1. \(D_{\rm cal}\) 由 \(n\) 条独立患者轨迹组成；同一患者不跨数据角色，轨迹内部允许任意时间依赖，\(T\) 固定。
2. Consistency、no interference、sequential exchangeability 和 nonanticipation 成立；给定充分状态或 history 与当前动作后，部署 policy 不直接改变 outcome/transition kernel。
3. 在所有 candidate policies 可访问的 history-action 上存在 causal positivity。数值 probability floor 不能替代这一真实支持假设。
4. Synthetic 使用真实 logging propensity；若使用 fitted \(\widehat\mu\)，要求在独立 \(D_{\rm pred}\) 上对所有相关 history-action uniformly ratio-consistent。
5. 存在确定的紧半径类 \(\mathcal Q_t\)，经验 grids 以概率趋于一落入该类；prefix-weighted numerator、denominator 和 width 函数类在完整

   \[
   \mathcal Q_{0:t}=\mathcal Q_0\times\cdots\times\mathcal Q_t
   \]

   上 uniformly Glivenko--Cantelli。
6. 完整 schedule 的 availability probability 趋于一；否则定理只在返回 schedule 的事件上成立，不能静默丢掉 unavailable seeds。

固定 \(T\)、有限动作、紧半径范围、policy 对 \(q\) 的适当连续性和结构性单步 ratio cap 为第 5 条提供一个有界 envelope。Clinical 中的 propensity misspecification、未观测混杂或真实 positivity failure 都会破坏 transport interpretation；其结论只能是 propensity-model-dependent 的渐近证据。

### 6.2 Identification

令真实 logging law 为 \(\mu_0\)，并定义

\[
W^0_t(q_{0:t})
=
\prod_{h=0}^{t}
\frac{\pi^{q_h}(A_h\mid S_h)}
{\mu_{0,h}(A_h\mid S_h)}.
\]

在上述 causal assumptions 下，逐阶段替换 trajectory density 中的 action kernels，transition/outcome kernels 相消，得到

\[
\boxed{
C_t(q_{0:t})
=
\Pr_{P_q}(R_t\le q_t)
=
\mathbb E_{\mu_0}\!\left[
W^0_t(q_{0:t})\mathbf1\{R_t\le q_t\}
\right].
}
\]

当前动作 ratio 必须包含在乘积中，因为 \(R_t\) 是在 \(A_t\) 后产生的；未来半径因 nonanticipation 不影响当前 event。

### 6.3 Data-dependent schedule 的 uniform validity

定义代码实际使用的 fitted-weight Hájek surface \(\widehat C_{n,t}\)，以及

\[
\Delta_n
=
\max_{0\le t<T}
\sup_{q_{0:t}\in\mathcal Q_{0:t}}
\left|
\widehat C_{n,t}(q_{0:t})-C_t(q_{0:t})
\right|.
\]

在 6.1 的条件下，uniform LLN、fitted propensity 的 uniform consistency，以及 Hájek denominator uniformly bounded away from zero 给出

\[
\Delta_n=o_p(1).
\]

在 selection available 事件上，算法的经验可行性逐阶段保证

\[
\widehat C_{n,t}(\widehat q_{0:t})\ge1-\alpha.
\]

因此不论 width argmin、grid 和 committed prefix 如何依赖 \(D_{\rm cal}\)，都有确定性链

\[
\boxed{
C_t(\widehat q_{0:t})
\ge
\widehat C_{n,t}(\widehat q_{0:t})-\Delta_n
\ge
1-\alpha-\Delta_n.
}
\]

固定有限的 \(T\) 下取最小值即可得到

\[
\min_{t<T}C_t(\widehat q_{0:t})
\ge1-\alpha-o_p(1).
\]

这个证明不需要唯一 population solution、exact fixed point、coverage 对半径单调、或 width minimizer 唯一。那些条件只在进一步声称 schedule 或 selected width 收敛时才需要。对完整连续类取 supremum 也正是处理 \(D_{\rm COT}\) grid 与 \(D_{\rm cal}\) evidence 复用的关键；仅说“每次运行只有 101 个候选”并不足以处理随机 data-dependent class。

由于 coverage 有界，availability probability 趋于一时还可推出正文 estimand

\[
\min_t\mathbb E_D[
C_t(\widehat q_{0:t})
]\ge1-\alpha-o(1).
\]

### 6.4 为什么这不是 exact weighted conformal

当前 SC-PCP 是

\[
\boxed{
\text{conformal score family}
+
\text{committed-prefix Hájek marginal calibration}.
}
\]

代码没有 test-point mass、\((n+1)\)-style rank correction、randomized tie handling，或针对 calibration-selected target policy 的 weighted-exchangeability construction。\(\widehat q\) 由 calibration data 选出，同时又决定 fresh law \(P_{\widehat q}\)；所以对预先固定 policy 成立的 weighted conformal rank theorem 不能直接复用。

删掉旧 LCB 换回了效率，也明确换成更弱的 marginal/asymptotic guarantee。不能描述成 finite-sample、distribution-free、PAC、data-conditional 或 exact conformal validity。

记录字段因此固定为：

- `selection_estimand = per_step_marginal`；
- `selection_parameter = stagewise_radii`；
- `selection_status = SELECTED_MARGINAL_POINT` 或 `UNAVAILABLE_NO_FEASIBLE_CANDIDATE`；
- `certificate_formal = false`；
- `guarantee_scope = asymptotic_per_step_marginal`。

每个 seed 中保存的 `worst_coverage=min_t C_{s,t}` 仍是有用的稳定性诊断，但它不是正文的 marginal WSC。正文先对每个固定 \(t\) 跨 seeds 求均值，再取最小值：

\[
\widehat{\operatorname{MarginalWSC}}
=
\min_t\frac1S\sum_{s=1}^{S}\widehat C_{s,t}.
\]

MeanCov 仅作为整体保守程度的补充诊断：

\[
\widehat{\operatorname{MeanCov}}
=
\frac1S\sum_{s=1}^{S}
\left(\frac1T\sum_{t=0}^{T-1}\widehat C_{s,t}\right).
\]

这一区分也解释了为什么 MeanCov 可以是 0.90，而某个固定阶段仍低于 0.90：阶段间的高 coverage 可以补偿低 coverage，但 per-step 目标不允许这种补偿。因此，正文必须以 MarginalWSC 判断 coverage 是否达标，并同时报告 MeanCov、average normalized width 和使用全部预设 runs 计算的 Selection Rate；不能用只保留 available seeds 的条件 coverage 隐藏 selection failure。

## 7. 设计取舍、贡献与限制

这版方法的核心不是“把若干现成组件拼在一起”，而是对顺序 performative calibration 做了一个结构化判断：当前阶段的 score law 只依赖已经发生的 policy prefix，因此可以边校准、边承诺，而不必一次搜索完整 schedule。

具体贡献是：

1. **把自洽 target score law 写成 committed-prefix estimand。** 当前候选的 action ratio 与所有已承诺历史 ratio 一起进入同一个 coverage event。
2. **定义因果顺序的 greedy schedule construction。** 每一步只评价当前 committed prefix 下的 \(K\) 个候选，然后把选择传给未来；它不枚举完整 schedules，也不声称等价求解一个全局 \(K^T\) objective。
3. **让 calibration 直接优化论文所关心的 coverage–width trade-off。** 可行性由 target-policy coverage 决定，候选内部由 target-policy normalized width 决定。
4. **以 raw log stabilization 保留完整 cumulative likelihood ratio。** 数值稳定不对累计 product 做 terminal clipping；单步 ratio cap 属于 target-policy estimand 本身。
5. **明确区分 marginal validity 与更强的 PAC validity。** 方法不靠把 coverage 推到 0.94 来换取一个与论文主问题不同的高概率 conditional certificate。

主要限制是：

1. 当前保证是渐近且依赖 transport assumptions，不是 distribution-free finite-sample CP。
2. Sequential greedy choice 不承诺全局最小化未来所有阶段的总 width；它利用 causal prefix 得到可执行的局部最优 schedule。
3. 长 horizon 或弱 overlap 会使 trajectory prefix weights 退化。ESS、raw log-weight span、endpoint 和 failure stage 必须随结果一起审计。
4. Clinical evaluator 是 real-data-grounded frozen controlled environment，不是真实患者部署，也不能消除 hidden confounding。
5. Axis-aligned normalized-max box 没有利用两个 outcomes 的相关结构。
6. 对弱 shift，Standard CP 可能已经达到目标并更窄；SC-PCP 的贡献是 coverage–efficiency Pareto 改善，而不是声称在所有数据集都单独最优。

最终实现位于 `src/scpcp/marginal_prefix.py`，生产调用链位于 `src/scpcp/experiment.py::run_seed`。主实验只使用 `Standard CP`、`ACI`、`MFCS`、`SPCI`、`PRC` 和 `SC-PCP` 六个名称；旧 profile/COT/LCB 路线只允许出现在明确标注的诊断或历史工件中。
