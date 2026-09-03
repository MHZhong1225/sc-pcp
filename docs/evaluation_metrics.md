# SC-PCP 正文评价指标

本文档只定义最终 paper suite 使用的统计量。当前主方法是
**committed-prefix marginal SC-PCP**；所有 coverage 和 width 都在半径选择完成后，
用冻结经验环境中独立生成的 fresh target-policy trajectories 评价。

设预先指定的 seed 集合为 \(\mathcal R\)，成功输出预测集的 seed 集合为
\(\mathcal S\subseteq\mathcal R\)。每个 seed 有 \(N\) 条长度为 \(T\) 的评价轨迹，
outcome 维数为 \(d=2\)。主实验目标为
\(c_\star=1-\alpha=0.90\)。

## 1. 单个 seed 的逐时点量

在 seed \(s\)、轨迹 \(i\)、时点 \(t\)，归一化 max score 为

\[
R_{s,i,t}=\max_{1\le j\le d}
\frac{|Y_{s,i,t+1,j}-\widehat\mu_{s,i,t,j}|}
{\widehat\sigma_{s,i,t,j}}.
\]

方法在每个时点使用半径 \(q_{s,t}\)，因此联合预测集命中变量为

\[
H_{s,i,t}=\mathbf 1\{R_{s,i,t}\le q_{s,t}\}.
\]

逐时点 coverage 是

\[
\widehat C_{s,t}=\frac1N\sum_{i=1}^N H_{s,i,t}.
\]

它存于 `per_time_coverage`。这里是二维 outcome 的 joint coverage，
不是一整条轨迹所有时点同时命中的 pathwise coverage。

## 2. 正文四个指标

### 2.1 Marginal worst-step coverage（主 coverage 指标）

先在 seed 间对每个固定时点求均值，再取最弱时点：

\[
\overline C_t=\frac1{|\mathcal S|}\sum_{s\in\mathcal S}\widehat C_{s,t},
\qquad
\boxed{
\widehat{\mathrm{WSC}}_{\mathrm{marg}}
=\min_{0\le t<T}\overline C_t
}.
\]

这就是当前主结果表、WSC/MeanCov 汇总图和正文 target eligibility 使用的 WSC；
历史协议中的同名 WSC 也必须按此公式读取。
它估计的是每一个固定 stage 的 marginal coverage 是否达到 0.90。

`records.csv` 仍保存 seed-level 字段

\[
\texttt{worst\_coverage}_s=\min_t\widehat C_{s,t},
\]

但正文**不能**再计算
\(\frac1{|\mathcal S|}\sum_s\min_t\widehat C_{s,t}\) 作为主 WSC。一般而言，

\[
\frac1{|\mathcal S|}\sum_s\min_t\widehat C_{s,t}
\le
\min_t\frac1{|\mathcal S|}\sum_s\widehat C_{s,t},
\]

因为每个 seed 的最差时点可能不同。seed-level `worst_coverage` 和
`worst_gap` 只保留为运行诊断。

方向：需要达到 0.90；超过目标并非越大越好，因为通常会增加宽度。

### 2.2 Mean coverage

先在每个 seed 内跨时点平均，再在 selected seeds 间平均：

\[
\boxed{
\widehat{\mathrm{MeanCov}}
=\frac1{|\mathcal S|}\sum_{s\in\mathcal S}
\left(\frac1T\sum_{t=0}^{T-1}\widehat C_{s,t}\right)
}.
\]

seed-level 输出字段是 `average_coverage`。MeanCov 描述整体覆盖水平，
但不能替代 WSC：很高的其他时点 coverage 可能掩盖一个失败 stage。

### 2.3 Average normalized width

对 outcome 第 \(j\) 维，用该 seed 的训练 outcome scale
\(\sigma^{\mathrm{out}}_{s,j}\) 消除量纲。单个患者–时点的平均坐标宽度为

\[
W_{s,i,t}=\frac1d\sum_{j=1}^d
\frac{2q_{s,t}\widehat\sigma_{s,i,t,j}}
{\sigma^{\mathrm{out}}_{s,j}}.
\]

seed-level 与正文汇总分别为

\[
W_s=\frac1{NT}\sum_{i=1}^N\sum_{t=0}^{T-1}W_{s,i,t},
\qquad
\boxed{
\overline W=\frac1{|\mathcal S|}\sum_{s\in\mathcal S}W_s
}.
\]

输出字段是 `average_normalized_width`；逐时点向量是
`per_time_normalized_width`。方向：在 coverage 达标的前提下越小越好。
它是归一化坐标长度的平均，不是面积或 log-volume。

### 2.4 Selection Rate

令 \(A_s=1\) 表示该 seed 的方法成功输出完整的逐时点半径，则

\[
\boxed{
\widehat{\mathrm{SelectionRate}}
=\frac1{|\mathcal R|}\sum_{s\in\mathcal R}A_s
}.
\]

输出字段是 `selection_available`。Selection Rate 使用全部预设 seeds；
WSC、MeanCov 和 width 只在 \(\mathcal S\) 上计算。因此表注必须同时给出
`n_selected / n_runs`，不能用条件汇总隐藏 abstention。

## 3. 95% uncertainty interval

### 3.1 WSC：seed-vector percentile bootstrap

WSC 是一个先跨 seed 求均值、再对 stage 取最小值的非线性统计量。
实现按完整逐时点向量重采样 seed，而不是分别重采样每个 stage：

1. 从 \(\mathcal S\) 有放回抽取 \(|\mathcal S|\) 个 seeds；
2. 保留每个被抽 seed 的完整
   \((\widehat C_{s,0},\ldots,\widehat C_{s,T-1})\)；
3. 重新计算
   \(\min_t |\mathcal S|^{-1}\sum_s\widehat C_{s,t}\)；
4. 重复 10,000 次，取 2.5% 和 97.5% 分位数。

随机流由 dataset 和 method 确定，因此重复渲染可复现。当前完整 signed-\(\gamma\)
敏感性图对同一方法的不同 feedback levels 复用同一 seed-resampling stream，以保持
paired 结构。

### 3.2 MeanCov 与 width：跨 seed Student-\(t\) interval

对 selected seeds 的 per-seed 值 \(X_s\)，令

\[
\bar X=\frac1n\sum_sX_s,
\qquad
\mathrm{SE}(\bar X)=\frac{\mathrm{sd}(X_s)}{\sqrt n}.
\]

正文报告

\[
\bar X\ \pm\ t_{0.975,n-1}\,\mathrm{SE}(\bar X).
\]

当前逐时点 coverage/width 图的 error bars 也使用跨 seed 的 pointwise
Student-\(t\) interval；它们不是 simultaneous bands。

### 3.3 Selection Rate：Wilson interval

若 \(k\) 个预设 runs 中有 \(x\) 个成功，\(\widehat p=x/k\)，令
\(z=\Phi^{-1}(0.975)\)，则 Wilson 区间为

\[
\frac{\widehat p+z^2/(2k)}{1+z^2/k}
\ \pm\
\frac{z}{1+z^2/k}
\sqrt{\frac{\widehat p(1-\widehat p)}k+\frac{z^2}{4k^2}}.
\]

## 4. Target eligibility 与加粗规则

正文 eligibility 不使用逐 seed 的达标比例。一个方法在某数据集上进入效率比较，
当且仅当

\[
\widehat{\mathrm{WSC}}_{\mathrm{marg}}\ge0.90
\quad\text{且}\quad
\widehat{\mathrm{SelectionRate}}\ge0.95.
\]

正文表只将满足这两个点估计条件的方法中
`average_normalized_width` 最小者加粗。CI 用于表达不确定性，不被事后改成新的
eligibility gate。

## 5. Committed-prefix marginal SC-PCP 如何选择半径

最终 SC-PCP 直接选择逐时点半径
\(q_0,\ldots,q_{T-1}\)，而不是选择一个全局 scale。

1. `D_COT` 只负责冻结每个 stage 的 101 点候选网格；网格冻结后不再改变。
2. `D_COT ∪ D_cert` 提供选择所用的 score 和 logged trajectories。
3. 在 stage \(t\)，对当前候选 \(q\) 使用已提交前缀加当前动作比率：

   \[
   w_{s,i,t}(q)=
   \left\{\prod_{u=0}^{t-1}
   \frac{\pi_{q_{s,u}}(A_{s,i,u}\mid S_{s,i,u})}
   {\mu(A_{s,i,u}\mid S_{s,i,u})}\right\}
   \frac{\pi_q(A_{s,i,t}\mid S_{s,i,t})}
   {\mu(A_{s,i,t}\mid S_{s,i,t})}.
   \]

4. 用累计 product 不截断、float64 log-stabilized 的 Hájek estimate 计算候选的
   target-policy coverage。这里“不截断”指累计校准权重；每个 action ratio 仍来自
   具有结构性单步 ratio cap 的 target policy：

   \[
   \widehat F_t(q)=
   \frac{\sum_iw_{s,i,t}(q)\mathbf1\{R_{s,i,t}\le q\}}
   {\sum_iw_{s,i,t}(q)}.
   \]

5. 在 \(\widehat F_t(q)\ge0.90\) 的候选中选择估计 normalized width 最小者，
   提交该 \(q_t\)，再进入下一 stage。扫描全部候选是必要的，因为半径会改变策略，
   induced distribution 和 width 不必随 \(q\) 单调。
6. 若某一 stage 没有可行候选，则该 seed `selection_available=False`，并记录
   `failure_stage`；否则状态为 `SELECTED_MARGINAL_POINT`。

该程序的声明范围是 `asymptotic_per_step_marginal`，不是 finite-sample PAC 或
data-conditional certificate。`estimated_min_coverage` 是所选逐时点 Hájek estimates
的最小值；它是选择诊断，不是 fresh deployment coverage。

## 6. 诊断字段与阅读顺序

以下字段只用于解释选择是否健康，不进入正文方法排名：

- `selected_indices`、`selected_endpoint`：选择是否落在候选网格边界；
- `failure_stage`：首个没有可行候选的 stage；
- `mean_ess`、`minimum_ess`、`minimum_candidate_ess`：prefix weights 的有效样本量；
- `estimated_coverage_by_time`：选择时的 weighted coverage；
- `pathwise_coverage`、`mean_log_volume`、`median_volume`、clinical cost：补充诊断。

阅读主结果时按以下顺序：

1. 看 marginal WSC 与 Selection Rate，确认方法是否达到目标且能稳定输出；
2. 在合格方法中比较 Average normalized width；
3. 用 MeanCov 判断是否存在整体过覆盖；
4. 最后查看 ESS、endpoint 和 failure 等机制诊断。
