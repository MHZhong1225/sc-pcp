# Baselines and Experimental Settings

本文档定义论文中的唯一主比较：`Standard CP`、`ACI`、`MFCS`、`SPCI`、
`PRC` 和 `SC-PCP`。结果文件、表格与图例只使用这六个 canonical names。
其中 MFCS、SPCI 与 PRC 是与本任务共同接口对齐的实现；适配范围和不能继承的
上游理论边界在各自小节中说明。

## 1. 共同任务与预测接口

所有方法共享同一套患者划分、冻结的 outcome model、normalized maximum score、
二维矩形 prediction set，以及 behavior-anchored target policy：

\[
R_{it}=\max_{j\in\{1,2\}}
\frac{|Y_{i,t+1,j}-\widehat\mu_j(S_{it},A_{it})|}
{\widehat\sigma_j(S_{it},A_{it})},
\]

\[
C_{q_t}(s,a)=\left\{y:
|y_j-\widehat\mu_j(s,a)|\le q_t\widehat\sigma_j(s,a),\quad j=1,2
\right\},
\]

策略先计算共同的 prediction-box worst-case cost (J_q(s,a))，再令

\[
d_q(s,a)=J_q(s,a)-\min_{a'}J_q(s,a'),
\qquad
u_q(s,a)=\max\{\exp[-(\eta/\tau)d_q(s,a)],10^{-12}\}.
\]

若 (c\ge1) 是 `policy_ratio_cap`，实现选择 (z(s,q)>0)，使

\[
\sum_a\mu_{\rm ref}(a\mid s)
\min\left\{\frac{u_q(s,a)}{z(s,q)},c\right\}=1,
\]

并定义

\[
\pi_{q_t}(a\mid s)=\mu_{\rm ref}(a\mid s)
\min\left\{\frac{u_{q_t}(s,a)}{z(s,q_t)},c\right\}.
\]

主设置为 \(\eta=1\)、\(\tau=1\)、target/reference policy-ratio cap 10、
propensity floor 0.01。policy-ratio cap 约束所有方法共同使用的 target-policy
构造；它不等于 SC-PCP 的 calibration importance-weight clipping。SC-PCP 的
observed-action prefix importance weights **不截断**。

评价目标是每一个阶段的 marginal coverage 均达到
\(1-\alpha=0.90\)，而不是 episode-level joint coverage。所有方法的半径都会
进入同一个 \(C_q\to\pi_q\to P_q\) feedback loop，所以比较差异来自半径如何
校准或适应，而不是 base predictor、score 或 set shape 不同。

SC-PCP、Standard CP、ACI 和 SPCI 可以直接输出自由的 stagewise radius vector。
MFCS 与 PRC 的任务适配器保留 baseline-specific profile-scale family：先在
`D_COT` 的 logged scores 上计算阶段 split-conformal quantiles
\(\widetilde b_t\)，再令

\[
b_t=\frac{\widetilde b_t}
{\left(\prod_u\widetilde b_u\right)^{1/T}},
\qquad q_t(s)=s b_t.
\]

这条一维 family 只属于 MFCS 和 PRC；最终 SC-PCP 不使用 profile 或共享 scale，
而是独立选择 \(q_0,\ldots,q_{T-1}\)。

## 2. 数据角色与信息预算

所有划分都以 patient 为单位，同一患者不会跨角色：

- Synthetic：40% `D_pred`、20% `D_COT`、40% `D_cert`；logging policy 已知。
- Clinical：40% `D_pred`、15% `D_COT`、30% `D_cert`、15% `D_env`。
  Outcome model 与 behavior nuisance model 都只在 `D_pred` 内拟合并冻结，但
  分别读取结局标签和动作标签；不再单独切出 `D_beh`。

最终 calibration sample 对所有方法统一为

\[
D_{\rm cal}=D_{\rm COT}\cup D_{\rm cert}.
\]

`D_COT` 和 `D_cert` 仍然是 patient-disjoint 的物理划分。对 SC-PCP，
`D_COT` 单独负责在选择前冻结每个 stage 的 101-point candidate grid；随后
`D_cal` 中的全部患者共同提供直接的 prefix-importance-weighted calibration
estimate。这里保留 `D_COT` 名称只是为了兼容既有 artifact schema；最终主路径
不训练 COT learner，也不把 `D_cert` 留作单独的统计检验集。

Standard CP 使用 `D_cal` 的全部 logged scores。MFCS、ACI、SPCI 和 PRC 也从
同一 `D_cal` 初始化；其中 MFCS/PRC 的 baseline profile 和 scale grid 仍按预设
只从 `D_COT` 冻结，避免用后续数据调整候选 family。

| Method | Target-policy trajectories used for selection/adaptation | Independent fresh evaluation trajectories |
| --- | ---: | ---: |
| Standard CP | 0 | 50,000 |
| ACI | 2,000 | 50,000 |
| MFCS | 0 | 50,000 |
| SPCI | 2,000 | 50,000 |
| PRC | 2,000 | 50,000 |
| SC-PCP | 0 | 50,000 |

ACI、SPCI 与 PRC **各自**得到 2,000 条 adaptation trajectories，并按
667、667、666 分为三轮；三者不共享这 2,000 条数据。所有方法在同一 seed 下
使用匹配的 50,000-trajectory fresh evaluation random stream，以降低方法间
比较噪声。Clinical adaptation 和 evaluation 都发生在冻结的 held-out empirical
environment 中，不代表真实临床在线干预。

上表的 50,000 条预算只适用于 2026-08-22 production-style paper suite。隔离的
2026-08-25 controlled all-six formal study 为保持 parent controlled protocol 的
Monte Carlo parity，预先冻结为每 method/seed/\(\gamma\) 20,000 条 fresh evaluation；
六方法的 0/2,000 adaptation 预算不变。该例外与完整结果见
[`formal_experiments_20260825.md`](formal_experiments_20260825.md)。

## 3. 五个 baselines

### 3.1 Standard CP

Standard CP 不修正 prediction-induced distribution shift。对 `D_cal` 中
阶段 \(t\) 的 \(n\) 个 scores，使用 finite-sample split-conformal rank

\[
k=\min\left\{n,\left\lceil(n+1)(1-\alpha)\right\rceil\right\},
\qquad q_t^{\rm CP}=R_{t,(k)}.
\]

它部署完整的 \(q_0^{\rm CP},\ldots,q_{T-1}^{\rm CP}\)，不把阶段半径压成
一个 scalar。设置为 \(\alpha=0.10\)，无 target-policy adaptation、无额外
可调超参数，也不受 101-point candidate grid 限制。

### 3.2 MFCS

MFCS adapter 使用从 logged `D_COT` scores 冻结的 baseline profile 和
101-point scale grid。对候选 \(s_k\)、trajectory \(i\) 和阶段 \(t\)，它使用
depth-\(d\) 的有限历史权重

\[
W_{it}^{s_k,d}=\min\left\{B,
\prod_{r=\max(0,t-d+1)}^t
\frac{\pi_{s_kb_r,r}(A_{ir}\mid S_{ir})}
{\mu_r(A_{ir}\mid S_{ir})}\right\}.
\]

其 diagonal estimate 为

\[
\widehat C_{k,t}^{\rm MFCS}=\frac1n\sum_i
W_{it}^{s_k,d}\mathbf 1\{R_{it}\le s_kb_t\}.
\]

它选择第一个 empirical worst-step coverage 达到 0.90 的 scale；不存在时返回
unavailable。主设置为 \(d=3\)、weight cap \(B=40\)。该 estimator 是 raw
Horvitz--Thompson mean，不 self-normalize；它是 empirical baseline，不能被
解释成 SC-PCP 的 asymptotic selection rule。

上游 MFCS 的任务接口不能直接处理本项目的 logged treatment trajectories、
radius-indexed policy 与 multivariate score。当前实现因此是 task-aligned
adapter，不继承未经适配的上游理论保证。

### 3.3 ACI

ACI 为每个阶段维护独立 score history \(H_t^{(r)}\) 和 error level
\(\alpha_t^{(r)}\)。第 \(r\) 轮部署

\[
q_t^{(r)}=Q_{1-\alpha_t^{(r)}}(H_t^{(r)}),
\qquad
\alpha_t^{(r+1)}=\operatorname{clip}_{[0.001,0.999]}
\left[\alpha_t^{(r)}+\gamma(\alpha-e_t^{(r)})\right],
\]

其中 \(e_t^{(r)}\) 是该轮 target-policy batch 的阶段误覆盖率。新 scores 加入
对应 stage history，每个 stage 最多保留 10,000 个。最终输出自由的 \(T\) 维
radius vector。主设置为 \(\gamma=0.01\)、3 rounds、2,000 条 adaptation
trajectories。

这里实现的是按 patient batch 和 treatment stage 更新的 ACI controller，而非
原生逐样本时间序列 reproduction，因此不直接继承原 ACI 的长期频率保证。

### 3.4 SPCI

SPCI adapter 使用共同的 multivariate normalized-maximum score，并为每个阶段
维护独立的 recent-score buffer：

\[
q_t^{(r)}=Q_{0.90}(H_t^{(r)}),
\qquad
H_t^{(r+1)}=\operatorname{last}_{1000}
\left(H_t^{(r)}\cup\{R_{it}^{(r)}\}_i\right).
\]

它输出自由的 \(T\) 维 radius vector。主设置为 buffer 1,000、3 rounds、
2,000 条 adaptation trajectories。

上游 MultiDimSPCI 采用 chronological multivariate series、sequential
residual model 和 ellipsoidal regions；这些接口与本任务不同。当前 adapter
保持所有方法共同的 rectangular set，因而不能把上游 ellipsoidal-set guarantee
赋予本实验结果。

### 3.5 PRC

PRC adapter 与 MFCS 使用相同的 logged baseline profile-scale family 和
frozen scale grid。它从能够包住 Standard CP stagewise radii 的 scale 初始化：

\[
s^{(0)}=\max_t\frac{q_t^{\rm CP}}{b_t}.
\]

第 \(r\) 轮部署 \(q_t^{(r)}=s^{(r)}b_t\)，取得新的 target-policy scores，
并在 grid 上对每个 \(s_k\) 计算

\[
\widehat C_{k,t}^{(r)}=\frac1{n_r}\sum_i
\mathbf1\{R_{it}^{(r)}\le s_kb_t\}.
\]

候选必须在所有阶段通过 finite-grid Hoeffding guard，并满足
\(|s_k-s^{(r)}|\le h_{\max}\)；随后选择最小可行 scale。若不存在可行候选，
则保持当前 scale：

\[
m_r=\sqrt{\frac{\log(KT/\delta)}{2n_r}},
\qquad
\min_t\widehat C_{k,t}^{(r)}-m_r\ge1-\alpha.
\]

主设置为 \(K=101\)、\(\delta=0.05\)、\(h_{\max}=0.35\)、3 rounds、
2,000 条 adaptation trajectories。

原生 PRC 假设 scalar monotone risk、已知 sensitivity 条件，并使用当前 shifted
distribution 的新样本。本项目不假设 performative coverage 关于 radius 单调，
因此采用 frozen-grid sequential adapter；当前行不继承原生 PRC theorem。

## 4. SC-PCP（ours）

最终 SC-PCP 直接校准 target-policy stagewise marginal quantile。它没有共享
scale、stage profile、learned occupancy-ratio model 或额外的 confidence-bound
screening layer。

首先，`D_COT` 在每个阶段分别按 0.50--0.999 empirical score quantiles 冻结
\(K=101\) 个候选：

\[
\mathcal G_t=\{r_{t1},\ldots,r_{tK}\},\qquad t=0,\ldots,T-1.
\]

然后令 \(D_{\rm cal}=D_{\rm COT}\cup D_{\rm cert}\)。在 stage \(t\)，已经选定
的 \(q_{<t}=(q_0,\ldots,q_{t-1})\) 被视为 committed prefix。对每个
\(r\in\mathcal G_t\)，observed-action prefix likelihood ratio 为

\[
W_{it}(r;q_{<t})=
\prod_{h<t}
\frac{\pi_{q_h,h}(A_{ih}\mid S_{ih})}
{\mu_h(A_{ih}\mid S_{ih})}
\frac{\pi_{r,t}(A_{it}\mid S_{it})}
{\mu_t(A_{it}\mid S_{it})}.
\]

Synthetic 使用已知 logging policy；clinical denominator 使用在独立
`D_pred` 上拟合并冻结的 behavior propensity model。累计前缀乘积不做 terminal
clipping 或 capping；但每一个因子来自上面定义的、具有结构性单步 ratio cap 的
target policy。实现以 float64 保存 raw log weights，并对每个 candidate 减去本列
最大 log weight 后再指数化。由于 Hájek ratio 的分子和分母同时乘以同一常数，
这个数值稳定化不改变 estimate 或 effective sample size。

候选的 target-policy coverage estimate 为

\[
\widehat C_t(r;q_{<t})=
\frac{\sum_i W_{it}(r;q_{<t})\mathbf1\{R_{it}\le r\}}
{\sum_i W_{it}(r;q_{<t})}.
\]

令

\[
B_{it}=\frac{1}{2}\sum_{j=1}^2
\frac{2\widehat\sigma_j(S_{it},A_{it})}{s_{Y,j}},
\]

其中 \(s_{Y,j}\) 是 `D_pred` outcome standard deviation；代码中的候选
normalized width estimate 是

\[
\widehat W_t(r;q_{<t})=
\frac{\sum_i W_{it}(r;q_{<t})\,rB_{it}}
{\sum_i W_{it}(r;q_{<t})}.
\]

最终逐阶段选择规则为

\[
\widehat q_t\in
\arg\min_{r\in\mathcal G_t:\,\widehat C_t(r;\widehat q_{<t})\ge0.90}
\widehat W_t(r;\widehat q_{<t}).
\]

选择后把 \(\widehat q_t\) 加入 prefix，下一阶段的所有 candidate weight 都包含
这一已提交决策。即使 coverage 或 width 随 radius 不单调，代码仍扫描全部
candidate，而不是假设“第一个达到阈值”的点必然最有效。如果某一阶段没有
可行 candidate，SC-PCP 返回 unavailable 并记录 failure stage；不以最大集合
兜底。最终输出是自由的 \(T\) 维向量
\((\widehat q_0,\ldots,\widehat q_{T-1})\)。

### Guarantee boundary

SC-PCP 的理论边界是 asymptotic per-step marginal coverage。对固定 \(T\)，若
满足 sequential ignorability/identification、positivity/overlap、prefix
likelihood ratios 已知或 uniformly ratio-consistent，完整紧 prefix-radius 类上的
uniform convergence 成立，且 selection availability probability 趋于一，则

\[
\min_{0\le t<T}C_t(\widehat q_{0:t})
\ge1-\alpha-o_p(1).
\]

这条结论来自所选 schedule 的经验可行性与完整 coverage surface 的统一误差界，
不要求唯一或稳定的 population selector。这不是 finite-sample distribution-free、
PAC 或 data-conditional guarantee。当前实现用 calibration data 同时估计并选择改变
deployment law 的半径，因此不能声称普通 split CP 式的 exact finite-sample validity。
Clinical 结果进一步依赖冻结的 empirical environment 和 fitted behavior nuisance，
只能解释为 controlled model-based evaluation，不能外推为真实患者干预保证。

## 5. 主设置、seeds 与报告规则

| Setting | Main value |
| --- | ---: |
| Target miscoverage \(\alpha\) | 0.10 |
| SC-PCP stage candidates \(K\) | 101 per stage |
| Candidate quantile range | [0.50, 0.999] |
| SC-PCP calibration sample | `D_COT` union `D_cert` |
| SC-PCP cumulative prefix weights | unclipped; float64 log stabilization |
| Target/reference policy-ratio cap | 10 |
| MFCS history depth | 3 |
| MFCS importance-weight cap \(B\) | 40 |
| ACI learning rate \(\gamma\) | 0.01 |
| ACI retained history per stage | 10,000 |
| SPCI buffer | 1,000 |
| Online adaptation rounds | 3 |
| Adaptation trajectories per online method/seed | 2,000 |
| PRC candidate scales \(K\) / \(\delta\) | 101 / 0.05 |
| PRC maximum one-round scale move | 0.35 |
| Fresh evaluation trajectories per method/seed | 50,000 |

RQ1 的完整设置为：Synthetic \(T=12\)，100 seeds；eICU、INSPIRE 和
MIMIC-IV \(T=12\)，各 20 seeds；MIMIC-CXR \(T=6\)，20 seeds。RQ3 的
Synthetic feedback stress 对 \(\beta=0,0.5,2\) 各运行 100 seeds，
\(\beta=1\) 直接复用 RQ1 Synthetic。RQ2 与 RQ4 只复用 suite manifest 指向的
冻结 artifacts，不增加另一组随机重复。GPU 数量只影响运行时间，不改变任何
统计预算。

主表必须明确 offline 与 online 的信息条件：Standard CP、MFCS、SC-PCP 是
zero-target-feedback methods；ACI、SPCI、PRC 各自获得额外 2,000 条
target-policy adaptation trajectories。六种方法都必须出现在每个完整 seed 中，
且只使用 canonical names。Vendored repositories 用于核对来源和接口，不应把
task-aligned adapter 写成未经修改的 native reproduction。

## 6. 正式运行、续跑与渲染

新建完整 suite（output root 必须为空）：

```bash
conda run -n ucp python scripts/run_paper_suite.py \
  --sections rq1,rq3 \
  --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire \
  --devices cuda:0,cuda:1 \
  --output-root results/work/paper_final
```

中断后只允许按完全相同的 manifest 续跑：

```bash
conda run -n ucp python scripts/run_paper_suite.py \
  --sections rq1,rq3 \
  --datasets synthetic,mimic_iv,mimic_cxr,eicu,inspire \
  --devices cuda:0,cuda:1 \
  --output-root results/work/paper_final \
  --resume
```

完整性检查通过后生成最终 PDF：

```bash
conda run -n ucp python tools/render_paper_results.py \
  --input results/work/paper_final \
  --output results/paper_final
```

Renderer 对缺失 dataset、seed、六方法记录或 `COMPLETE` marker 的 suite
fail closed，并要求最终 output directory 只包含 PDF。

## References

1. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116). *Journal of the American Statistical Association*, 113(523), 1094--1111.
2. Gibbs, I. and Candès, E. (2021). [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html). *Advances in Neural Information Processing Systems*, 34.
3. Prinster, D., Stanton, S. D., Liu, A., and Saria, S. (2024). [Conformal Validity Guarantees Exist for Any Data Distribution (and How to Find Them)](https://proceedings.mlr.press/v235/prinster24a.html). *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:41086--41118.
4. Xu, C., Jiang, H., and Xie, Y. (2024). [Conformal Prediction for Multi-Dimensional Time Series by Ellipsoidal Sets](https://proceedings.mlr.press/v235/xu24m.html). *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:55076--55099.
5. Li, V., Chen, B., Mao, Y., Lei, Q., and Deng, Z. (2025). [Performative Risk Control: Calibrating Models for Reliable Deployment under Performativity](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6c71e8beb41e142e463b16818537ed0-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, 38.
