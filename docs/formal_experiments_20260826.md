# 理论与鲁棒性正式实验（2026-08-26）

本文档是 2026-08-26 post-freeze 证据的权威结果记录：horizon×overlap、
calibration-size convergence、propensity robustness、strict-split robustness，
以及随后冻结完成的 dataset-native controlled clinical extension v2。前四项是
theory/robustness diagnostics；最后一项是带前置 fidelity/overlap gates 的受控
clinical stress study。全部研究均已完成，工件完整，并经过独立的逐行数值与
provenance 审计。

这些研究不引入新 estimator，不改变 canonical selector，也不覆盖 2026-08-22
paper suite、2026-08-25 formal studies 或 Synthetic native \(\beta=2\) setting。
SC-PCP 的唯一正式定义仍见
[`final_method.md`](final_method.md)，主 coverage 指标始终是

\[
\operatorname{WSC}
=\min_t \frac1S\sum_{s=1}^S C_{s,t}.
\]

它不是 `mean_seed(min_t C_seed,t)`，也不是 MeanCov。

## 1. 结论先行

1. **Horizon 与 overlap 的统计代价得到直接量化。** 在 200 个 paired exact-MDP
   instances 中，SC-PCP 在全部 25 个预设 \(T\times\mathrm{TV}\) cells 都可用且 WSC
   高于 0.90；但在最强 cell \(T=20,\mathrm{TV}=0.15\) 中，median minimum
   selected ESS/n 从 1 降到 0.1127，committed-surface sup error 从 0.0206
   升到 0.0391。正确 identification 并不消除 horizon--overlap curse。
2. **Coverage-surface estimation 呈清楚的样本收敛。** 在 100 个 MDP problems、
   每个 20 个 nested logged resamples 中，固定 population grid 的 mean sup error
   从 \(n=250\) 的 0.05599 降到 \(n=10{,}000\) 的 0.009015，ESS 基本稳定。
   Canonical selector 的 availability 始终为 100%，逐实例 target attainment 从
   62.75% 上升到 100%。但 WSC 收敛到约 0.92，而不是恰好 0.90；不能写成
   monotone nominal convergence 或 finite-sample guarantee。
3. **正确设定的 fitted propensity 几乎复现 oracle。** 固定 oracle target law 后，
   correctly specified multinomial arm 与 oracle 的 WSC 完全相同，ESS 仅下降
   0.27 pp。严重 reduced-state misspecification 使 minimum stage-mean ESS/n
   下降 22.86 pp，并在后期阶段造成最高 0.478 pp 的 coverage loss，但本设计的
   WSC 仍高于 0.90。因此该实验支持 nuisance sensitivity，不支持“任意
   misspecification 下鲁棒”或“misspecification 必然导致 headline failure”。
4. **Strict split 与 canonical 方法表现接近。** 在同一个 \(D_{\rm COT}\)-frozen
   grid 上，仅用独立 \(D_{\rm cert}\) selection 并未对 Synthetic 或 MIMIC-IV
   产生可分辨的 WSC/width 改变。在 controlled \(\gamma=-2\) 中，strict arm
   的 width 几何平均增加 0.85%，但 WSC 差异不确定。该结果是 robustness audit，
   不构成事后替换 canonical SC-PCP 的规则。
5. **Dataset-native clinical extension 给出一个通过 gate 的 MIMIC-IV stress
   replication，以及三个正式 K0 NO-GO。** 四个 clinical datasets 的 support gate
   都通过 20/20 seeds；MIMIC-IV 的 logging-mixture K0 fidelity 为 20/20，并通过
   \(\gamma=-4\) donor-overlap screen，因此保存完整 signed curves。在 confirmatory
   \(\gamma=-4\) cell，Standard CP WSC 为 0.86358，SC-PCP 为 0.90089，paired
   improvement 为 3.73 pp；SC-PCP 点估计达标但其 95% CI 下端为 0.89431。
   eICU、INSPIRE、MIMIC-CXR + IV/ED 的 K0 fidelity 分别只有 12/20、13/20、
   10/20，低于预设 19/20，因此没有生成任何 scientific coverage row。这里的
   NO-GO 是结果，不允许用 MIMIC-IV、旧 v1 或 production curves 填补。

## 2. Provenance 与不可混用边界

四项工件均由同一个 post-snapshot 源树产生：

```text
source_tree_sha256 = 296569d628875de774cb5012004c345d624653c1f4ecd2d3b6ff02e292f99226
experiment_tree_sha256 = 1a470b8998656b2ae9abc58d3e08334bd582f405065fc9852d0bb30ae0011e07
```

可恢复源码快照为
[`results/work/extension_source_snapshot_296569d6_20260826.tar.gz`](../results/work/extension_source_snapshot_296569d6_20260826.tar.gz)：

```text
archive_sha256  = 57f2195a11d802fef9af84dd7f61e8df8a1d81853c899b3c03f8cac293dfa314
manifest_sha256 = 221d1adca4e997e878aac00b4fb75f1125469ea66269716e7072d458f2bf6e7f
```

该快照是本地 content-addressed recovery snapshot，不是外部时间戳、签名 tag 或
预注册凭证。每项新研究同时验证并绑定 2026-08-25 的 parent formal snapshot；这表示
代码 lineage，而不表示新增文件存在于 parent archive 中。

| Study | Formal root | `summary.json` SHA-256 | `COMPLETE` SHA-256 |
|---|---|---|---|
| Horizon×overlap | [`horizon_overlap_v1`](../results/work/horizon_overlap_v1) | `1166d04e3e156e9e4e282cfaf2c2fd1571d707272d3bdfe19483d18528c7dd8a` | `870a027bc93d6c126675dbb4ad23c9665e04d02d02b3556b5daad0917542ef3b` |
| Calibration-size convergence | [`rq6_ncal_convergence_v1`](../results/work/rq6_ncal_convergence_v1) | `1178ff46caa7d7fd91837a52bbb65e5f7ffe95f5c2991ea0a59e355be563d5c2` | `dd4219aef40e2c69f6111decb14ba88877db61fae6b93edf0d239879dac10c36` |
| Propensity robustness | [`propensity_robustness_v1`](../results/work/propensity_robustness_v1) | `7176adab5da84f5326d60f926c66199ae01669f14d7836f19fe9254e5c07fd00` | `b0b555d9e9395ce9c23c7533fd4741eda894c746dc7a8aab68066f5966e9c359` |
| Strict split | [`strict_split_robustness_v1_20260826`](../results/work/strict_split_robustness_v1_20260826) | `c2a555fdc3fb9ddaab3acd9f8ed49ea5ed1a2be3a81e7a46dab80d9ffcb2e1a2` | `a9091e3faf5748d7b81d3d44d8dd90386795f069e8c02051289e3050f70bc636` |

正式工件固定的是运行时源 hash。后续对 isolated runners 的 resume-only maintenance
不会改动这些工件或科学结果；直接对旧目录 resume 时仍应遵循 stored provenance，
而不是把维护后的活动树说成 bitwise-identical source。

## 3. Study D：Horizon × overlap phase diagram

### 3.1 Protocol

- 200 个 paired `M3_full_feedback` exact finite-MDP instances；
- 每个 instance 先生成 \(T=20\) 过程，再截断为
  \(T\in\{2,4,8,12,20\}\)；
- outcome-blind policy mixing 被求解到 nominal one-step TV
  \(\{0,.025,.05,.10,.15\}\)，并另存 realized TV；
- 每个 cell 使用 3,000 条 calibration trajectories；
- 比较 `Standard CP`、history-only、current-only 和 full `SC-PCP`；后三者是
  theorem-facing diagnostics，不是新的 canonical baselines；
- 所有 conditions 共用 paired kernel 与 behavior log；population coverage/width
  由 exact recursion 计算。

为使预设 TV levels 可达，RQ5 专用 policy response center 在 outcome-blind design
阶段从 parent M3 的 2.5 固定为 radius minimum 1.4。其余 M3 transition/outcome
family 不变。该 protocol variant 被显式记录，不能与 2026-08-25 exact-MDP study
拼接。

### 3.2 SC-PCP phase diagram

下面三张表的行是 horizon，列是 nominal TV。所有 25 个 cells 的 availability 都是
1.000。WSC 在 horizon 方向重复并非实现错误：每个 TV cell 的 worst stage 已在前两步
出现，因此延长 horizon 没有改变 `min_t mean_instance(C_it)`。

**WSC**

| \(T\backslash\mathrm{TV}\) | 0 | .025 | .05 | .10 | .15 |
|---:|---:|---:|---:|---:|---:|
| 2 | .91727 | .91759 | .91563 | .91591 | .91459 |
| 4 | .91727 | .91759 | .91563 | .91591 | .91459 |
| 8 | .91727 | .91759 | .91563 | .91591 | .91459 |
| 12 | .91727 | .91759 | .91563 | .91591 | .91459 |
| 20 | .91727 | .91759 | .91563 | .91591 | .91459 |

**Median minimum selected ESS/n**

| \(T\backslash\mathrm{TV}\) | 0 | .025 | .05 | .10 | .15 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.0000 | .9926 | .9711 | .8930 | .7825 |
| 4 | 1.0000 | .9851 | .9424 | .7969 | .6145 |
| 8 | 1.0000 | .9704 | .8878 | .6358 | .3850 |
| 12 | 1.0000 | .9557 | .8355 | .5078 | .2418 |
| 20 | 1.0000 | .9268 | .7412 | .3329 | .1127 |

**Median committed-surface sup error**

| \(T\backslash\mathrm{TV}\) | 0 | .025 | .05 | .10 | .15 |
|---:|---:|---:|---:|---:|---:|
| 2 | .01163 | .01177 | .01174 | .01202 | .01222 |
| 4 | .01491 | .01454 | .01432 | .01481 | .01543 |
| 8 | .01728 | .01708 | .01700 | .01778 | .01959 |
| 12 | .01851 | .01838 | .01886 | .01996 | .02488 |
| 20 | .02063 | .02088 | .02164 | .02582 | .03907 |

### 3.3 T=20 的结构比较

| TV | Method | WSC | Width | Median min ESS/n | Median surface error |
|---:|---|---:|---:|---:|---:|
| 0 | Standard / history / current / SC-PCP | .91727 | 5.5834 | 1.0000 | .02063 |
| .15 | Standard CP | .92453 | 5.5834 | 1.0000 | .12005 |
| .15 | History-only Prefix-IW | .92105 | 4.9788 | .1191 | .04109 |
| .15 | Current-only IW | .91712 | 5.5431 | .8771 | .12018 |
| .15 | **SC-PCP** | **.91459** | **4.9200** | **.1127** | **.03907** |

这项 study 没有显示 Standard CP undercoverage，也不是六 baseline 排名。它说明完整
prefix 在强 overlap stress 下得到较小的 committed-surface error 和较窄的 selected
width，但以显著更低 ESS 为代价；SC-PCP 的 WSC 更接近 nominal 不是“更高 coverage”。

## 4. Study E：Calibration-size convergence

### 4.1 Protocol 与 estimands

100 个固定 M3 exact-MDP problems，每个 problem 有 20 个 independent logged
resamples；\(T=4,K=7\)，policy TV 在预设 midpoint radius 处 outcome-blind 地固定为
0.05。六个总 calibration budgets 是 250、500、1,000、2,000、5,000、10,000，
并保持 \(D_{\rm COT}:D_{\rm cert}=1:2\)。同一 problem/replicate 的较小样本是最大
role-specific pools 的 nested prefixes。

- **Track A** 在固定 population grid 上穷举 2,401 个 complete schedules，并计算
  full-prefix Hájek coverage surface 相对 exact population surface 的 sup error。
- **Track B** 每个 \(n\) 由 \(D_{\rm COT}\) 生成 empirical grid，再调用未修改的
  canonical selector，并用 exact recursion 评价 selected schedule。

95% 区间以 100 个 MDP problems 为 cluster、保留每个 problem 内 20 个 logged
resamples做 10,000 次 bootstrap。它们不是 2,000 个独立 clusters。

### 4.2 完整结果

| \(n_{\rm cal}\) | Track-A mean sup error [95% CI] | Min ESS/n | Track-B WSC [95% CI] | Width [95% CI] | Availability | Rowwise target attainment | Endpoint |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | .05599 [.05532,.05666] | .73424 | .93358 [.93246,.93465] | 5.5584 [5.5339,5.5818] | 1.000 | .6275 | .5175 |
| 500 | .03992 [.03943,.04040] | .73362 | .93210 [.93110,.93275] | 5.4487 [5.4282,5.4691] | 1.000 | .7620 | .3085 |
| 1,000 | .02846 [.02812,.02882] | .73290 | .92450 [.92342,.92560] | 5.3300 [5.3101,5.3500] | 1.000 | .8700 | .1140 |
| 2,000 | .02002 [.01982,.02023] | .73277 | .92093 [.92024,.92166] | 5.2722 [5.2539,5.2906] | 1.000 | .9660 | .0265 |
| 5,000 | .01281 [.01266,.01295] | .73245 | .91947 [.91917,.91977] | 5.2576 [5.2399,5.2748] | 1.000 | .9970 | .0005 |
| 10,000 | .00901 [.00891,.00912] | .73238 | .91979 [.91957,.92000] | 5.2604 [5.2430,5.2774] | 1.000 | 1.0000 | 0 |

Track-A 的六点 descriptive log--log slope 是 \(-0.4951\)。它与 root-\(n\)
decay 相容，但只是未预先声称的描述统计，不是已证明的 exponent。Track-B 的 WSC
和 width 在 5,000 到 10,000 之间轻微反向且区间重叠；不能声称单调。

这里必须区分两个指标：canonical WSC 是
`min_t mean_problem,replicate(C_it)`；rowwise target attainment 是每个
problem/replicate 的 `min_t C_it >= .9` 比例。在 \(n=250\) 时，前者是 .93358，
而 `mean(rowwise min)` 是 .90569。二者不可互换。

## 5. Study F：Propensity robustness

### 5.1 Primary 与 appendix 分层

100 个 paired exact-MDP instances；(S=8,A=3,T=8,K=7)；每个 instance 有独立
5,000 条 nuisance trajectories 和 5,000 条 calibration trajectories。

Primary 固定 oracle-\(\mu\) anchored target law，只替换 committed-prefix transport
denominator：oracle、correct full-state multinomial logistic、以及 deliberately
misspecified reduced-state logistic。Appendix 让每个 fitted propensity 同时成为
policy anchor 和 denominator，因此 target law 发生改变；这两个 estimands 绝不能合并。

### 5.2 Nuisance 质量

| Arm | MAE | Excess log loss | Mean absolute relative error | Max relative error | Min propensity |
|---|---:|---:|---:|---:|---:|
| Oracle | 0 | 0 | 0 | 0 | .08751 |
| Correct multinomial | .00492 | .000205 | .01857 | .07361 | .08557 |
| Misspecified reduced-state | .04325 | .01573 | .18889 | .73653 | .14901 |

Misspecified arm 较高的 minimum propensity 来自 state reduction 的平滑作用，不代表
模型更准确或 positivity 更好。

### 5.3 Primary fixed-target-law results

| Arm | WSC [95% CI] | Width [95% CI] | Minimum stage-mean ESS/n [95% CI] | Selected-policy TV |
|---|---:|---:|---:|---:|
| Oracle | .91482 [.91207,.91769] | 5.4959 [5.4810,5.5099] | .96934 [.96864,.97009] | .02343 |
| Correct multinomial | .91482 [.91207,.91769] | 5.4933 [5.4784,5.5073] | .96663 [.96556,.96773] | .02335 |
| Misspecified reduced-state | .91579 [.91293,.91876] | 5.4661 [5.4469,5.4854] | .74078 [.73863,.74284] | .02248 |

Correct arm 相对 oracle 的 WSC difference 为 0，width difference 为 -0.00263，
ESS difference 为 -0.00271。Misspecified arm 的 WSC 反而高 0.000965，不能叫
headline degradation；但 ESS 下降 0.22856，且完整 schedule 与 oracle 的一致率从
correct arm 的 96/100 降到 69/100。所有 arms availability 均为 100/100。

WSC 在所有 arms 都由 stage 0 决定，因此会掩盖后期差异。Misspecified arm 在 stage 6
相对 oracle 的 coverage loss 最大，为 0.00478，paired 95% CI
[.00222,.00778]；不过该 stage coverage 仍高于 .90。

### 5.4 Appendix end-to-end target-law drift

| Arm | WSC | Width | Min stage-mean ESS/n | Target-law candidate-surface TV drift from oracle |
|---|---:|---:|---:|---:|
| Oracle | .91482 | 5.4959 | .96934 | 0 |
| Correct multinomial | .91483 | 5.4950 | .96939 | .00740 |
| Misspecified reduced-state | .91467 | 5.4959 | .97466 | .06406 |

Misspecified deployment 相对 primary oracle deployment 的 selected-policy TV 是 .06461。
Appendix 中较高 ESS 与相近 width 部分来自 anchor/denominator 抵消，不能被表述为固定
target policy robustness。

## 6. Study G：Strict-split robustness

### 6.1 Protocol

两 arms 使用同一个只由 \(D_{\rm COT}\) 构造的 stagewise grid。Canonical arm 在
\(D_{\rm COT}\cup D_{\rm cert}\) 上 selection；strict arm 仅在独立
\(D_{\rm cert}\) 上 selection。两 arms 使用匹配 evaluation randomness。

Synthetic main 的 100 seeds 与 MIMIC-IV 的 20 seeds 是冻结 paper-suite seed reuse；
controlled \(\gamma=-2\) 使用 20 个 fresh 99k seeds。该 study 没有 post-hoc upgrade
gate，不能根据结果切换正文方法。

2026-08-25 all-six 的 \(\gamma=-4\) hero-display convention 不回写本独立
robustness protocol；strict-split controlled arm 仍只在 \(\gamma=-2\) 运行，不能
称为 \(\gamma=-4\) strict-split evidence。

### 6.2 完整结果

| Setting | Variant | Availability | WSC [95% CI] | Width [95% CI] | Selected min ESS/n | Candidate min ESS/n |
|---|---|---:|---:|---:|---:|---:|
| Synthetic | Canonical | 100/100 | .90177 [.90055,.90204] | 1.84359 [1.84003,1.84716] | .95434 | .93677 |
| Synthetic | Strict | 100/100 | .90170 [.90018,.90197] | 1.84399 [1.84018,1.84782] | .95421 | .93668 |
| MIMIC-IV | Canonical | 20/20 | .90118 [.89845,.90411] | 2.18438 [2.15653,2.21288] | .98933 | .97882 |
| MIMIC-IV | Strict | 20/20 | .90155 [.89875,.90461] | 2.18684 [2.15783,2.21653] | .98928 | .97879 |
| Controlled \(\gamma=-2\) | Canonical | 20/20 | .89989 [.89486,.90075] | 3.76785 [3.57496,3.96361] | .31444 | .30876 |
| Controlled \(\gamma=-2\) | Strict | 20/20 | .90079 [.89432,.90186] | 3.79819 [3.61123,3.99074] | .30851 | .30179 |

Paired strict-minus-canonical comparisons：

| Setting | ΔWSC [95% CI] | Geometric width ratio [95% CI] | Δ selected min ESS/n [95% CI] |
|---|---:|---:|---:|
| Synthetic | -.000075 [-.000982,.000543] | 1.00021 [.99957,1.00085] | -.00013 [-.00032,.00006] |
| MIMIC-IV | +.000370 [-.000098,.001024] | 1.00109 [.99787,1.00399] | -.00005 [-.00013,.00002] |
| Controlled \(\gamma=-2\) | +.000898 [-.002690,.003788] | 1.00855 [1.00142,1.01596] | -.00593 [-.01523,.00316] |

三项 WSC interval 均跨零；Synthetic/MIMIC width-ratio interval 也跨 1。Controlled
strict width inflation 为 0.85%，其 interval 不跨零。ESS differences 的 intervals
均跨零。Absolute ESS 不能直接解释为 strict 与 canonical 使用同样 selection sample
size，因为 strict arm 本来就少用 \(D_{\rm COT}\) 的 calibration rows。

## 7. 论文可以与不可以写什么

### 可以写

- exact identification 之外，full-prefix transport 的 finite-sample difficulty 随
  horizon 和 policy divergence 增长；ESS 与 committed-surface error 直接展示这一点；
- 在 fixed-grid exact-MDP 中，coverage-surface estimation error 随 calibration size
  清楚下降，canonical selection availability 与逐实例 target attainment 同时改善；
- correctly specified propensity plug-in 在本 benchmark 中近似 oracle，而 severe
  misspecification 显著降低 overlap 并造成后期 stage discrepancy；
- strict independent-calibration variant 与 canonical method 在两个主 settings 中表现
  接近，在 controlled stress 中付出约 0.85% width；
- 这些结果与 2026-08-25 exact identification、controlled all-six 和 copula NO-GO
  共同界定 SC-PCP 何时有用、何时困难，以及什么样的 feedback 才与 calibration
  相关。

### 不可以写

- finite-sample、distribution-free、PAC、data-conditional 或 episode-wise guarantee；
- RQ6 证明了理论 \(n^{-1/2}\) rate，或 WSC 单调/精确收敛到 0.90；
- RQ5 证明 SC-PCP 在所有 horizon/overlap 上优于 Standard，或给出六 baseline 排名；
- propensity misspecification 在本设计造成 nominal WSC failure，或 SC-PCP 对任意
  misspecification 都鲁棒；
- strict split 显著改善 coverage、证明 equivalence，或应事后替换 canonical method；
- 这些 exact/semi-synthetic diagnostics 是自然 ICU causal evidence 或 universal SOTA。

## 8. 完整证据链现在是什么

当前实验链条已经闭合：

1. 2026-08-25 exact MDP 说明 full prefix 在同时存在 history 与 current-action channels
   时是 population-identification correct 的；
2. 2026-08-25 controlled all-six 说明 adverse score-law shift 下 SC-PCP 大幅恢复
   Standard 的 coverage loss，但存在 width 与轻微 residual undercoverage；
3. 2026-08-25 copula NO-GO 说明 prediction-mediated feedback 本身并不自动导致
   calibration-relevant shift；
4. 本轮 horizon×overlap 量化正确 estimator 的 overlap/horizon 代价；
5. 本轮 \(n_{\rm cal}\) study 连接 uniform-surface convergence 与 selected-schedule
   behavior；
6. propensity 与 strict-split audits 则回答 nuisance 与 sample-reuse 两个 reviewer
   最直接的问题。
7. dataset-native controlled clinical v2 进一步回答该 stress construction 能否跨
   clinical geometries 使用：MIMIC-IV 通过并产生曲线，另外三个 setting 在 coverage
   打开前被 K0 fidelity gate 正式拒绝。

因此最准确的论文主张不是“SC-PCP 在所有地方都是 SOTA”，而是：SC-PCP 为
prediction-radius-dependent longitudinal policies 提供完整 committed-action-prefix 的
可识别 calibration surface，并在 overlap 足够、propensity 质量受控时，以渐近
per-step marginal validity 为目标；正式实验同时展示其 correction 能力、统计代价与
适用边界。Clinical v2 的三个 K0 NO-GO 进一步说明，“dataset-native”不等于机制可在
所有数据集自动成立。

## 9. 投稿图与 source data

两张由上述四个 immutable artifacts 确定性导出的论文图位于
[`results/paper_theorem_robustness_20260826`](../results/paper_theorem_robustness_20260826)：

- [`figure_theory_diagnostics.pdf`](../results/paper_theorem_robustness_20260826/figure_theory_diagnostics.pdf)：
  horizon×overlap phase diagrams 与 \(n_{\rm cal}\) convergence；
- [`figure_robustness_audits.pdf`](../results/paper_theorem_robustness_20260826/figure_robustness_audits.pdf)：
  propensity 与 strict-split robustness。

Paper 目录只含两个 PDF。完整 editable SVG、600-dpi TIFF、PNG preview、source-data
CSV、analysis JSON、render manifest 与视觉 QA 位于
[`results/work/theorem_robustness_report_20260826`](../results/work/theorem_robustness_report_20260826)。
Renderer 只读 frozen summaries 和 intervals，不重跑实验或 bootstrap。

连同 method schematic、five-setting Pareto、stagewise profiles、exact-MDP heatmap
和 controlled all-six 图表的完整正文顺序、caption 与 claim boundary，统一记录在
[`figure_portfolio_20260826.md`](figure_portfolio_20260826.md)。

## 10. Dataset-native controlled clinical extension v2

### 10.1 这项 extension 回答什么

正式工件为
[`results/work/controlled_clinical_extension_v2`](../results/work/controlled_clinical_extension_v2)。
它检验同一个 calibration-aligned signed stress construction 能否在四个 clinical
dataset 各自的 patient population、action ontology、horizon 和 donor geometry 中
通过预先冻结的 fidelity/overlap gates。它没有把 MIMIC-IV transition relabel 成其他
数据集，也不允许复用旧 MIMIC v1 结果。

必须把三类曲线分开：

1. 2026-08-22 **production/native five-setting suite** 有 Synthetic、MIMIC-IV、
   MIMIC-CXR + IV/ED、eICU 和 INSPIRE 五套真实主设置；
2. Synthetic strong-feedback **native \(\beta=2\)** 仍是原 synthetic DGP，
   \(\beta\) 不是本 study 的 signed \(\gamma\)，不能称为 clinical controlled curve；
3. 本节 **clinical controlled v2** 只包含四个 clinical datasets，并先经过硬 gate。
   只有通过 gate 的 setting 才能出现 scientific coverage/width curves。

因此，这项 extension 不是“又一张五数据集主表”，也不替代 production-style
coverage--width 结果。它提供的是受控机制在 dataset-native clinical geometries 上
是否可解释、是否可运行的 gated evidence。

### 10.2 冻结 protocol 与 gate 顺序

每个数据集使用 20 个预设 seeds。所有 patient 在拟合前按 unique patient ID 划分为
互不相交的 \(D_{\rm pred}/D_{\rm fidelity}/D_{\rm env}=40/20/40\)：

- \(D_{\rm pred}\) 只拟合 outcome model 与 behavior-policy nuisance；
- \(D_{\rm fidelity}\) 只定义 source-score q80/q95 probes，并执行 K0
  logging-mixture one-step replay；
- \(D_{\rm env}\) 只构造 dataset-native donor transition environment。

Signed environment 使用
\(\gamma\in\{-4,-2,0,2,4\}\)。在每个固定 \(\gamma\) 内，source 与 target
共用同一个 \(K_\gamma\)；两者只在 logging policy 与 prediction-radius-dependent
target policy 上不同。\(K_\gamma\) 的 donor reweighting 使用 stagewise frozen
conformity-score rank 与数据集自身的 action-cost coordinate。这是刻意的、与
calibration 对齐的 controlled stress，不是自然 clinical shift、疾病严重度定义或
causal treatment-effect estimator。

每个 science seed 使用 3,000 条 calibration trajectories（前 1,000 条冻结 grid）、
20,000 条 fresh reference trajectories；ACI、SPCI、PRC 各自另用 2,000 条
target-policy adaptation trajectories。Target-policy 单步 ratio cap 为 3；SC-PCP
累计 committed-prefix calibration weights 不截断。WSC 与 paired differences/ratios
使用 10,000 个共享的 complete-seed bootstrap resamples；mean width 的绝对区间使用
selected seeds 上的 two-sided Student-\(t\) interval，selection rate 使用 Wilson
interval。

Science 启动前依序执行：

1. **Support gate**：每个 stage/action 至少 20 个 unique donor patients；至少
   19/20 seeds 必须通过。
2. **K0 fidelity gate**：每 seed 进行 16 个 systematic logging-mixture one-step
   replays。任一 exact structural invariant 失败即 `IMPLEMENTATION_INVALID`；在结构
   全通过后，score KS \(\le0.10\)、signed-residual W1 \(\le0.25\)、successor-mean
   W1 \(\le0.25\)、successor-q95 W1 \(\le0.50\) 的 numeric fidelity 仍须至少
   19/20 seeds 可用。
3. **Donor-overlap screen**：在 \(\gamma=-4\) 的 q-mid 与 q-high/max-response
   两个 3,000-trajectory probes 上检查 patient-aggregated local ESS 与 donor
   probabilities。冻结阈值为 local-ESS 1% quantile \(\ge10\)、median ESS fraction
   \(\ge0.25\)、maximum donor probability \(\le0.25\)。低 overlap 只允许
   descriptive curves；通过也只是 empirical interpretation screen，不是 positivity
   或 coverage guarantee。

Support/K0 失败的 consequence 是不生成 scientific score/coverage rows，不能在
图中画插值线、旧协议曲线或其他数据集的替代值。

### 10.3 Gate 结果：一个 CURVES，三个 K0 NO-GO

| Dataset-native setting | Horizon | Support | K0 fidelity | Donor overlap | Final panel status | Scientific rows |
|---|---:|---:|---:|---:|---|---|
| MIMIC-IV | 12 | 20/20 PASS | 20/20 PASS | 20/20 PASS | `CURVES` / `EMPIRICAL_OVERLAP_SCREEN_PASSED` | Yes |
| eICU | 12 | 20/20 PASS | 12/20 FAIL | Not opened | `K0_FIDELITY_NO_GO` | **No** |
| INSPIRE | 12 | 20/20 PASS | 13/20 FAIL | Not opened | `K0_FIDELITY_NO_GO` | **No** |
| MIMIC-CXR + IV/ED | 6 | 20/20 PASS | 10/20 FAIL | Not opened | `K0_FIDELITY_NO_GO` | **No** |

三个 NO-GO setting 没有 baseline ranking，也没有 SC-PCP coverage failure 或成功值；
正确读法是 frozen controlled transition 没有在足够多 seeds 上复现 logging-mixture
one-step law，所以后续 science 未获授权。Production/native suite 中这三个数据集的
完整六方法曲线仍然有效，但属于不同 estimand，不能拿来填本表。

### 10.4 MIMIC-IV \(\gamma=-4\) 六方法结果

MIMIC-IV 的完整五点 signed curve均已保存；预先声明的 confirmatory endpoint 是
\(\gamma=-4\)，其他四点标为 descriptive signed-control curve。下表的 WSC 均按

\[
\operatorname{WSC}=\min_t\frac1{20}\sum_{s=1}^{20}C_{s,t}
\]

计算，不是 `mean_seed(min_t C_seed,t)`。WSC 方括号是 artifact 中冻结的
complete-seed-vector bootstrap 95% CI，width 方括号是跨 selected seeds 的
Student-\(t\) 95% CI；六方法都成功选择 20/20 seeds。

| Method | WSC [95% CI] | Mean normalized width [95% CI] | Point-eligible |
|---|---:|---:|:---:|
| Standard CP | 0.86358 [0.85912, 0.86575] | 4.20938 [4.08648, 4.33228] | No |
| ACI | 0.88061 [0.87745, 0.88140] | 4.54100 [4.41991, 4.66209] | No |
| MFCS | 0.91894 [0.91159, 0.92456] | 5.98445 [5.63318, 6.33573] | Yes |
| SPCI | 0.89769 [0.89306, 0.89913] | 5.02748 [4.88802, 5.16694] | No |
| PRC | 0.87590 [0.86794, 0.87800] | 4.55542 [4.42127, 4.68957] | No |
| **SC-PCP** | **0.90089 [0.89431, 0.90105]** | **5.06708 [4.93983, 5.19433]** | **Yes** |

Point eligibility 沿用预先声明的 `WSC >= 0.90` 且 `Selection >= 0.95` 规则；它不是
把 95% CI 下端也要求高于 0.90 的 certification rule。因此 SC-PCP 的点估计达标，
并且是两个 point-eligible methods 中更窄者，但其 CI 明确跨过 0.90，不能写成
“95% 证据证明 coverage”或 finite-sample guarantee。MFCS 的 coverage 更高且 CI
在 0.90 上方，但集合显著更宽；超过 0.90 越多不自动表示更有效。

Seed-paired SC-PCP comparisons 为：

| Baseline | SC-PCP minus baseline WSC [95% CI] | SC-PCP / baseline geometric width [95% CI] |
|---|---:|---:|
| Standard CP | +3.731 pp [3.163, 3.907] | 1.2044 [1.1945, 1.2151] |
| MFCS | -1.805 pp [-2.715, -1.279] | 0.8518 [0.8122, 0.8884] |

这支持的结论是：在一个通过 fidelity/overlap gates 的 dataset-native MIMIC-IV
controlled stress cell，SC-PCP 相对 Standard CP 大幅纠正 undercoverage，并以约
20.4% 的 width 增量换取该纠正；相对 MFCS，SC-PCP 更窄但 coverage 低 1.81 pp。
它不支持 universal dominance、自然 MIMIC-IV performative effect 或 causal efficacy。

### 10.5 与 2026-08-25 MIMIC v1 的不可混用边界

2026-08-25 的 `controlled_six_method_confirm20_20260825` 仍是有效、正式的 v1
protocol-specific result；其 \(\gamma=-4\) SC-PCP WSC 为 0.898277，不能被 v2 的
0.900887 覆盖或改写。V2 使用新的 patient-role split、dataset-native gates、seed
bank、source tree 与 controlled-transition contract。两者都可作为各自协议的证据，
但不得挑选较好数值、合并 seeds、拼接 CI 或把 v2 称为对 v1 的同分布复现。

### 10.6 Provenance、两次透明 retry 与最终工件

最终工件绑定：

```text
formal_root             = results/work/controlled_clinical_extension_v2
root_manifest_sha256    = 06996fee1f6eeed861a06ff2802253bebda1eaddb8e0e84b5c6577c07d599db0
source_tree_sha256      = e929fd61e2671190cc2daf10df2ca8168fb1b9131e421321fe542d539a75259d
source_snapshot_sha256  = e0191329e036d05caff7d4b72e661e0a05cef8fc0e0d12118c7809021b773f91
source_manifest_sha256  = 28919b8e25fa4159f7dd74592cb419748fadb8d3f3bd187de70814d760581f4a
```

正式完成前发生两次工程 retry，均保留 failure archive 并绑定 amendment：

1. `precoverage_cuda_indexed_median_retry_20260826`：deterministic CUDA 拒绝
   indexed `median(dim=1)` kernel。失败发生在 MIMIC-IV K0、任何 coverage/science
   打开之前；修复以 exact lower-median index 替换 kernel，数学语义、DGP、gate、
   seed 与 threshold 均不变。Amendment SHA-256 为
   `56ee4b43f503ce05beb0a25c47eb9a1739bb9210bc638405b7678e3c5cedf4df`；
   failure tar SHA-256 为
   `0fec4f676d86dce583bd135622bd7e888d0dac0198aa63a12d3b64ef96664906`。
2. `postcompute_preinspection_json_key_order_retry_20260826`：canonical JSON 使用
   `sort_keys=True`，但两个 reload validators 错把 object key order 当作语义。
   MIMIC-IV scientific rows 已生成，但 coverage/science values 未被查看或用于修改；
   修复只把 ordered-key equality 改成 exact key-set equality，并保留全部 value/type/
   cross-field checks。相同预设 seeds 的所有阶段从零重算，旧结果不复用。
   Amendment SHA-256 为
   `201ee0bea4cce868345c9f69ca3f296c77af4cb90b52debdfab95d4e1c082fa4`；
   failure tar SHA-256 为
   `bfd28a92a574bac0e25e3ec5f3b03ef5c5c33ef319ac996e57b766e292d9e54e`。

这两次 retry 是可审计的 engineering corrections，不是 result-guided tuning。最终
metadata 明确记录 `canonical_scpcp_mutation_permitted=false`、
`existing_mimic_v1_substitution_permitted=false`，并在正式 launch 前完成 1,304 个
新 RNG streams 的零碰撞审计。

### 10.7 可以与不可以写什么

可以写：

- dataset-native MIMIC-IV controlled environment 通过预设 fidelity 与 overlap gates；
- 在该 \(\gamma=-4\) cell，SC-PCP 相对 Standard CP 提高 3.73 pp WSC，并将
  point WSC 提到 0.90089，但付出 1.204 的 width ratio；
- eICU、INSPIRE、MIMIC-CXR + IV/ED 是正式 K0 NO-GO，说明 controlled transition
  construction 不能无条件跨临床数据集迁移；
- 负结果与通过结果共同限定方法在受控机制 benchmark 中的适用范围。

不可以写：

- 四个 clinical datasets 都产生了 controlled \(\gamma=-4\) curves；
- 用 production/native curves、Synthetic \(\beta=2\) 或旧 MIMIC v1 替换三个
  NO-GO panels；
- SC-PCP 的 95% CI 已证明 0.90 coverage，或该结果是 finite-sample certificate；
- 这是自然 clinical performative-treatment shift、真实 treatment effect、跨数据集
  conjunction、universal SOTA 或 universal superiority evidence。

确定性渲染输出为：

- [`figure_stagewise_profiles.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_stagewise_profiles.pdf)：
  2026-08-22 production/native RQ1 的五个 setting，每个 setting 均有六方法曲线；
- [`figure_controlled_stress_grid.pdf`](../results/paper_five_setting_stage_profiles_20260826/figure_controlled_stress_grid.pdf)：
  separate Synthetic native \(\beta=2\)、MIMIC-IV clinical v2 \(\gamma=-4\) curves，
  以及三个 clinical v2 K0 NO-GO gate cards。

Paper directory 只含上述两个 PDF。Source CSV、setting-status contract、method summary、
editable SVG、600-dpi TIFF、PNG preview、QA 和 render manifest 位于
[`results/work/five_setting_stage_profiles_20260826`](../results/work/five_setting_stage_profiles_20260826)。
Renderer 只读冻结工件，不重跑任何 model、rollout、scientific seed 或 bootstrap。
