# Introduction

A prediction set can change the future it is supposed to cover. Consider a
longitudinal care system in which a wide set prompts a cautious action, while a
narrow set supports a different one. The chosen action changes the next
outcome and may also change the state observed at the following visit. The next
prediction error is therefore drawn from a distribution partly created by
earlier prediction sets. Once uncertainty guides action, the uncertainty
report becomes part of the data-generating process.

Standard split conformal prediction is designed mainly for a more passive
setting. Under exchangeability, it can give finite-sample marginal coverage
without requiring a correct parametric outcome model
([Shafer and Vovk, 2008](https://www.jmlr.org/papers/v9/shafer08a.html);
[Lei et al., 2018](https://doi.org/10.1080/01621459.2017.1307116)). Weighted
conformal methods allow certain shifts when the change in distribution is
fixed independently of the calibrated prediction set
([Tibshirani et al., 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html)).
Our setting breaks this separation between calibration and deployment. A
radius \(q_t\) determines the size of the prediction set, but it also changes
the action policy \(\pi_t^{q_t}\). That policy then changes the distribution of
the error \(R_t\) that the same radius must cover:

\[
q_{0:t}\;\longrightarrow\;\pi^{q_{0:t}}
\;\longrightarrow\;P_{q_{0:t}}(R_t).
\]

A radius can therefore be calibrated on historical trajectories and still
fail after deployment because its own use changes which trajectories occur.
We study how to choose a separate radius at each stage using only logged
longitudinal trajectories, with no outcomes from the policy that will
eventually be deployed.

Several lines of research address parts of this feedback problem.
Performative prediction formalises how a predictor can alter its evaluation
population
([Perdomo et al., 2020](https://proceedings.mlr.press/v119/perdomo20a.html)).
Conformal extensions have considered agent-induced shifts, prediction for an
externally specified sequential policy, and repeated recalibration using data
from successive deployments
([Prinster et al., 2024](https://proceedings.mlr.press/v235/prinster24a.html);
[Zhang et al., 2023](https://proceedings.mlr.press/v206/zhang23c.html);
[Li et al., 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6c71e8beb41e142e463b16818537ed0-Abstract-Conference.html)).
Recent methods also protect policies selected with calibration data or a
single contextual decision affected by a prediction set
([Prinster et al., 2026](https://arxiv.org/abs/2603.02196);
[Zheng and Jin, 2026](https://arxiv.org/abs/2607.02206)). The remaining
longitudinal difficulty is different. The current radius changes both what
counts as coverage and the action that generates the next outcome. Earlier
radii have already changed which states reach the current stage. Thus the
target distribution is neither fixed in advance nor determined by the current
decision alone.

The order of events reveals the required correction. Past actions determine
which states arrive at stage \(t\). The current action determines the outcome
and prediction error observed immediately afterwards. Future actions cannot
affect an error that has already occurred. Under standard sequential causal
assumptions, the reweighting must include action likelihood ratios up to and
including the current action, but exclude future actions. The historical
ratios account for the distribution of states
reached at the current stage. The current ratio accounts for the action that
produces the next outcome. A current-only correction misses the first channel,
whereas a history-only correction misses the second. We call this required
current-plus-history boundary the **committed prefix**. Our identification
result locates the causal boundary that determines the distribution of each
prediction error observed after the current action.

SC-PCP turns this insight into a forward, fully offline procedure. At a given
stage, radii chosen earlier are treated as committed decisions. For every
candidate current radius, SC-PCP considers the policy prefix that the candidate
would induce and estimates coverage under that candidate's own distribution.
It also estimates prediction-set width under the same distribution. Among
the candidates whose estimated coverage meets the target, the method selects
the one with the lowest estimated width at the current committed prefix. It
then commits that radius and moves to the next stage. The result is a schedule
with a separate radius for each stage, rather than one shared radius or a
prespecified profile.
Because changing a radius changes both the threshold and the action policy,
coverage need not vary monotonically across candidates; each candidate is
therefore evaluated directly.

This data-dependent schedule creates a second statistical challenge. The same
calibration sample helps define the candidates, evaluates them, and selects the
final schedule, so a theorem for one fixed policy is not enough. We assume that
coverage estimates converge uniformly over all bounded radius prefixes.
Together with adequate overlap and uniformly consistent estimates of logging
action probabilities, this controls the random choice of schedule. Provided
that SC-PCP returns a complete schedule with probability tending to one, we
prove that its fresh-deployment coverage is at least
\(1-\alpha-o_p(1)\) at every fixed stage. The result is asymptotic and
stagewise. It is not a finite-sample, data-conditional, or whole-trajectory
guarantee.

The experiments are organised to test the argument rather than merely rank
methods. Exact finite-state systems isolate the historical and current-action
channels, showing when either partial correction targets the wrong
distribution. Horizon, overlap, and calibration-size studies expose the cost
of longer prefixes and less informative logged data. Controlled experiments
vary the direction and strength of feedback to test whether calibration
responds to the policy-induced shift. Across four separate settings, SC-PCP's
estimated worst-stage coverage ranged from 89.93% to 90.02% without using
target-policy outcomes. A more conservative offline comparator covered more.
The comparison was therefore not coverage-matched, but SC-PCP's prediction
sets were 6.4% to 14.2% narrower. These setting-specific comparisons are
consistent with the nominal stagewise target, but they do not provide
finite-sample certification.

Our contributions are threefold:

1. **Problem formulation and identification.** We
   formulate stagewise coverage when prediction radii determine both the sets
   and the policy that generates future errors. We then show why the induced
   distribution is identified by the committed action prefix.

2. **An offline selector with validity after selection.** SC-PCP evaluates
   coverage and width under each candidate's own induced distribution. At each
   committed prefix, it selects the lowest-width candidate whose estimated
   coverage meets the target. We provide asymptotic stagewise marginal coverage
   for the final data-dependent schedule.

3. **Evidence that separates necessity, benefit, and cost.** Our experiments
   test why both feedback channels matter and how their correction changes the
   coverage-width trade-off. They also show how horizon, overlap, sample size,
   and propensity estimation limit the method.

The central message is simple: when prediction sets change sequential
decisions, calibration must follow the decisions that those sets put into
motion.
