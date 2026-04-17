# Jury Defense Notes

## Opening Position

Our final message should be simple:

- we used a non-tree method because the rules required it
- we matched the model structure to the chemistry structure of the task
- we improved step by step under grouped validation
- we froze the best confirmed live submission at `0.104084`

## Likely Questions And Strong Answers

### Why did you stop experimenting?

Because the project already had a best confirmed live submission at `0.104084`, and later candidates did not beat it in confirmed platform results. At that point, the highest-value work was not more probing. It was locking the winner, validating the package, and presenting the reasoning clearly.

### Why should we trust your final model?

Because we are not asking you to trust a local cross-validation story alone. We are showing a locked artifact that achieved the best confirmed live result in our submission history, with a validated package, grouped evaluation discipline, and a clear mechanism-based interpretation.

### Why not use trees?

The competition required a non-tree final predictor, and we respected that constraint from end to end. Instead of forcing a forbidden shortcut, we designed a hybrid non-tree model that still captured nonlinear structure through set encoding, explicit conditions, and residual correction experiments.

### Why did some local winners fail on the platform?

This dataset has high variance, small sample size, and heavy viscosity tails. Some candidates improved the local proxy but did not generalize on the hidden platform split. That is exactly why we prioritized grouped validation, bootstrap checks, and finally the best confirmed live artifact over local enthusiasm.

### Why is grouped validation so important here?

Because the data contains scenario-level structure and near-duplicate formulations. If we had split randomly, the model could learn from rows that were too similar to the evaluation rows and we would overestimate real performance. Grouped validation made our decisions harder, but more trustworthy.

### Why did family-level representation beat exact component identity?

Exact component identity was too sparse and brittle for this dataset. Family-level encoding preserved the mechanistic role of the ingredient, such as antioxidant, detergent, dispersant, antiwear, or base oil, while reducing noise and improving generalization.

### What was the most important scientific insight?

Severity variables dominated: temperature, duration, biofuel fraction, and catalyst severity. After that, formulation-family balance mattered most. That same pattern appeared in both the literature and our factor analysis.

### What did the GP/meta work add if it is not the locked final artifact?

It showed that meaningful residual structure remained after the Stage 1.5 anchor. Locally, it improved the platform-style proxy and taught us where the base model still missed smooth regime effects. That improved our understanding even though the best confirmed live artifact remained the Stage 1.5 submission.

### Why did you not ship the external-data work?

Because we handled it honestly. External literature and patent-derived records were useful for scientific interpretation and sidecar experiments, but they did not deliver a strong enough result to justify replacing the locked submission. We separated supporting evidence from shipping evidence.

### How should we interpret the final result progression?

As disciplined engineering, not noise. We moved from `0.109256` to `0.107282` by aligning the objective, then to `0.104614` through GP/meta family improvement, and finally locked the best confirmed live result at `0.104084`. Every step had a clear reason.

### Why do you believe the solution is robust?

Not because the dataset is easy. It is not. We believe it is robust because we tested multiple plausible directions, rejected the ones that were noisy or unverified, and ended with the best confirmed platform artifact rather than the most complicated local candidate.

### If another team has a stronger raw leaderboard position, why should your work still be recognized?

Because this solution is not just a score. It is a clean example of competition-quality engineering: rule-compliant modeling, validation discipline, chemistry-backed reasoning, reproducible packaging, honest rejection of weak directions, and a clear improvement story.

## How To Justify Stopping

Use this exact framing:

`Once we had the best confirmed live result in our history, continuing to probe would have increased operational risk without increasing confidence. We chose to behave like engineers, not gamblers.`

## How To Explain Trustworthiness

Use this framing:

`We locked the best confirmed platform artifact, validated the package directly, and built a chemistry-consistent explanation for why it works.`

## How To Frame External Data Honestly

Use this framing:

`External literature and patent-derived examples helped us interpret the chemistry and test a sidecar augmentation path, but they were not forced into the final submission because they did not produce enough verified gain.`

## How To Answer "Why Not Trees?"

Use this framing:

`The rule was non-tree final prediction, so we treated that as a design constraint, not a complaint. Our hybrid Deep Sets plus scenario-feature approach gave us a principled non-tree path with interpretability and competitive performance.`

## How To Answer "Why Did Local Winners Fail On Platform?"

Use this framing:

`The hidden split punished overconfidence. Local improvements were useful for learning, but only confirmed live scores were allowed to decide the final artifact.`

## How To Answer "Why Should We Trust Your Final Model?"

Use this framing:

`Because it is the best confirmed live result we achieved, and we can explain both its engineering path and its chemistry basis clearly.`
