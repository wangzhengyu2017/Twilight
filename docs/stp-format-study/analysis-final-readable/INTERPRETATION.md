Llama 3.1-8B-Instruct, frozen WikiText-2: 289,075 targets per full run. Each point is a separate model trajectory on the same corpus. There are no independent corpus replicates, error bars, or statistical significance claims. Lines connect thresholds for readability.

Q4K4

- p=0.85, SM: PPL=6.675108, selected/allowed=0.270309; ΔCE vs TC=-0.171200497, vs INT Top-p=+0.009056380; lower CE with more selected-key expenditure than TC (fraction difference=+0.153479).
- p=0.85, MXINT8-TC4: PPL=7.238384, selected/allowed=0.090393; ΔCE vs TC=-0.090187876, vs INT Top-p=+0.090069001; lower CE with fewer selected-key expenditure than TC (fraction difference=-0.026436).
- p=0.85, MXFP4: PPL=6.638278, selected/allowed=0.080939; ΔCE vs TC=-0.176733333, vs INT Top-p=+0.003523545; lower CE with fewer selected-key expenditure than TC (fraction difference=-0.035890).
- p=0.90, SM: PPL=6.557247, selected/allowed=0.351271; ΔCE vs TC=-0.109720184, vs INT Top-p=+0.003737869; lower CE with more selected-key expenditure than TC (fraction difference=+0.197358).
- p=0.90, MXINT8-TC4: PPL=6.921718, selected/allowed=0.123240; ΔCE vs TC=-0.055626966, vs INT Top-p=+0.057831086; lower CE with fewer selected-key expenditure than TC (fraction difference=-0.030673).
- p=0.90, MXFP4: PPL=6.544645, selected/allowed=0.115115; ΔCE vs TC=-0.111643879, vs INT Top-p=+0.001814174; lower CE with fewer selected-key expenditure than TC (fraction difference=-0.038797).
- p=0.95, SM: PPL=6.497612, selected/allowed=0.489443; ΔCE vs TC=-0.050888745, vs INT Top-p=+0.001936922; lower CE with more selected-key expenditure than TC (fraction difference=+0.259216).
- p=0.95, MXINT8-TC4: PPL=6.673347, selected/allowed=0.193648; ΔCE vs TC=-0.024201986, vs INT Top-p=+0.028623681; lower CE with fewer selected-key expenditure than TC (fraction difference=-0.036578).
- p=0.95, MXFP4: PPL=6.492194, selected/allowed=0.189258; ΔCE vs TC=-0.051722988, vs INT Top-p=+0.001102679; lower CE with fewer selected-key expenditure than TC (fraction difference=-0.040969).

Q8K4

- p=0.85, SM: PPL=6.697492, selected/allowed=0.137980; ΔCE vs TC=+0.004514094, vs INT Top-p=+0.012404109; higher CE with more selected-key expenditure than TC (fraction difference=+0.072315).
- p=0.85, MXINT8-TC4: PPL=6.675618, selected/allowed=0.062355; ΔCE vs TC=+0.001242699, vs INT Top-p=+0.009132714; higher CE with fewer selected-key expenditure than TC (fraction difference=-0.003310).
- p=0.85, MXINT8-Q8 / MXFP4-K4: PPL=6.603273, selected/allowed=0.068868; ΔCE vs TC=-0.009653633, vs INT Top-p=-0.001763619; lower CE with more selected-key expenditure than TC (fraction difference=+0.003204).
- p=0.90, SM: PPL=6.574305, selected/allowed=0.191876; ΔCE vs TC=+0.001603487, vs INT Top-p=+0.006336009; higher CE with more selected-key expenditure than TC (fraction difference=+0.099654).
- p=0.90, MXINT8-TC4: PPL=6.571011, selected/allowed=0.087135; ΔCE vs TC=+0.001102213, vs INT Top-p=+0.005834735; higher CE with fewer selected-key expenditure than TC (fraction difference=-0.005086).
- p=0.90, MXINT8-Q8 / MXFP4-K4: PPL=6.530161, selected/allowed=0.098276; ΔCE vs TC=-0.005133764, vs INT Top-p=-0.000401243; lower CE with more selected-key expenditure than TC (fraction difference=+0.006055).
- p=0.95, SM: PPL=6.498413, selected/allowed=0.297910; ΔCE vs TC=-0.000288823, vs INT Top-p=+0.002060136; lower CE with more selected-key expenditure than TC (fraction difference=+0.146086).
- p=0.95, MXINT8-TC4: PPL=6.503503, selected/allowed=0.142893; ΔCE vs TC=+0.000494143, vs INT Top-p=+0.002843102; higher CE with fewer selected-key expenditure than TC (fraction difference=-0.008931).
- p=0.95, MXINT8-Q8 / MXFP4-K4: PPL=6.484025, selected/allowed=0.163698; ΔCE vs TC=-0.002505322, vs INT Top-p=-0.000156363; lower CE with more selected-key expenditure than TC (fraction difference=+0.011873).

The selected fraction is the ratio of total selected to causal-allowed key pairs, summed over layers, heads and full input windows; it excludes future-masked pairs. Read quality and global selected-key expenditure together. Lower PPL with a larger fraction is compatible with a budget contribution; lower PPL with a smaller fraction is a descriptive improvement on these measured axes. Neither comparison identifies a unique causal mechanism or kernel speed. No matched-budget intervention was executed.

Saved same-input smoke comparisons isolate predictor differences on each candidate’s own operands; different candidates generally follow different trajectories. The 24 layer0 captures with exact operand equality permit a local same-input contrast, not attribution of corpus NLL. SM’s higher retained mass and key counts must be read with its rank and set precision diagnostics. Temperature flattening can increase crossing counts at fixed ranks, but SM is not established to be a pure temperature change.

Scalar-prefix entropy loss and conditional variance establish many-to-one scalar loss under their specified conditioning. They do not bound whole-predictor PPL or exclude side information. TC mixed-term identities explain local relative-logit distortion and cancellation; common row shifts cancel under softmax. MX additionally changes block scales and requantization, so its result does not isolate bit encoding alone.

Conditional FP4 was activated and is follow-up evidence.

Dense FP/INT lines are actual validated historical full-corpus references, with their existing parser-matched-gate limitation. RS remains source blocked and has no numeric substitute. Supplemental experiments remain proposals, unapproved and unexecuted by this tool.
