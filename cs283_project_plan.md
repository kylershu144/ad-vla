**Evaluating Uncertainty Quantification and Failure Detection\
for Adversarial Vision Attacks on Autonomous Driving VLAs**

Kyler Shu

Stanford University

CS 283 -- Governing Artificial Intelligence: Law, Policy, and
Institutions

Fall 2026

# Finalized Research Question

**To what extent do uncertainty quantification and failure detection
(UQ/FD) techniques identify adversarial vision attacks in autonomous
driving vision-language-action (VLA) models?**

I test adversarial patch attack methods against AutoVLA
[@zhou2025autovla], a representative open-code autonomous driving VLA,
which will address the sub-question: *to what extent are autonomous
driving VLAs affected by adversarial patch attacks?* I then apply
state-of-the-art failure detection methods to AutoVLA and test this
augmented model against adversarial patch attacks, realized as textured
objects that can be placed in the vehicle's camera view. A UQ/FD method
succeeds to the extent that it flags the resulting failures early enough
to trigger a fallback. The primary outcome is therefore detection
performance (ROC) at a fixed interval prior to collision, reported
alongside the attack's effect on driving scores.

# Core Contribution

This project addresses three gaps in the existing literature.

1.  **Applying and assessing UQ/FD for driving VLAs.** UQ and failure
    detection for VLAs has centered on manipulation, including
    token-probability confidence calibration [@zollo2025calibration] and
    latent-feature failure detection [@gu2025safe]. I adapt these
    methods to driving, where outcomes are more complex than binary task
    success.

2.  **Testing driving VLAs with adversarial patch attacks.** Prior
    analyses of attacks on autonomous driving targeted CNN-based models
    [@deng2020analysis] that are neither end-to-end nor
    language-conditioned the way VLAs are. Furthermore, these
    adversarial attacks relied on image perturbation methods that are
    fairly infeasible attack in an autonomous driving setting, unlike a
    patch attack that can take form as a physical object in the
    vehicle's environment.

3.  **Adapting patch attacks from manipulation to navigation.** Patch
    attacks on VLAs have been demonstrated on manipulation-centric
    models [@wang2025adversarial; @xu2025edpa; @yang2026uao]. There is
    no existing work that tests them on driving VLAs or pairs them with
    UQ/FD.

4.  **Informing evidence requirements for driving-automation
    standards.** Beyond the technical findings, this study speaks to a
    governance gap. SAE J3016 classifies driving automation by who
    performs the dynamic driving task and its fallback, but it is a
    taxonomy that sets no performance criteria [@sae2021j3016]. Levels 3
    and 4 depend on the system recognizing its own limits. At Level 3,
    the system must detect when to request that a fallback-ready user
    intervene. At Level 4, it must detect when to execute a minimal-risk
    maneuver itself. Safety standards such as ISO 21448 (SOTIF), which
    addresses performance limitations and insufficient perception
    [@iso2022sotif], and ISO/PAS 8800, which addresses safety and AI in
    road vehicles [@iso2024pas8800], address related hazards. However,
    neither specifies evidence that a learned system's self-monitoring
    still works when its visual input is deliberately manipulated, which
    the cybersecurity-engineering standard ISO/SAE 21434 treats as a
    separate concern [@iso2021cyber]. My results can ground a concrete,
    testable evidence criterion: for a stated operational design domain,
    a developer claiming a given automation level would report a failure
    detector's true-positive rate at a fixed false-positive rate, and
    its time-to-detection relative to the fallback latency that level
    requires, against physically realizable patch attacks. If UQ/FD
    methods perform well under attack, the study supports using them as
    auditable runtime evidence for human handover and
    minimal-risk-maneuver claims. If they degrade, it shows that
    confidence-based safeguards alone cannot support such claims, and
    that regulators should require adversarial testing rather than
    clean-condition calibration.

# Methods

## Adversarial Patch Generation

I will generate patches with method 1, with method 2 as a fallback.
Method 1 is preferable because it treats the internal state of the VLA
as inaccessible, which is the most realistic scenario for an attack.
Method 2 is tuned to the visual and textual latent representations,
which is VLA specific and may not be open knowledge in all cases. For
both methods, the final patch will be mapped onto a 3D object that can
be inserted into simulation environments. This work is simulation-only,
and the final optimized patch textures will not be posted publicly.
Findings will also be shared with the model developers.

**Method 1: multi-objective patch optimization.** Following the
Universal Adversarial Object of @yang2026uao, I optimize a patch with a
multi-objective loss combining trajectory deviation, task failure,
action perturbation, and visual naturalness. Because gradients inside
VLA models are generally inaccessible, that work estimates them with
Simultaneous Perturbation Stochastic Approximation (SPSA) and requires
only training data. Since the patch must exist as an object in the
vehicle's environment, I must additionally enforce invariance to scale,
rotation, and translation. I will optimize in closed loop in CARLA. The original paper runs on a very large GPU (NVIDIA A100 80G) for a week, so to reduce compute I may experiment with optimizing DCT coefficients instead of a 128x128 patch, or use a smaller patch. 

**Method 2: embedding-disruption patch.** Following the Embedding
Disruption Patch Attack (EDPA) [@xu2025edpa], I maximize the discrepancy
between visual and textual latent representations, and between the
latent representations of adversarial and clean images. EDPA is
model-agnostic: it instead uses the vision and language encoder
embeddings but not full access to model parameters, unlike earlier
patches whose action-discrepancy and geometry-aware objectives require
gradients through the whole model [@wang2025adversarial]. The encoders
should be publicly available for AutoVLA. A limitation of this method is that the embedding discrepancy is only a proxy for poor VLA behavior, which may not necessary result from a affected internal state.

## UQ/FD Techniques

**SAFE.** The SAFE failure detector [@gu2025safe] trains an MLP or LSTM
on the VLA's internal features during rollouts, regresses them to a
scalar failure score, and sets the alert threshold with conformal
prediction. It needs white-box access and a set of successful and failed
rollouts to train, which I will source from CARLA. SAFE formulates
outcomes as binary success or failure, which may not suit driving. A
natural binary adaptation is obstacle collision. I will also experiment
with continuous indicators such as trajectory deviation or minimum
distance to obstacles.

**Token-probability baselines.** Following @zollo2025calibration, I
aggregate the probabilities of a VLA's generated action tokens using
simple functions such as the mean and maximum. That work also applies
action-wise Platt scaling, which maps probabilities to calibrated
confidence scores across heterogeneous action dimensions, and I will
test it if time allows. Evaluating calibration with Expected Calibration
Error (ECE) requires a notion of when an output is accurate, which needs
to be defined for driving applications.

## Evaluation Protocol

For each VLA and UQ/FD method, I compare clean scenes with scenes
containing the adversarial object. I measure the attack's effect with
the CARLA benchmark scores, and the failure detection with ROC curves
for each UQ/FD method.

# Measuring Success

The project succeeds if it produces the following measurable outcomes.

-   A 3D object with an adversarial patch mapped onto it that can be
    placed into simulation environments.

-   The effect of the patch attack on CARLA benchmark scores, compared
    across the two driving VLAs.

-   ROC values for each UQ/FD method applied to AutoVLA.

-   The detection success of each UQ/FD method under attack, measured by
    ROC values.

The project does not depend on UQ/FD performing well. A finding that
these methods degrade under adversarial patches would directly answer
the research question.

# Resources and Access

-   **Simulation and benchmarking:** an environment where adversarial
    objects can be inserted, most likely the CARLA simulator
    [@dosovitskiy2017carla], plus benchmark scores for assessment.

-   **Patch training data:** AutoVLA inference on CARLA scenarios
    suffices for Method 1. For Method 2, the open-source Qwen vision and
    language encoders used by the VLAs provide the embeddings.

-   **Failure-detector training data:** Rollouts in CARLA can be reused
    to train and calibrate SAFE.

-   **Compute:** patch generation was performed on large GPUs. I only
    have a local laptop GPU, but it may be sufficient for failure
    detection training and running inference for AutoVLA since it does
    not need to run at realtime speed in sim.

# Potential Obstacles and Fallback Plans

-   **Compute for patch generation.** @yang2026uao optimized their patch
    for about one week on a very large GPU (NVIDIA A100 80GB). I do not
    have access to compute at this scale. *Fallback:* use a smaller
    patch size.

-   **Redefining failure for driving.** UQ/FD methods built for
    manipulation assume binary success or failure, which may not
    transfer. *Fallback:* test collision rate, minimum distance to
    obstacles, and trajectory deviation as alternative definitions.
    Adapting these methods is itself a source of novelty.