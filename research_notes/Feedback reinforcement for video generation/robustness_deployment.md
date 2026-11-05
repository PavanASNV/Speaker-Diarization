# Reward hacking, robustness and safe deployment of feedback-driven learning loops for generative video (state as of 5 Nov 2026)

Target system: Prompt-to-Scene Creator. Claude plans scenes and writes prompts for hosted text-to-video models (Seedance 2.0, Veo 3.1, Kling 3.0, Sora 2, Wan 2.2). A vision-LLM judge scores 7 dimensions. User ratings and A/B choices update (a) Thompson-sampling posteriors over LLM-written prompt strategies, rewritten by LLM "reflection", (b) online judge-weight calibration and (c) logged preferences for Flow-GRPO/MixGRPO on Wan 2.2, later an in-house DiT. The goal is photoreal live action, never a 3D-animation look.

Source conventions used below:
- **(secondary)**: the detail was seen only in an aggregator or review page, or in a search-result summary, not in the primary text.
- **(K)**: an established paper (mostly 2022–2024) cited to its canonical arXiv page from the researcher's prior knowledge of its content. arXiv cannot be fetched in this environment, so these were not re-checked this session.
- **(listed)**: a paper known only from its one-line entry in the curated GitHub list [awesome-rl-for-video-generation](https://github.com/chrisliu298/awesome-rl-for-video-generation). It was not opened.

Research limits: the shared web-search budget ran out partway through. The network policy blocked deploymentsafety.openai.com, digital-strategy.ec.europa.eu, labs.cloudsecurityalliance.org and rl-diffusion.github.io.

---

## 1. Documented reward hacking in text-to-image/video RL and in VLM-as-judge setups (which papers quantify it?)

### Takeaway
When diffusion or flow models are RL-tuned against a single learned scorer, reward hacking is the expected outcome, not an edge case. Documented modes include:
- oversaturation and colour drift;
- over-stylisation, including an "unnaturally dark" style;
- diversity collapse (the same faces and camera angles);
- "copy-paste", expressionless faces under identity rewards;
- in video, drift toward near-static clips, because both human annotators and video reward models prefer artifact-free slow motion.

LLM and VLM judges can also be gamed: by single "master-key" tokens, by injected instructions, and by text rendered inside images. The best-quantified 2025–2026 cases are GRPO-Guard, IPRO, SHIFT, DDRL and One-Token-to-Fool.

### Cited Findings

**Image (T2I) diffusion/flow RL**
- **Flow-GRPO** (May 2025): the authors say the KL term "effectively prevents reward hacking, where reward increases at the cost of image quality or diversity". With a suitable KL, training reached the KL-free reward while keeping quality, but took longer. They describe their PickScore runs as showing "very little reward hacking", with DrawBench quality metrics "essentially unchanged" — [Flow-GRPO, arXiv HTML v4](https://arxiv.org/html/2505.05470v4) (via search snippet).
- **GRPO-Guard** (Oct 2025, CVPR 2026) measures "implicit over-optimization": under Flow-GRPO the proxy reward keeps rising while the gold score falls.
  - In its PickScore comparison, Flow-GRPO "suffers from a drastic degradation in diversity, detail richness, visual quality, and text-image consistency" — [alphaXiv overview](https://www.alphaxiv.org/overview/2510.22319v2); [arXiv HTML](https://arxiv.org/html/2510.22319v1).
  - This contradicts Flow-GRPO's own description. Neither source shows whether the two papers used the same training budgets or KL coefficients.
- **GRPO-Guard's diagnosis:** the importance-ratio distribution is shifted left (mean below 1) and its variance differs across denoising steps. As a result, PPO-style clipping fails to bound updates from over-confident positive-advantage samples. Low-noise steps produce disproportionately large gradients, which overfit specific noise conditions and neglect early-step exploration and diversity — [papernotes CVPR 2026 summary](https://en.papernotes.org/CVPR2026/image_generation/grpo-guard_mitigating_implicit_over-optimization_in_flow_matching_via_regulated_/) (secondary); [alphaXiv](https://www.alphaxiv.org/overview/2510.22319v2).
- **Pref-GRPO** (Aug 2025) names an "illusory advantage". A pointwise reward model gives near-identical scores to images in the same group, and dividing by the small group standard deviation inflates those tiny gaps into large advantages, which drives over-optimization. In the authors' example, the UnifiedReward score rose throughout training while images drifted toward an "unnaturally dark style" — [arXiv HTML](https://arxiv.org/html/2508.20751v1); [project page](https://codegoat24.github.io/UnifiedReward/Pref-GRPO).
- **SRPO / Direct-Align** (Tencent Hunyuan, Sep 2025): direct reward-optimization methods, Direct-Align included, "perform suboptimally on realism, even falling short of the baseline FLUX model due to reward hacking". The documented hack is colour oversaturation. Training restricted to late timesteps increased the rate of hacking — [EmergentMind](https://www.emergentmind.com/papers/2509.06942) (secondary); [The Moonlight review](https://www.themoonlight.io/review/directly-aligning-the-full-diffusion-trajectory-with-fine-grained-human-preference) (secondary).
- **DDRL** (Ye et al., with Ermon and Zou; NVIDIA internship work; Dec 2025): "most existing algorithms are often vulnerable to reward hacking, such as quality degradation, over-stylization, or reduced diversity". The authors identify regularizing against on-policy samples as "the primary cause of reward hacking in diffusion RL" — [arXiv HTML v2](https://arxiv.org/html/2512.04332v2); [EmergentMind topic](https://www.emergentmind.com/topics/data-regularized-diffusion-reinforcement-learning-ddrl).
- **DiverseGRPO** (Dec 2025): GRPO-trained image models "suffer from mode collapse (similar faces, camera angles, etc.)" — [arXiv PDF](https://arxiv.org/pdf/2512.21514v1).
- **PEC, "When Policy Entropy Constraint Fails"** (May 2026, NeurIPS 2026): RLHF on flow-matching T2I models "often leads to severe diversity collapse". Policy entropy stays flat, because the noise schedule is fixed, while perceptual diversity drops. This makes policy entropy a misleading monitor for flow models — [project page](https://xiaofeng-tan.github.io/projects/PEC/); [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/155843).
- **GARDO** (HKUST, Kuaishou, CUHK MMLab and Edinburgh, Dec 2025): "proxy scores increase while real image quality deteriorates and generation diversity collapses" — [arXiv HTML](https://arxiv.org/html/2512.24138v1).
- **DDPO** (May 2023): reported over-optimization of the aesthetic reward. In a case that is often cited, fine-tuning against the LLaVA-based prompt-alignment reward led the model to render loosely legible text into images, exploiting the VLM through typography — [arXiv 2305.13301](https://arxiv.org/abs/2305.13301) (K; the project page was blocked, so this was not re-verified).
- **Gao et al.** (Oct 2022, LLM RLHF; this is the canonical quantification). Let d be the square root of the KL from the initial policy.
  - Gold reward follows d(α−βd) for best-of-n and d(α−β·log d) for RL.
  - Larger reward models over-optimize less.
  - A KL penalty behaved like early stopping and did not improve the gold-reward-vs-KL frontier — [arXiv 2210.10760](https://arxiv.org/abs/2210.10760) (K).

**Video RL: static or frozen video, motion, identity**
- **DenseDPO** (Snap, NeurIPS 2025): vanilla video-DPO pairs are generated from independent noise, so they differ in motion. Annotators tend to pick the artifact-free slow-motion clip, and DPO then amplifies this "motion bias" toward static content.
  - A secondary summary gives VBench Dynamic Degree values of 55.4 and 48.2 for vanilla DPO against 56.3 for DenseDPO. Which configuration each number belongs to is unclear; check the paper.
  - The authors also note that some video reward models appear to favour static videos, and that VLMs still struggle with complex motion such as tennis or cartwheels.
  - Sources: [HF papers](https://huggingface.co/papers/2506.03517); [NeurIPS 2025 PDF](https://papers.neurips.cc/paper_files/paper/2025/file/fa9755043814e7f08d859a286bb83c35-Paper-Conference.pdf); [alphaXiv overview](https://www.alphaxiv.org/overview/2506.03517v2) (numbers secondary).
- **SHIFT** (Mar 2026): Dynamic Degree collapsed from 0.67 to 0.33 under SFT, "indicating near-static video generation". The authors note that "the risk of reward hacking is especially severe in video generation due to its high dimensionality and sparse supervision" — [arXiv HTML v2](https://arxiv.org/html/2603.17426v2); [arXiv PDF](https://arxiv.org/pdf/2603.17426).
- **OnlineVPO** (Dec 2024; WACV 2026): directly applying standard off-policy DPO to video "failed to yield significant performance improvements and even led to notable degradation in metrics such as dynamic degree" — [CVF PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Zhang_Align_Video_Diffusion_Model_with_Online_Video-Centric_Preference_Optimization_WACV_2026_paper.pdf).
- **Reward Forcing** (Dec 2025; streaming distillation, not RL in the strict sense): frames become over-dependent on static initial "sink" tokens, giving "copied initial frames and diminished motion dynamics" — [arXiv HTML](https://arxiv.org/html/2512.04678v1).
- **IPRO** (Shen et al., Oct 2025; CVPR 2026; tested on Wan 2.2 I2V and an in-house 15B model). An ArcFace identity reward scored against the reference frame produces "copy-paste": faces "over-adhering to the input image, resulting in stiff faces, lack of expression, and poor response to prompt". The ablation used Gemini 2.5 Pro as a VLM to judge hacking:

  | Configuration | FaceSim | Hacking rate |
  |---|---|---|
  | Base Wan 2.2 | 0.578 | 7% |
  | Without KL | 0.754 | 58% |
  | Without the multi-view scoring pool (FSM) | 0.739 | 52% |
  | Full IPRO | 0.694 | 10% |

  Sources: [CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Identity-Preserving_Image-to-Video_Generation_via_Reward-Guided_Optimization_CVPR_2026_paper.pdf); [papernotes](https://en.papernotes.org/CVPR2026/video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization/); [Moonlight](https://www.themoonlight.io/review/identity-preserving-image-to-video-generation-via-reward-guided-optimization) (secondary).
- **DDRL video comparison** on Cosmos2.5-2B (secondary): DanceGRPO reached a higher VideoAlign T2V reward (0.715) but a −10.5% change in human preference. DDRL reached 0.604 with a 0% change. The DDRL abstract cites more than 1M GPU-hours and about 10k double-blind human evaluations — [EmergentMind](https://www.emergentmind.com/papers/2512.04332) (secondary); [arXiv HTML](https://arxiv.org/html/2512.04332v1).
- **2026 papers that target static-video or latent hacking** (all listed):
  - "Scaling RL for Diffusion Models via Velocity Matching" (arXiv 2608.23664): a dynamic-tracking reward that "counters static-video reward hacking".
  - "Manifold Drift in Flow Preference Optimization: A Root Cause of Reward Hacking" (arXiv 2608.20011, ThermoDPO): preference updates push samples off the data manifold.
  - "CoRe: Co-Evolving Reward Models for Mitigating Latent Reward Hacking in Video Diffusion Models" (arXiv 2609.36245).
  - SoliReward (arXiv 2512.22170): a tie-aware video reward model that resists hacking and annotation noise.
  - "Inference-Time Scaling for Joint Audio-Video Generation" (arXiv 2606.03183): adaptive reward weighting "to curb single-verifier hacking".

  Source: [awesome-rl-for-video-generation](https://github.com/chrisliu298/awesome-rl-for-video-generation). A targeted search for CoRe did not find the paper outside the list.
- Identity rewards trade off against motion. MagicID (arXiv 2503.12689) "balances identity against motion", and PersonalVideo (arXiv 2411.17048) adds dynamics and semantic rewards to identity — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation) (listed).

**LLM/VLM-as-judge gaming**
- **One Token to Fool LLM-as-a-Judge** (Tencent AI Lab; Jul 2025; NeurIPS 2025). Generative reward models give false-positive rewards to "master keys": non-word symbols such as ":" or ".", and generic openers such as "Thought process:".
  - In one RLVR run the policy collapsed into emitting short reasoning openers and was still rewarded.
  - LLaMA3-70B-Instruct had a 73.0% false-positive rate on "Thought process:", against 0% for Master-RM-7B (secondary).
  - Inference-time strategies were not a reliable mitigation.
  - Sources: [arXiv HTML](https://arxiv.org/html/2507.08794v1); [NeurIPS 2025](https://neurips.cc/virtual/2025/131135); [EmergentMind](https://www.emergentmind.com/topics/one-token-to-fool-llm-as-a-judge) (FPR number secondary).
- **Null models** (Oct 2024; ICLR 2025): a constant response that ignores the input reached an 86.5% length-controlled win rate on AlpacaEval 2.0 — [arXiv 2410.07137](https://arxiv.org/abs/2410.07137) (K).
- **Typographic and artifact attacks on VLMs:**
  - They are "particularly harmful" to LLaVA, Qwen-VL, VILA and Imp — [arXiv 2405.14169](https://arxiv.org/abs/2405.14169).
  - Self-generated typographic attacks reduce LVLM classification performance "by up to 33%" — [arXiv 2402.00626](https://arxiv.org/html/2402.00626v2).
  - Web-artifact attacks found for one model keep "up to 90%" of their effectiveness on another — [arXiv 2503.13652](https://arxiv.org/html/2503.13652v1).
- **Judge biases:**
  - The MT-Bench study documented position, verbosity and self-enhancement biases in LLM judges — [arXiv 2306.05685](https://arxiv.org/abs/2306.05685) (K).
  - LLM evaluators recognise and favour their own generations, and self-preference correlates with self-recognition — [arXiv 2404.13076](https://arxiv.org/abs/2404.13076) (K).

### Inferences
- The most likely hacks in Prompt-to-Scene Creator:
  - Prompt strategies drift toward low-motion shot language (locked-off camera, slow push-ins), because both the judge and users reward artifact-free takes (DenseDPO, SHIFT).
  - The photorealism / anti-3D dimension rewards a glossy, oversaturated "clean CG" look (SRPO, Pref-GRPO). This directly conflicts with the live-action goal.
  - The identity-continuity dimension rewards frontal, expressionless faces (IPRO copy-paste).
  - The composition-match dimension rewards literal, centred framing.
  - Text in frames, or prompt text placed in the judge's context, steers the judge (typographic attacks, master keys).
- Choosing the best of N takes with the judge is itself optimization against the judge (best-of-n). Gao et al.'s best-of-n curve implies that gold quality can fall as N grows, even with no RL at all.
- The static-video bias shows up in human labels (DenseDPO) as well as in reward models and VLMs. More user ratings alone will not fix it; motion needs its own guardrail metric, measured independently of the judge.
- Flow-GRPO and GRPO-Guard disagree on how much hacking occurs. "Use the KL coefficient from the paper" is therefore not a safe default; validate on your own setup against gold or human metrics.

### Gaps
- No public, quantified reward-hacking analysis was found for the commercial models: Seedance 2.0, Veo 3.1, Kling 3.0 or Sora 2.
- No study was found that measures an upstream LLM prompt-writer learning to game a downstream VLM judge in video (Goodhart at the prompt-strategy level). The LLM-judge attack papers are the closest analogue.
- The DDPO typographic-exploit detail comes from prior knowledge; the project page was blocked.
- CoRe, Manifold Drift and Velocity Matching were seen only in the curated list.

---

## 2. Mitigations: KL, ensembles and uncertainty, per-reward normalization, pairwise rewards, reward-model refresh, early stopping, constrained RL, diversity, GRPO-Guard (which work best?)

### Takeaway
No single mitigation suffices. The strongest 2025–2026 evidence supports a combination:
- anchor to real data or a refreshed reference, not only an on-policy KL (DDRL, GARDO);
- fix the GRPO advantage and ratio estimator (GRPO-Guard's RatioNorm, Pref-GRPO's win-rate advantages, GDPO's per-reward normalization);
- apply uncertainty or ensemble penalties selectively;
- add explicit diversity and motion terms;
- design rewards for the task (a multi-reference identity pool, negative control prompts);
- co-evolve or refresh the reward model.

All of these must be gated by held-out human evaluation, because proxy curves can rise while quality falls. Evidence is mostly the authors' own; head-to-head comparisons are rare.

### Cited Findings

**KL / reference / data regularization**
- **Flow-GRPO** argues that KL is not just early stopping: the same reward is reachable with KL, at the cost of longer training — [arXiv HTML v4](https://arxiv.org/html/2505.05470v4) (snippet). In LLM RLHF, by contrast, Gao et al. found that the KL penalty behaves like early stopping — [arXiv 2210.10760](https://arxiv.org/abs/2210.10760) (K).
- **IPRO on Wan 2.2:** removing KL raised the hacking rate from 10% to 58% — [CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Identity-Preserving_Image-to-Video_Generation_via_Reward-Guided_Optimization_CVPR_2026_paper.pdf).
- **DPOK** (May 2023): KL-regularized online RL for T2I outperformed supervised reward-weighted fine-tuning on both alignment and image quality — [arXiv 2305.16381](https://arxiv.org/abs/2305.16381) (K).
- **DDRL:**
  - Method: anchors the policy to off-policy data (real or pretraining) with a forward-KL term, combining reward maximization with diffusion-loss minimization.
  - Results: consistently improves rewards with the highest human preference, across VideoAlign, VBench and OCR rewards, in both T2V and I2V.
  - Deployment: used in Cosmos-Predict2.5.
  - Sources: [arXiv HTML](https://arxiv.org/html/2512.04332v1); [Moonlight](https://www.themoonlight.io/review/data-regularized-reinforcement-learning-for-diffusion-models-at-scale) (secondary).
- **GARDO:**
  - Gated KL: the penalty applies only to a small subset of samples (about 10% per batch) with the highest reward uncertainty. One summary says uncertainty is measured by disagreement within a reward-model ensemble (secondary).
  - Adaptive reference: the reference model is periodically updated to match the online policy.
  - Diversity-aware optimization: rewards are amplified for high-quality samples that are also diverse.
  - Reported outcome: hacking mitigated without sacrificing sample efficiency or exploration.
  - Sources: [arXiv HTML](https://arxiv.org/html/2512.24138v1); [Moonlight](https://www.themoonlight.io/review/gardo-reinforcing-diffusion-models-without-reward-hacking) (secondary).
- **DiverseGRPO:** replaces the uniform KL with a Wasserstein constraint, and schedules regularization by denoising stage because early steps are critical for diversity — [arXiv PDF](https://arxiv.org/pdf/2512.21514v1).
- **Other regularizers** (listed) — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation):
  - Reg-DPO (arXiv 2511.01450): adds a weighted SFT term to limit distribution shift in video DPO.
  - ROCM (arXiv 2503.06171): f-divergence regularization.
  - Flow-DPPO (arXiv 2606.11025): a Gaussian-KL constraint in place of ratio clipping.
  - "Aligning Human Sense" (arXiv 2608.21425): distributional alignment in place of KL in GRPO.

**Estimator fixes: ratio normalization, pairwise, per-reward normalization**
- **GRPO-Guard:**
  - RatioNorm standardizes each step's importance ratios (mean near 1, consistent variance), which restores clipping.
  - Gradient reweighting balances gradient size across timesteps (one summary gives the factor as 1/dt).
  - Results: average GenEval gold score 0.89 against 0.84 for Flow-GRPO, on SD3.5-M and FLUX.1-dev, with both Flow-GRPO and DanceGRPO variants, "without relying on heavy KL regularization".
  - Trade-off: full variance alignment suppresses hacking more strongly but clips more high-advantage samples.
  - One summary said the code had not been released.
  - Sources: [alphaXiv](https://www.alphaxiv.org/overview/2510.22319v2); [papernotes](https://en.papernotes.org/CVPR2026/image_generation/grpo-guard_mitigating_implicit_over-optimization_in_flow_matching_via_regulated_/) (secondary).
- **Pref-GRPO:** a pairwise preference reward model compares images within each group, and each image's win rate becomes its reward. This gives larger, more stable advantage differences and "substantially alleviates reward hacking". The paper also releases the UniGenBench benchmark. No independent replication was found — [arXiv HTML](https://arxiv.org/html/2508.20751v1); [project page](https://codegoat24.github.io/UnifiedReward/Pref-GRPO).
- **GDPO** (NVIDIA, Jan 2026, ICML 2026):
  - Problem: summing several rewards and then group-normalizing can collapse distinct reward combinations into identical advantages.
  - Fix: normalize each reward within its group, then apply batch-wise normalization to the summed advantage.
  - Results: up to +2.7% accuracy and more than +4% format compliance over GRPO (secondary).
  - Implementations: HF-TRL, verl and NeMo-RL.
  - The evidence is from LLM tasks only.
  - Sources: [HF papers](https://huggingface.co/papers/2601.05242); [NVIDIA Research](https://research.nvidia.com/labs/twn/publication/icml_2026_gdpo/); [Moonlight](https://www.themoonlight.io/review/gdpo-group-reward-decoupled-normalization-policy-optimization-for-multi-reward-rl-optimization) (secondary).
- **Multi-reward calibration for diffusion** (listed) — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation):
  - Calibrated Multi-Preference Optimization (arXiv 2502.02588): calibrates heterogeneous reward scales.
  - Bayesian Prior-Guided Optimization (arXiv 2511.18919): down-weights uncertain reward dimensions.
  - MapReduce LoRA (arXiv 2511.20629): merges preference-specialised adapters to push the Pareto front.
  - Multi-Objective Preference Optimization (arXiv 2505.10892): constrained optimization when objectives conflict.

**Reward models: ensembles, scale, refresh and co-evolution**
- **Ensembles (LLM RLHF):**
  - Coste et al. (ICLR 2024): worst-case and uncertainty-weighted optimization over a reward-model ensemble countered over-optimization, with up to about 70% improvement under best-of-n. Combined with a small KL penalty, it prevented over-optimization in PPO "at no performance cost" — [arXiv 2310.02743](https://arxiv.org/abs/2310.02743) (K).
  - Eisenstein et al.: ensembles "mitigate but do not eliminate" hacking, because members share error patterns. Ensembles that differ in pretraining seed beat ensembles that differ only in fine-tuning seed — [arXiv 2312.09244](https://arxiv.org/abs/2312.09244) (K).
  - WARM (weight-averaged reward models): a policy trained with WARM had a 79.4% win rate against one trained with a single reward model — [arXiv 2401.12187](https://arxiv.org/abs/2401.12187) (K).
- **RewardDance** (ByteDance Seed, Sep 2025): a generative reward model (the probability of a "yes" token) scaled to 26B parameters on InternVL. Seedream-3.0 image-text alignment rose from 74.1 to 84.8. Summaries claim robust resistance to hacking at scale, but no direct measurement was seen — [EmergentMind](https://api.emergentmind.com/papers/2509.08826); [Moonlight](https://www.themoonlight.io/review/rewarddance-reward-scaling-in-visual-generation) (secondary).
- **Co-evolving or refreshed reward models:**
  - SHIFT alternates generator updates with reward-model refinement (adversarial hybrid fine-tuning) and adds pixel-motion rewards from optical flow and point tracks — [arXiv HTML v2](https://arxiv.org/html/2603.17426v2).
  - Listed in the [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation) only: OnlineVPO refreshes preference pairs online; Dual-IPO (arXiv 2502.02088) alternates data improvement with generator optimization; CoRe refits a latent reward head on current samples, anchored to real videos.
- **Frozen VLM critic:** Diffusion-DRF avoids training a reward model at all; its authors argue the signal is therefore harder to exploit — [arXiv PDF](https://arxiv.org/pdf/2601.04153) (snippet).
- **Uncertainty-aware rewards:** SURE (arXiv 2608.06125) uses a latent reward model that gives a Gaussian estimate for each noisy latent, so the generator knows which feedback is reliable — [chatpaper listing](https://chatpaper.com/paper/318617) (secondary).
- **Hardening the judge itself:** Master-RM trains on truncated-reasoning negatives, which drove the false-positive rate on master keys to about 0 — [arXiv HTML](https://arxiv.org/html/2507.08794v1).

**Task-specific reward shaping (realism, identity, motion, diversity)**
- **Realism / anti-hacking (SRPO):** "semantic relative preference" uses the reward difference between prompts with positive and negative control words, which cancels reward bias. Negative rewards regularize against oversaturation. Human evaluation reported realism and aesthetics on FLUX.1-dev improved by more than 3× (secondary) — [EmergentMind](https://www.emergentmind.com/papers/2509.06942) (secondary).
- **Identity (IPRO):** a multi-view facial scoring pool built from all frames of the ground-truth video avoids the copy-paste trap — [papernotes](https://en.papernotes.org/CVPR2026/video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization/).
- **Motion:**
  - DenseDPO builds pairs by denoising corrupted copies of a real video, so both clips share motion, and collects a preference per segment of about 1 s. It improved motion with about one-third of the labels, and GPT-class VLM segment labels came close to human labels — [HF papers](https://huggingface.co/papers/2506.03517).
  - Listed only: LongCat-Video (arXiv 2510.22200) uses multi-reward GRPO with separate motion and text-alignment critics; Seedance 1.5 pro (arXiv 2512.13507) uses three RLHF reward models (audio-video alignment, motion, aesthetics) — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation).
- **Diversity:**
  - PEC regularizes perceptual entropy, measured in the reward model's representation space, and reports the best overall score (0.702) with diversity of 0.773 (project-page numbers) — [project page](https://xiaofeng-tan.github.io/projects/PEC/).
  - DPP-guided policy optimization for diverse video (arXiv 2511.20647) is listed only — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation).
- **Constrained RL:**
  - Moskovitz et al. (ICLR 2024) treat a composite reward model per component and use Lagrange multipliers to keep each component at or below its "proxy point", the value beyond which over-optimization begins — [arXiv 2310.04373](https://arxiv.org/abs/2310.04373) (K).
  - Safe RLHF separates a helpfulness reward from a harmlessness cost and uses Lagrangian optimization — [arXiv 2310.12773](https://arxiv.org/abs/2310.12773) (K).
- **Older diffusion-specific method:** TDPO-R (ICML 2024) uses per-timestep rewards (temporal inductive bias) and periodically resets critic neurons (primacy bias) to counter over-optimization — [arXiv 2402.08552](https://arxiv.org/abs/2402.08552) (K).

**Evidence on what works best**
- DDRL is the only method found with large-scale double-blind human evaluation (about 10k judgments). It shows that the method with the best proxy reward (DanceGRPO) loses human preference (−10.5%, secondary), while DDRL holds even — [EmergentMind](https://www.emergentmind.com/papers/2512.04332) (secondary).
- GRPO-Guard shows that an estimator fix beats Flow-GRPO on gold metrics (0.89 against 0.84) — [alphaXiv](https://www.alphaxiv.org/overview/2510.22319v2).
- IPRO's ablation shows that the KL term and the reward design each matter on their own; removing either one pushes the hacking rate above 50% — [CVPR 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Identity-Preserving_Image-to-Video_Generation_via_Reward-Guided_Optimization_CVPR_2026_paper.pdf).

### Inferences
- **Model-level RL (Flow-GRPO + MixGRPO on Wan 2.2), a suggested stack:**
  1. **GRPO-Guard-style RatioNorm and per-timestep gradient balancing.** MixGRPO's sliding ODE/SDE window ([listed](https://github.com/chrisliu298/awesome-rl-for-video-generation)) changes which timesteps receive gradients. Log ratio mean and variance per window.
  2. **A DDRL-style anchor on curated real live-action footage.** This also pulls the model away from the 3D/CG look.
  3. **GDPO-style per-dimension normalization of the 7 judge scores** before they are weighted.
  4. **Within-group win rates from a pairwise judge (Pref-GRPO)** instead of absolute VLM scores, which tend to cluster.
  5. **GARDO-style gated KL** on samples where a judge ensemble disagrees.
  6. **Explicit motion (optical-flow) and diversity (perceptual-entropy) terms or constraints.**
  7. **A multi-view identity pool** for the identity dimension.
  8. **Early stopping on held-out human or gold evaluation**, not on proxy reward.
- **Judge weights:** online changes to the judge weights change the reward function. Version each change, and re-run the held-out gold evaluation before RL or the bandit consumes new weights. Otherwise the policy chases a moving target that it is itself shaping.
- **Constraints over weighted sums:** treating motion, safety and "no CG look" as constraints (Lagrangian, or hard thresholds) is more robust than folding them into a weighted sum, where a large gain on one dimension can buy a regression on another.

### Gaps
- No public head-to-head video benchmark compares KL, ensembles, DDRL, GRPO-Guard and Pref-GRPO.
- GDPO is validated only on LLM tasks.
- No standard KL β ranges for video flow models were found.
- RewardDance's hacking-resistance claim is not backed by measurements seen in this session.

---

## 3. Evaluation and promotion protocols (held-out suites, blind A/B, arenas with CIs, guardrails, canary/shadow, statistics; how labs evaluate)

### Takeaway
Public disclosure of pre-ship evaluation is thin:
- **Google (Veo 3 card):** describes side-by-side human-rater comparisons on Meta's MovieGenBench (1,003 video and 527 video+audio prompts) and on VBench-I2V (355 image+text pairs). It also describes four layers of safety evaluation: development evaluations, independent assurance evaluations, internal and external red teaming, and ethics and safety reviews.
- **OpenAI (Sora 2 card):** emphasises the safety stack and provenance.
- **ByteDance and Kuaishou:** publish technical reports covering multi-reward RLHF and DPO on human pairs, but almost nothing on how they gate releases.

For a product, the defensible protocol is:
- a frozen, stratified held-out suite;
- blind pairwise human A/B, analysed with Bradley–Terry or Elo with bootstrap confidence intervals;
- prediction-powered inference (PPI) to combine cheap judge labels with a few human labels;
- guardrail metrics with non-inferiority margins;
- staged rollout under always-valid sequential tests.

### Cited Findings
- **Veo 3 model card** (Jan 2026 version):
  - **Benchmarks:**
    - MovieGenBench: "1,003 prompts for video and 527 prompts for video+audio".
    - Competitors on the benchmark's provided videos: Meta MovieGen, Kling 2.0, Minimax and Sora Turbo.
    - Freshly sampled models: Kling 2.0, Sora, Runway Gen-3, WAN 2.1 and MiniMax.
    - VBench-I2V (355 pairs), against Runway Gen-4, Kling 2.0 Pro, WAN 2.1 and MiniMax I2V-01.
  - **Result:** "Veo 3 performed best on overall preference, and for its capability to follow prompts accurately".
  - **Safety evaluation:**
    - "Assurance evaluations ... developed and run by a group outside of the model development team".
    - Red teaming by "specialist internal and external teams".
    - "Ethics & Safety Reviews" ahead of release.
    - System-level evaluations "within the context of specific applications".
  - **Findings:** Veo 3 "appears to skew towards lighter skin tones when race is not specified in the prompt". Its deepfakes "were of worse quality than those made by dedicated deepfake tools and can be mitigated in part through the use of SynthID watermarking".
  - **Mitigations and maintenance:** SynthID watermarking plus "production filtering". The card is "updated from time to time ... to include updated evaluations as the model is improved".
  - Source: [Veo 3 Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Veo-3-Model-Card.pdf).
- The Veo 3.1 Lite card (8 Apr 2026) builds on Veo 3 and points back to the Veo 3 card's evaluation approach — [Veo 3.1 Lite model card](https://deepmind.google/models/model-cards/veo-3-1-lite/) (snippet).
- **Sora 2 system card** (30 Sep 2025):
  - **Provenance:** C2PA metadata on all assets, a visible moving watermark on downloads from sora.com and the app, and internal detection tools.
  - **Likeness:** opt-in consent and controls for likeness through the cameo feature.
  - Source: [Sora 2 system card](https://deploymentsafety.openai.com/sora-2) (via search snippet; the host is blocked here, so the card's quantitative safety evaluations were not read).
  - A March 2026 OpenAI safety post contained no external evaluation data — [dig.watch](https://dig.watch/?p=248303).
- **Meta Movie Gen** (Oct 2024) introduced the Movie Gen Video Bench prompt set (1,003 prompts; the same count Veo 3 used). Models are evaluated by human A/B, reported as net win rate (wins minus losses) — [arXiv 2410.13720](https://arxiv.org/abs/2410.13720) (K).
- **ByteDance, Kuaishou and Tencent post-training disclosures** (listed) — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation):
  - Seedance 1.0 (arXiv 2506.09113): multidimensional rewards.
  - Seedance 1.5 pro (arXiv 2512.13507): three RLHF reward models.
  - Seaweed-7B (arXiv 2504.08685): video DPO from annotators' best and worst picks.
  - Kling-Omni (arXiv 2512.16776): multi-round DPO on human pairs.
  - HunyuanVideo 1.5 (arXiv 2511.18870): a multidimensional reward model with online RL and DPO.
- **Kling 3.0** launched on 4–5 Feb 2026 with unified generation, editing and audio, and up to 6 shots in a 15-second clip. No public human-evaluation protocol was found — [dev.to guide](https://dev.to/jsam/kling-30-a-practical-guide-to-kuaishous-latest-ai-video-generator-2026-58a4) (secondary); [vo3ai blog](https://www.vo3ai.com/blog/kling-30-unifies-generation-editing-and-audio-in-one-architecture-why-rival-ai-v-2026-04-09) (secondary); [Tiger news](https://www-web.itiger.com/news/1103502699) (secondary).
- **Arenas:**
  - Artificial Analysis Video Arena: Elo from blind votes updated in real time. One model's interval was ±7 with more than 13,500 samples — [cutout.pro blog](https://www.cutout.pro/learn/blog-happyhorse-ai-video-leaderboard/) (secondary; the official methodology was not retrieved).
  - LMArena open-sourced its leaderboard methodology as "Arena-Rank" — [LMArena blog](https://news.lmarena.ai/arena-rank).
  - Chatbot Arena uses a Bradley–Terry model with bootstrap confidence intervals, active sampling and anomalous-user detection — [arXiv 2403.04132](https://arxiv.org/abs/2403.04132) (K).
  - TTS Arena requires 100 votes before a model is listed and ranks by the lower bound of the interval — [TTS Arena docs](https://tts-agi-docs.hf.space/ranking).
  - Design Arena uses an approximate 95% Wilson score interval — [Design Arena methodology](https://notes.designarena.ai/methodology/).
  - Evalica is an open toolkit for reproducible leaderboards — [arXiv 2412.11314](https://arxiv.org/pdf/2412.11314).
- **Prediction-powered inference (PPI):**
  - It combines a small human-labelled set, which measures and corrects the judge's systematic error, with a large judge-labelled set. The result is unbiased, and each additional judge-labelled item shrinks the variance — [HF papers 2606.05308](https://huggingface.co/papers/2606.05308.md); [arXiv HTML](https://arxiv.org/html/2606.05308).
  - Stratified PPI (NeurIPS 2024) gives "substantially tighter confidence intervals" — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c9fcd02e6445c7dfbad6986abee53d0d-Abstract.html).
  - Original PPI paper: [arXiv 2301.09633](https://arxiv.org/abs/2301.09633) (K).
  - The 2026 ranking-PPI result rests on about 100 human labels from one industry group, so treat it as preliminary — [arXiv HTML](https://arxiv.org/html/2606.05308).
- **Sequential testing:** always-valid p-values allow A/B tests to be monitored continuously without inflating false positives — [arXiv 1512.04922](https://arxiv.org/abs/1512.04922) (K).
- **LLM-judge agreement:** in the MT-Bench study, GPT-4 judgments agreed with human preferences more than 80% of the time, about the same as humans agree with each other. The same study documented position and verbosity biases — [arXiv 2306.05685](https://arxiv.org/abs/2306.05685) (K).
- **Newer benchmarks** (listed) — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation):
  - EvalVerse (arXiv 2605.23271): expert-calibrated cinematic scoring.
  - RAVEN-Eval (arXiv 2608.09111): rubric-based pairwise preferences.
  - FIRM-Video (arXiv 2608.21839): checklist verification before scoring.
  - UnifiedReward-Flex (arXiv 2602.02380): per-prompt rubrics.

### Inferences
- **Suggested promotion gate**, applied to a new prompt strategy, new judge weights or a Wan/DiT checkpoint:
  1. **Frozen held-out suite.** Never shown to the reflection LLM or the bandit. Stratify it by shot type (dialogue close-up, action and high motion, crowd, low light, multi-character identity, photographic-surreal) and by target model.
  2. **Blind pairwise human A/B against the incumbent.** Fit Bradley–Terry with bootstrap intervals (at least 1,000 resamples). Promote only if the lower bound of the net win rate is above 0.
  3. **PPI estimate:** combine judge labels on the whole suite with roughly 100–300 human-labelled pairs per decision, for an unbiased quality estimate. (The label count is a heuristic, not sourced.)
  4. **Guardrails with non-inferiority margins:**
     - motion magnitude (optical flow or dynamic degree);
     - perceptual diversity (not policy entropy; see PEC);
     - a CG/3D-look classifier rate;
     - the distribution of identity similarity, so that a variance collapse flags copy-paste;
     - safety-filter and provider-refusal rates;
     - judge–human agreement.
  5. **Staged rollout:** shadow, then canary, then ramp, under always-valid sequential tests. With many candidate strategies, correct for multiple comparisons, for example by controlling the false discovery rate.
- Veo's "assurance evals by a group outside the model development team" translates to one rule: the agent that writes and reflects on strategies must not see, write or tune the evaluation suite or the judge rubric.

### Gaps
- Public material does not show how OpenAI, ByteDance, Kuaishou, Alibaba (Wan 2.2/2.5) or Runway gate model updates: no canary percentages, statistical thresholds or regression suites.
- The Sora 2 card's evaluation figures could not be read.
- The official Artificial Analysis methodology was not retrieved.
- Alibaba and Runway were not researched, because the search budget ran out.

---

## 4. Continual / online learning from user feedback (forgetting, distribution shift, feedback loops, delayed feedback, drifting taste) and safe exploration for production bandits

### Takeaway
When users curate the system's own outputs and the curated outputs are fed back into training, the loop acts as implicit preference optimization. It amplifies the curators' initial biases and can lock onto one of several single-mode outcomes ("singleton equilibria") unless real reference data is mixed in at a sufficient rate.

Model-level RL on logged production preferences is also off-policy: those takes come from other models (Seedance, Veo, Kling, Sora). Off-policy DPO is documented to degrade video dynamics.

Production bandits should use:
- informative but shrunk priors;
- conservative constraints against a baseline;
- discounting for drifting taste;
- batched updates that tolerate delays;
- logged sampling probabilities (propensities), so that rewritten strategies can be evaluated offline before promotion.

### Cited Findings
- **Ferbach et al.** (NeurIPS 2024): in self-consuming retraining loops, user curation acts as implicit preference optimization. Experiments on synthetic data and CIFAR-10 show that the procedure "amplifies biases of the reward model" — [arXiv HTML](https://arxiv.org/html/2407.09499v1); [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/94368).
- **Co-evolving human preferences** (NeurIPS 2026 poster): users' tastes shift as they see model outputs, and iterative curation "amplifies initial biases and drives the system toward one of multiple singleton equilibria". Injecting reference data at a large enough rate gives a single, globally attracting equilibrium. The paper proposes an algorithm for choosing that reference data — [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/148960).
- **Multi-model extension:** when several models interact, they can dampen or even invert the alignment benefit of curation. The paper also studies noisy and adversarial curation — [catalyzex listing](https://www.catalyzex.com/author/Xiukun%20Wei) (secondary).
- **Model collapse:**
  - Training indiscriminately on model-generated data causes irreversible defects in which the tails of the original distribution disappear — [Shumailov et al., Nature 2024](https://www.nature.com/articles/s41586-024-07566-y) (K).
  - Without enough fresh real data in each generation, quality or diversity progressively declines ("Self-consuming generative models go MAD", ICLR 2024) — [arXiv 2307.01850](https://arxiv.org/abs/2307.01850) (K).
- **Degenerate feedback loops in recommender systems** (AIES 2019) — [arXiv 1902.10730](https://arxiv.org/abs/1902.10730) (K).
- **Off-policy degradation:** OnlineVPO attributes the dynamic-degree drop under DPO to its being off-policy, and uses online, video-centric preference pairs instead — [CVF PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Zhang_Align_Video_Diffusion_Model_with_Online_Video-Centric_Preference_Optimization_WACV_2026_paper.pdf).
- **Forgetting and regularization for continual updates:**
  - DDRL's forward KL to real data — [arXiv HTML](https://arxiv.org/html/2512.04332v1).
  - GARDO's periodically refreshed reference model — [arXiv HTML](https://arxiv.org/html/2512.24138v1).
  - "RL's Razor" (Sep 2025): online RL fine-tuning forgets less than SFT. The amount of forgetting is predicted by the KL divergence from the base policy, measured on the new task — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259) (K; the arXiv ID was not re-verified).
- **Conservative exploration:**
  - Stage-wise conservative linear Thompson sampling (SCLTS, NeurIPS 2020) maximizes reward while satisfying a per-round lower bound relative to a baseline. It plays the baseline only O(log T) times, against O(√T) for prior methods. The motivating example is a recommender that wants to "avoid recommendations that are extremely disliked by the users at any single round" — [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/804741413d7fe0e515b19a7ffc7b3027-Abstract.html); [arXiv 2010.00081](https://arxiv.org/abs/2010.00081v1).
  - "Improved Algorithms for Conservative Exploration in Bandits" keeps performance "almost never worse than the performance of the baseline". It also considers checkpoint constraints, checked only at predefined points — [arXiv 2002.03221](https://arxiv.org/abs/2002.03221).
- **Delays and non-stationarity:**
  - Chapelle & Li (NeurIPS 2011): in display-advertising simulations, Thompson sampling held up well under delayed, batched feedback — [NeurIPS 2011](https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) (K).
  - Russo et al.'s Thompson-sampling tutorial covers specifying priors, caution and constraints, non-stationary systems (discounting) and concurrent updates — [arXiv 1707.02038](https://arxiv.org/abs/1707.02038) (K).
- **Analogues of LLM-driven prompt optimization in video** (listed) — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation):
  - VPO (arXiv 2503.20491): DPO on a prompt rewriter, for quality and safety.
  - VISTA (arXiv 2510.15831): a test-time agent that runs pairwise tournaments with specialist critiques.
  - "Retrieval, Refinement, and Ranking for T2V" (arXiv 2603.01509): black-box prompt optimization with reranking.
  - PhyPrompt (arXiv 2603.03505): GRPO prompt refinement for physical plausibility.

### Inferences
- **Bandit layer.** Reflection keeps creating new arms, so the bandit is non-stationary by construction. Practical policies (all parameter values are unsourced heuristics):
  - Give each rewritten strategy its parent's posterior, shrunk toward the prior (for example, scale the parent's Beta counts by 0.2–0.5), not a uniform prior.
  - Cap the number of live arms.
  - Discount old evidence (for example, a decay of 0.98–0.995 per day) to follow drifting taste.
  - Keep the incumbent strategy as a conservative baseline: a traffic floor plus a stop-loss on rating drops, in the spirit of SCLTS.
  - Log each arm's sampling probability, which can be estimated by Monte Carlo, so that inverse-propensity or doubly-robust estimators can check strategies offline before promotion.
  - Run reflection on fixed windows with a minimum sample size, not after every rating, so it does not chase noise. Thompson sampling copes with delayed, batched ratings (Chapelle & Li).
- **Self-reinforcing loops.** Users rate only the takes they are shown, so logged feedback is shaped by the current policy. Keep some exploration traffic and a fixed set of real live-action reference footage in every model-level training round (co-evolving preferences, MAD). Do not train Wan 2.2 or the in-house DiT only on in-app winners.
- **Off-policy logs.** Preferences logged from Seedance, Veo and Kling takes are cross-model and off-policy for Wan 2.2. Use them mainly to train and calibrate a reward model. Then run on-policy Flow-GRPO on Wan samples with a real-data anchor (DDRL), not direct DPO on the logged pairs (OnlineVPO).
- **Forgetting regression tests.** Before each checkpoint is promoted, re-run the frozen held-out suite, plus a "capabilities retained" slice: motion-heavy shots, multi-character identity and rare styles. Track KL from the base model as an early-warning metric (RL's Razor).

### Gaps
- No published production case study was found of Thompson sampling over LLM-generated prompt strategies with reflection-based rewriting.
- No measurements were found of how fast users' aesthetic taste for AI video drifts.
- The delayed-feedback and non-stationary bandit literature after 2024 was not searched, because the budget ran out.
- No study was found on catastrophic forgetting across sequential reward fine-tuning of video diffusion models.

---

## 5. Adversarial and abusive feedback (poisoned preferences, spam ratings, prompt injection into judges and reflection; defenses)

### Takeaway
Small fractions of poisoned preference data can steer models: 1–5% in LLM RLHF, and clean-label feature-collision attacks on T2I reward models. LLM judges can be hijacked by optimized injected sequences, master-key tokens, instruction overrides or typography. Simple defenses (perplexity filters, known-answer detection, inference-time prompting) are documented as insufficient. Defenses therefore need to be architectural:
- isolate untrusted text from instruction-bearing contexts;
- cap per-user influence and track rater reputation;
- use gold check questions;
- use committees of diverse judges;
- control which data is admitted to RL training.

### Cited Findings
- **Best-of-Venom** (Apr 2024): injecting 1–5% poisoned preference pairs, each semantically and lexically plausible, steers the model to produce a target entity with a target sentiment — [arXiv HTML](https://arxiv.org/html/2404.05530v2).
- **Universal jailbreak backdoors** (Rando & Tramèr, ICLR 2024): malicious RLHF annotators can plant a universal "sudo"-style trigger that unlocks harmful responses. The authors' later competition models used a high poisoning rate (25%) to make backdoors strong — [arXiv 2311.14455](https://arxiv.org/abs/2311.14455); [competition report](https://arxiv.org/html/2404.14461v1); [code](https://github.com/ethz-spylab/rlhf-poisoning).
- **BadReward** (Jun 2025): a stealthy clean-label attack that poisons T2I RLHF reward models through "feature collisions between visually contradicted preference data instances". It does not require control of the annotation process — [arXiv HTML](https://arxiv.org/html/2506.03234v1).
- **DPO poisoning** (AAAI): a gradient-free method based on DPO scores poisons with a fraction of the data that random poisoning needs — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34968/37123).
- **Backdoors in diffusion models:**
  - BadT2I: a diffusion model "can be easily backdoored within a few fine-tuning steps" — [arXiv 2305.04175](https://arxiv.org/abs/2305.04175v2).
  - ControlNet backdoor: poisoning 1% of the fine-tuning corpus gives a 90–98% attack success rate (secondary) — [arXiv 2507.04726](https://arxiv.org/abs/2507.04726v2).
  - Nightshade: prompt-specific poisoning corrupts a Stable Diffusion prompt with fewer than 100 poisoned samples — [arXiv 2310.13828](https://arxiv.org/abs/2310.13828) (K).
- **Judge injection:**
  - **JudgeDeceiver** (Mar 2024): an optimized sequence injected into the attacker's candidate response makes the judge pick it. Known-answer detection, perplexity detection and windowed-perplexity detection were insufficient — [arXiv 2403.17710](https://arxiv.org/pdf/2403.17710).
  - **Maloyan & Namiot** (Apr 2025) describe two attacks: the Comparative Undermining Attack, which targets the verdict, and the Justification Manipulation Attack, which targets the judge's reasoning. They suggest committees of models with diverse architectures as a defense — [ar5iv 2504.18333](https://ar5iv.labs.arxiv.org/html/2504.18333).
  - **"Blind" attacks** overwrite the evaluation instructions so the judge always returns favourable scores; the proposed defense is counterfactual evaluation — [arXiv 2507.23453](https://arxiv.org/pdf/2507.23453).
  - **Security in LLM-as-a-Judge: a SoK** (Mar 2026, arXiv 2603.29403) names detecting prompt injection and evaluation manipulation as an open problem. It was seen only in a listing — [EmergentMind open problem](https://api.emergentmind.com/open-problems/detect-prompt-injection-and-evaluation-manipulation-laaj) (secondary).
  - **Recommended measures** include goal prioritization, attention tracking, adversarial training, input filtering and independent verification. These are recommendations from a vulnerability database, not tested results — [Promptfoo LM security DB](https://www.promptfoo.dev/lm-security-db/vuln/undefined-886657fa).
- **Master keys and typography:** see Q1 — [arXiv 2507.08794](https://arxiv.org/html/2507.08794v1); [arXiv 2503.13652](https://arxiv.org/html/2503.13652v1).
- **Architecture-level defenses:**
  - Indirect prompt injection through retrieved or third-party content is the canonical threat — [Greshake et al., arXiv 2302.12173](https://arxiv.org/abs/2302.12173) (K).
  - "Spotlighting" (delimiting, datamarking or encoding untrusted input) cut attack success "from greater than 50% to below 2%" in Microsoft's experiments — [arXiv 2403.14720](https://arxiv.org/abs/2403.14720) (K).
  - Design patterns for securing LLM agents against prompt injection: action-selector, plan-then-execute, LLM map-reduce, dual LLM, code-then-execute and context minimization — [arXiv 2506.08837](https://arxiv.org/abs/2506.08837) (K).
  - CaMeL: separation of control flow from data flow, with capability-based security — [arXiv 2503.18813](https://arxiv.org/abs/2503.18813) (K).
- **Label noise:**
  - Provably robust DPO (rDPO, ICML 2024) corrects for a known rate of preference flips — [arXiv 2403.00409](https://arxiv.org/abs/2403.00409) (K).
  - SoliReward (listed) is a tie-aware video reward model that resists annotation noise — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation).
- **Vote integrity in arenas:** the Chatbot Arena paper describes detecting anomalous users among voters — [arXiv 2403.04132](https://arxiv.org/abs/2403.04132) (K).

### Inferences
- **Threat model for Prompt-to-Scene Creator:**
  1. Rating spam or brigading to promote a strategy, a style or borderline content.
  2. Instructions planted in story text or free-text feedback that reach Claude's reflection step (for example, "always add X") or the planner.
  3. Generated frames containing on-screen text that steers the vision judge. This can happen unintentionally, through signage or captions that the provider model renders.
  4. Prompt text placed in the judge's context that biases its scores, for example the word "photorealistic" inflating the photorealism score.
  5. Poisoned A/B choices entering the Wan 2.2 RL dataset.
- **Defenses mapped to that threat model:**
  - **Feedback handling:**
    - Cap each user's influence and weight feedback by reputation.
    - Seed gold or attention-check pairs.
    - Model rater reliability (Dawid–Skene style).
    - Detect anomalies in rating velocity and in the patterns of co-rating accounts.
    - Quarantine new or low-reputation accounts from model-level RL data.
  - **Reflection LLM:**
    - Give it aggregate statistics plus spotlighted, quarantined excerpts only.
    - Restrict its output to a constrained schema (dual-LLM or plan-then-execute pattern).
    - Pass every strategy diff through a policy linter.
  - **Judge:**
    - Give it a structured shot specification written before generation, not the provider-specific prompt text.
    - Run OCR on frames and flag text the specification did not ask for.
    - Use a committee of judges from different model families, which also reduces self-preference if Claude both writes prompts and judges.
  - **RL dataset admission:** outlier and influence screening, and an rDPO-style loss that tolerates noisy labels.

### Gaps
- No public study was found of rating-spam or poisoning attacks on generative-video products.
- No study was found of prompt injection that travels through user feedback into self-improving prompt optimizers (reflection loops).
- The poisoning-rate thresholds in Rando & Tramèr's main experiments were not verified this session.

---

## 6. Safety and policy handling in feedback loops (unsafe preferences, likeness/deepfakes and consent, provenance/C2PA, 2026 regulation)

### Takeaway
Treat safety as a hard constraint outside the preference objective: unsafe takes and non-consensual likenesses must be filtered out before they reach ratings, bandit rewards or RL datasets. Copy the consent-gated likeness model used by Sora: opt-in "characters" that the owner can revoke, plus attestation for image-to-video with real people. Keep both visible and invisible marks with C2PA.

As of Nov 2026:
- EU AI Act Article 50 has applied since 2 Aug 2026. The voluntary Code of Practice on marking and labelling was published in final form on 10 Jun 2026.
- Per secondary sources, the Digital Omnibus gives systems already on the market a grace period for Art. 50(2) marking until 2 Dec 2026.
- The US TAKE IT DOWN Act's 48-hour removal duty has been enforced since 19 May 2026.
- The NO FAKES Act is still pending.

### Cited Findings
- **Preference loops amplify curators' biases** (Ferbach et al.) — [arXiv HTML](https://arxiv.org/html/2407.09499v1).
- **Safety constraints and data:**
  - Safe RLHF keeps harmlessness as a separate cost under a Lagrangian constraint — [arXiv 2310.12773](https://arxiv.org/abs/2310.12773) (K).
  - SafeSora provides human safety preferences and harmlessness labels for T2V — [arXiv 2406.14477](https://arxiv.org/abs/2406.14477); [HF dataset](https://huggingface.co/datasets/PKU-Alignment/SafeSora).
  - Listed only: VPO aligns a prompt rewriter for both quality and safety; MJ-VIDEO includes a safety expert among its reward dimensions — [curated list](https://github.com/chrisliu298/awesome-rl-for-video-generation).
- **Veo 3:**
  - Training data was "filtered to remove unsafe captions and personally identifiable information".
  - Content-policy violations and abuse found during testing "were mitigated prior to launch".
  - The model "was found to be able to produce deepfakes".
  - Post-training mitigations: SynthID watermarking and production filtering.
  - Source: [Veo 3 Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Veo-3-Model-Card.pdf).
- **Sora 2** (Sep 2025):
  - C2PA metadata on all assets, a visible moving watermark and internal detection tools.
  - Opt-in consent for cameo likeness — [Sora 2 system card](https://deploymentsafety.openai.com/sora-2) (snippet).
  - In a March 2026 update, cameos were renamed "characters"; owners decide who may use them and can revoke access — [FutureFive NZ](https://futurefive.co.nz/story/openai-details-sora-2-safeguards-on-likeness-teens); [SecurityBrief UK](https://securitybrief.co.uk/story/openai-details-sora-2-safeguards-on-likeness-teens) (secondary).
  - Image-to-video with real people requires users to attest consent and the right to upload, and carries stricter guardrails than characters — [StartupHub](https://startuphub.ai/news/sora-2-openai-s-safety-playbook) (secondary).
- **Seedance 2.0** (Feb 2026):
  - Viral celebrity deepfakes and copyrighted-character clips led to a Hollywood backlash, including legal threats from Disney.
  - ByteDance said it would strengthen safeguards against unauthorized use of IP and likeness, without saying what the measures were — [Euronews, 17 Feb 2026](https://www.euronews.com/2026/02/17/bytedance-says-it-will-add-safeguards-to-ai-video-tool-seedance-20-following-hollywood-bac); [Techstrong](https://techstrong.ai/articles/hollywood-hits-back-bytedance-pledges-safeguards-after-ai-smash-and-grab/).
  - Secondary reports say ByteDance suspended real-person image and video references in China, and pulled before launch a feature that generated a voice from a face photo — [Implicator](https://www.implicator.ai/bytedances-seedance-2-0-goes-viral-as-ai-celebrity-deepfakes-alarm-hollywood/) (secondary).
  - Reports on the global API's status conflict: one says it was "suspended indefinitely", another says people are supported, but not unrestricted celebrity likenesses — [lilting.ch](https://lilting.ch/en/articles/seedance-2-hollywood-copyright-api-suspended); [laozhang blog](https://blog.laozhang.ai/en/posts/seedance-2-api-real-people) (both secondary).
- **EU AI Act Art. 50 Code of Practice:**
  - Published in final form 10 Jun 2026; voluntary.
  - Two-layer marking: "secured metadata and watermarking", with optional fingerprinting.
  - Covers detection, including free detection tools, and a common set of EU labelling icons.
  - Two sections: providers' marking and detection, and deployers' labelling of deepfakes and of AI-generated text on matters of public interest.
  - Adherence "does not in itself constitute conclusive evidence of compliance".
  - Draft Art. 50 guidelines were published 8 May 2026. The obligations apply from 2 Aug 2026.
  - Sources: [Commission press release IP/26/1328](https://europa.eu/newsroom/ecpc-failover/pdf/ip-26-1328_en.pdf); [Jones Day](https://www.jonesday.com/en/insights/2026/06/european-commission-publishes-final-code-of-practice-on-marking-and-labelling-aigenerated-content); [SCL](https://www.scl.org/european-commission-publishes-code-of-practice-on-marking-and-labelling-ai-generated-content/); [ComplexDiscovery](https://complexdiscovery.com/europes-ai-labeling-rules-arrive-with-a-voluntary-code-and-a-hard-deadline/); [CEPIS](https://cepis.org/european-commission-publishes-final-ai-content-labelling-code-of-practice/).
  - Sources disagree on the drafting history: one gives a second draft on 5 Mar 2026, another a draft on 8 May 2026.
- **Digital Omnibus (secondary sources):**
  - The provisional trilogue deal of 7 May 2026 moves the Art. 50(2) marking deadline for systems already on the market from 2 Aug 2026 to 2 Dec 2026.
  - New systems placed on the market after 2 Aug 2026 get no transitional window, and the Art. 50(1) duty to tell users they are interacting with an AI system is not delayed.
  - Sources disagree on the length of the grace period (four months versus three), and one cites "Regulation (EU) 2026/1744".
  - An earlier leaked draft would have delayed Art. 50(2) penalties to 2 Aug 2027, but that is not the final outcome.
  - Sources: [CMS](https://cms.law/en/gbr/publication/digital-omnibus-on-ai-adjusted-timelines-and-transitional-provisions-explained); [CMS on the leaked draft](https://cms.law/en/gbr/publication/leaked-eu-s-digital-omnibus-on-ai-points-to-pragmatic-fixes-for-a-smoother-ai-act-rollout); [CSA research note, 29 Jul 2026](https://labs.cloudsecurityalliance.org/research/csa-research-note-eu-ai-act-article-50-transparency-20260729/); [European Outdoor Group](https://www.europeanoutdoorgroup.com/members-knowledge-hub/european-parliament-adopts-provisional-agreement-on-ai-omnibus); [Pillitteri](https://pasqualepillitteri.it/en/news/5190/digital-omnibus-ai-act-eu-parliament).
- **US TAKE IT DOWN Act:**
  - The FTC has enforced the platform notice-and-removal requirements since 19 May 2026. Valid requests must be acted on within 48 hours, with civil penalties of up to $53,088 per violation.
  - Before the deadline, the FTC Chairman wrote to more than a dozen companies.
  - The sources show no enforcement action against a named platform — [WilmerHale, 15 Jun 2026](https://wilmerhale.com/en/insights/client-alerts/20260615-the-take-it-down-act-goes-live); [Wiley](https://www.wiley.law/alert-May-19-Deadline-for-TAKE-IT-DOWN-Act-Compliance-Is-Your-Company-Prepared); [McNees](https://www.mcneeslaw.com/take-it-down-act-ftc-compliance-enforcement/).
- **NO FAKES Act:** reportedly cleared the Senate Judiciary Committee in June 2026; not enacted — [stackcyber tracker](https://stackcyber.com/posts/ai-deepfake-laws) (single secondary source, dated around Aug 2026).

### Inferences
- **Pipeline order:**
  - Run safety classifiers and likeness or consent checks before a take is shown, rated or logged.
  - Exclude unsafe or non-consensual takes from bandit rewards and from RL positives; keep them only in a separate safety dataset.
  - Treat the safety-violation rate as a hard constraint (Lagrangian or gate) that judge weights can never trade off.
- **Do not reward evasion of provider moderation.** If "generation succeeded" or "passed provider moderation" enters the reward, the bandit or reflection loop can learn prompt phrasings that evade Seedance, Veo, Kling or Sora filters. Treat provider refusals as neutral (no reward signal either way), and lint strategies for evasion patterns.
- **Real-person references:**
  - Require a verified, revocable consent record, following the Sora "characters" pattern.
  - Keep a data-lineage index, so that revocation also purges derived preference pairs, cached embeddings and any RL training shards that contain the likeness.
  - Provider policies can change abruptly (Seedance 2.0), so the planner must degrade gracefully when references are refused.
- **Provenance:**
  - Keep upstream C2PA and SynthID marks through editing.
  - Re-sign a C2PA manifest at export with the app recorded as editor.
  - Add a visible label for EU deepfake-type outputs. Photographic-surreal live action that resembles real people, places or events will often fall under the deployers' deepfake-labelling duty.
- **Provider status under Art. 50(2) (not legal advice).** When the app serves its own fine-tuned Wan 2.2 or in-house DiT, it plausibly becomes a provider under Art. 50(2). A system placed on the market after 2 Aug 2026 would get no grace period.
- **TAKE IT DOWN Act.** If users can publish or share outputs inside the app, plan for a 48-hour notice-and-removal workflow.

### Gaps
- China's labelling rules for AI-generated synthetic content (prior knowledge: in force from 1 Sep 2025, with explicit visible labels plus implicit metadata labels) were not verified this session. They matter because the Seedance, Kling and Wan providers are Chinese.
- Denmark's proposed likeness-rights law and state digital-replica laws were not checked.
- The state of the C2PA specification in 2026 and how well watermarks survive removal attacks were not researched.
- The final adoption and publication of the Digital Omnibus text in the Official Journal could not be confirmed (sources conflict). The Commission's own page was blocked.
- The Sora 2 system card's quantitative safety-evaluation results could not be read (host blocked).
