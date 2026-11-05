# RL and preference-optimization algorithms for aligning video diffusion / flow-matching models (state as of 2026-11-05)

Compiled 2026-11-05 for "Prompt-to-Scene Creator" (current loop: Flow-GRPO + MixGRPO sliding SDE window; target Wan 2.2 A14B MoE, later in-house multi-shot DiT; rewards = 7-dim VLM judge + learned RMs + logged pairwise human preferences; look = photoreal / photographic-surreal live action, never 3D animation).

Source-reliability tags used below:
- **[P]** = primary page read directly in this session (GitHub README / config / launch script).
- **[S]** = primary paper text seen only through search-engine snippets. arxiv.org was not fetched (blocked per assignment), and huggingface.co, alphaxiv.org, emergentmind.com and verl-omni.readthedocs.io all failed DNS resolution in this sandbox. These hosts must be allowed before anyone can verify the numbers directly.
- **[2nd]** = third-party summary, review, curated list or social-media post.
- **[BG]** = recalled from a paper abstract, not re-verified this session.

---

## Q1. Online RL for flow/diffusion video models: Flow-GRPO and its 2025–2026 descendants

### Takeaway
Between 2025 and 2026 the field moved from "SDE noise at every step + per-step log-probs" (Flow-GRPO, DanceGRPO, May 2025) in two directions. The first is "one or a few stochastic steps, ODE everywhere else": Flow-GRPO-Fast, MixGRPO, TempFlow-GRPO, G²RPO, and in 2026 Flash-GRPO (ICML 2026), which LingBot-Video adopted for video. The second is likelihood-free, forward-process or group-preference objectives (DiffusionNFT, DGPO, AWM, DiffusionART). These report 3–25× better efficiency, and in DiffusionART's case about 12× fewer GPU-hours on a 13B video model. They can also use deterministic or black-box samplers and train without CFG. The stability fixes that matter most for flow GRPO are:
- ratio normalization and per-timestep gradient reweighting (GRPO-Guard, Flash-GRPO "temporal gradient rectification");
- noise-aware timestep weighting (TempFlow);
- pairwise or win-rate rewards in place of raw pointwise scores (Pref-GRPO).

### Cited Findings

#### Flow-GRPO (Liu et al., May 2025; arXiv 2505.05470): the baseline the app already implements
- **Method.** Converts the deterministic flow ODE into a marginal-preserving SDE so group rollouts can explore. It optimizes a PPO-style clipped surrogate with KL to a reference model, and the paper says the KL helps prevent reward hacking. "Denoising reduction" means training rollouts use fewer steps than inference. [S] — [Emergent Mind summary](https://www.emergentmind.com/papers/2505.05470); [verl-omni Flow-GRPO doc (search snippet)](https://verl-omni.readthedocs.io/en/latest/algo/flowgrpo.html)
- **SDE noise.** [BG] σ_t = a·√(t/(1−t)), with a = noise level; a = 0.7 in the paper's SD3.5 runs. Headline gains on SD3.5-M: GenEval 63% → 95%, visual text rendering 59% → 92%. — [arXiv abstract](https://arxiv.org/abs/2505.05470) (not re-fetched). DiffusionNFT's comparison independently gives Flow-GRPO GenEval 0.95 after more than 5k steps with CFG. — [ICLR 2026 DiffusionNFT abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8e68ddfe22520b45b8fb8d5cbde5a21-Abstract-Conference.html)
- **Official SD3.5-M configs.** [P] 512 px; 10 training / 40 evaluation sampling steps; CFG 4.5; group size 24 (`num_image_per_prompt`); `timestep_fraction` 0.99; EMA on. KL β is 0.04 (GenEval and OCR), 0.01 (PickScore) and 0.02 (CLIPScore). The README's accounting is group_number = 48, group_size = 24. — [flow_grpo config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py); [flow_grpo README](https://github.com/yifan123/flow_grpo)
- **Official Wan2.1 video config (`general_ocr_wan2_1`).** [P] 240×416, 33 frames; 20 training / 50 evaluation steps; CFG 4.5; group size 4; β = 0.004; `clip_range` 1e-3; learning rate 1e-4; `timestep_fraction` 0.99; EMA on; LoRA (rank not visible); OCR reward. The reference video RL setup therefore runs at very low resolution and length. — [flow_grpo config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- **Other official configs.** [P]
  - FLUX: 6/28 steps, CFG 3.5, noise 0.9, β = 0.
  - Qwen-Image: 10/50 steps, CFG 4, group size 16, noise 1.2, SDE window size 2 within steps (0, 5), β = 0.
  - BAGEL: `train.cfg = True`, full fine-tune by default.

  — [config/grpo.py](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- **Flow-GRPO-Fast (added 2025-07-31).** [P] Generates a deterministic ODE trajectory, injects noise at a random intermediate step, and uses SDE sampling only for 1–2 steps. Pre-branch sampling uses one prompt without group expansion. On PickScore, 2 trained steps beat full Flow-GRPO (10 steps) and 1 step is "slightly worse". The method "may crash" unless `clip_range` is small; the fast configs use clip 1e-5, β = 0, noise 0.8 with CPS, and SDE window 3 within steps (0, 5) for SD3 or (0, 3) for FLUX. — [flow_grpo README](https://github.com/yifan123/flow_grpo); [config](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- **Official tips.** [P]
  - Start with β = 0 to check that reward rises at all.
  - Watch for over-optimization: the proxy score keeps rising while the gold score falls.
  - Log-prob errors are larger at low-noise steps, so training only high-noise steps "yields better results".
  - Removing CFG in training and test effectively performs CFG distillation.
  - Changelog: Wan2.1 support 2025-08-15; CPS and No-CFG 2025-10-14; GRPO-Guard 2025-11-04; Flow-GRPO in verl-omni 2026-05-07.

  — [flow_grpo README](https://github.com/yifan123/flow_grpo)

#### DanceGRPO (Xue et al., ByteDance Seed / HKU, May 2025; arXiv 2505.07818)
- **Results.** One GRPO formulation for diffusion and rectified flow, covering T2I, T2V and I2V (SD, FLUX, HunyuanVideo, SkyReels-I2V). It reports gains of "up to 181%" across benchmarks. On HunyuanVideo with VideoAlign, relative gains are 56% (visual quality) and 181% (motion quality). [S] — [arXiv](https://arxiv.org/abs/2505.07818v1); [project page](https://dancegrpo.github.io). A secondary summary gives motion quality 1.37 → 3.85. [2nd] — [Emergent Mind](https://www.emergentmind.com/topics/dance-grpo)
- **Video-reward stability.** The text–video-alignment (TA) reward was unstable and was excluded from the final analysis. Sampling more frames per second for VideoAlign improved training stability. [S] — [arXiv HTML v4](https://arxiv.org/html/2505.07818v4)
- **Official HunyuanVideo GRPO launch script.** [P]
  - Hardware: 4 nodes × 8 GPUs.
  - Video and sampling: 480×480, 53 frames, fps 8; 16 sampling steps; `eta` 0.25; `shift` 5; `cfg` 0.0 (HunyuanVideo is guidance-distilled).
  - Grouping: `num_generations` 24; `timestep_fraction` 0.6; `--use_same_noise` (shared initial noise in a group); `--use_group`; `--bestofn` 8.
  - Optimizer: learning rate 1e-5; weight decay 1e-4; `max_grad_norm` 1.0; gradient accumulation 4; bf16; EMA decay 0.999; 202 max steps.
  - Reward: VideoAlign with `vq_coef` 1.0 and `mq_coef` 0.0.
  - The script has no clip-range or KL flags.

  — [finetune_hunyuan_grpo.sh](https://github.com/XueZeyue/DanceGRPO/blob/main/scripts/finetune/finetune_hunyuan_grpo.sh)
- **README.** [P]
  - Training hardware: HunyuanVideo on 16 or 32 H800; SkyReels-I2V on 32 H800; Wan-2.1 on 8 H800; FLUX for 200 iterations ≈ 12 h on 16 H800. A LoRA FLUX variant uses about 20 GB VRAM per GPU on 8 GPUs.
  - Learning rate must be 5e-6–2e-5; 1e-6 "always leads to training failure".
  - The open-source HunyuanVideo run was cut from 73 to 53 frames. Visual-quality gain is about 50–60%. Wan-2.1 "needs more iterations to converge" (script released 2025-10-16).
  - VideoAlign de-normalization: VQ = VQ·2.2476 + 3.6757 and MQ = MQ·1.3811 + 1.1646.
  - If reward collapses, lower `max_grad_norm`. Use a large gradient accumulation when CFG is on. FSDP is preferred over DeepSpeed.

  — [DanceGRPO README](https://github.com/XueZeyue/DanceGRPO)

#### MixGRPO / MixGRPO-Flash (Tencent Hunyuan, Jul 2025; arXiv 2507.21802; accepted to ECCV 2026)
- **Method and speed.** Mixed ODE–SDE sampling with a sliding window: SDE sampling and optimization happen only inside the window, ODE outside. Training time drops about 50% versus DanceGRPO. MixGRPO-Flash runs a higher-order ODE solver (DPM-Solver++) outside the window and cuts training time by 71%. [S] — [arXiv HTML v3](https://arxiv.org/html/2507.21802v3)
- **Per-iteration cost.** 291.284 s for DanceGRPO versus 150.839 s for MixGRPO; policy forward passes fall from 14 to 4. [S] — [arXiv HTML v3](https://arxiv.org/html/2507.21802v3). A secondary summary gives 112 s and 83 s per iteration for the Flash variants and says the 71% comes from a frozen-window "Flash*" variant. [2nd] — [Emergent Mind](https://www.emergentmind.com/topics/mixgrpo-mixgrpo-flash)
- **Repo.** [P]
  - Default 32 GPUs (4×8).
  - FLUX.1-dev checkpoint trained with HPSv2 + ImageReward + PickScore as multi-reward.
  - 2025-10-02: paper updated with comparisons against Flow-GRPO, offline Flow-DPO and online Flow-DPO on SD3.5-M LoRA.
  - 2026-02-03: CPS sampler support.
  - 2026-07-01: ECCV 2026 acceptance.

  — [MixGRPO GitHub](https://github.com/Tencent-Hunyuan/MixGRPO)
- **Production use.** HunyuanVideo 1.5 (I2V online RL) and HunyuanImage 3.0 (DPO + MixGRPO). [S] — [HunyuanVideo 1.5 report](https://arxiv.org/html/2511.18870v1); [HunyuanImage 3.0 report](https://arxiv.org/pdf/2509.23951)

#### BranchGRPO (PKU / BNU / ByteDance, Sep 2025; arXiv 2509.06040) and TreeGRPO (Dec 2025)
- **BranchGRPO.** Replaces independent rollouts with tree branching that reuses shared prefixes. A tree-based advantage estimator turns sparse terminal rewards into step-level signals, and pruning removes low-value paths. Reported gain: up to 16% better HPDv2.1 alignment than DanceGRPO. [S] — [arXiv](https://arxiv.org/abs/2509.06040v4)
- **Conflict on the video setting.** The v5 abstract says T2I and image-to-video. The project page shows WanX-1.3B video with higher VideoAlign scores and sharper, more temporally consistent frames than DanceGRPO. — [project page](https://fredreic1849.github.io/BranchGRPO-Webpage)
- **TreeGRPO (arXiv 2512.08153).** A separate tree-advantage GRPO for diffusion post-training. [2nd] — [Emergent Mind](https://www.emergentmind.com/papers/2512.08153)

#### TempFlow-GRPO (ZJU / Tencent WeChat Vision, Aug 2025; arXiv 2508.04324)
- **Trajectory branching.** SDE exploration starts from one state on an otherwise deterministic ODE trajectory, so reward variance localizes to the branch point. This gives process-level credit without intermediate reward models.
- **Noise-aware weighting.** Larger updates at early, high-noise steps; gentler updates at late steps.
- **Seed grouping** controls for the effect of initial noise.
- **Result.** Noise-aware reweighting alone takes GenEval from 0.82 to 0.92 at 1,200 steps versus Flow-GRPO.

[S] — [arXiv HTML v2](https://arxiv.org/html/2508.04324v2)

#### Pref-GRPO (Aug 2025; arXiv 2508.20751)
- **Method.** A pairwise preference reward model compares samples within a group, and each sample's win rate becomes its reward.
- **Problem it targets.** Reward hacking is traced to an "illusory advantage": pointwise reward models give tightly clustered scores, and group normalization blows up tiny gaps. Example: UnifiedReward scores rise steadily while images drift toward an "unnaturally dark style".
- **Ablation.** Converting pointwise scores to win rates helps somewhat; explicit pairwise fitting helps more.

[S] — [alphaXiv abstract (via search)](https://alphaxiv.org/abs/2508.20751); [arXiv PDF](https://arxiv.org/pdf/2508.20751)
- **Cost.** 64 H20 GPUs, 25 sampling steps, 8 rollouts per prompt. [2nd] — [Liner review](https://liner.com/review/prefgrpo-pairwise-preference-rewardbased-grpo-for-stable-texttoimage-reinforcement-learning)

#### G²RPO (Oct 2025; arXiv 2510.01982; v3 of 22 Nov 2025 retitled "Fine-Grained GRPO for Precise Preference Alignment in Flow Models")
- **Singular Stochastic Sampling** confines stochasticity to individual steps.
- **Multi-Granularity Advantage Integration** denoises each sample with several ODE step counts and aggregates the advantages.
- **Result.** The sampling change alone gives +6.52% relative HPS-v2.1 over DanceGRPO.

[S] — [arXiv HTML v3](https://arxiv.org/html/2510.01982v3); [project page](https://bujiazi.github.io/g2rpo.github.io)

#### GRPO-Guard (Oct 2025; arXiv 2510.22319; listed as CVPR 2026)
- **Diagnosis.** In Flow-GRPO the importance-ratio mean "is consistently below 1", especially at low-noise steps (for example step 8 of SD3.5-M), and its variance differs across steps. Clipping therefore fails to restrain over-confident positive-advantage updates ("implicit over-optimization").
- **Fixes.** RatioNorm corrects the ratio bias and unifies statistics across steps; Gradient Reweight balances gradient contributions across denoising steps.

[P] — [flow_grpo README](https://github.com/yifan123/flow_grpo); [S] — [arXiv PDF](https://arxiv.org/pdf/2510.22319)
- **Results.** Reduces over-optimization while keeping or improving quality on SD3.5-M and FLUX.1-dev. GenEval gold score 0.89 versus 0.84 for Flow-GRPO. Trade-off: the proxy score rises more slowly. [2nd] — [alphaXiv overview](https://www.alphaxiv.org/overview/2510.22319v2); [papernotes (CVPR 2026)](https://en.papernotes.org/CVPR2026/image_generation/grpo-guard_mitigating_implicit_over-optimization_in_flow_matching_via_regulated_/)

#### DiffusionNFT (NVIDIA / Tsinghua, Sep 2025; arXiv 2509.16117; ICLR 2026 Oral)
- **Method.** Online RL on the forward process through the flow-matching loss. Positive and negative generations are contrasted to define an implicit policy-improvement direction. It works with any black-box solver, needs no likelihood estimation, needs only clean final samples rather than trajectories, and is CFG-free.
- **Results.** 3×–25× more efficient. GenEval 0.24 → 0.98 within 1k steps, versus Flow-GRPO 0.95 after more than 5k steps with CFG.

[S] — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8e68ddfe22520b45b8fb8d5cbde5a21-Abstract-Conference.html); [ICLR oral page](https://iclr.cc/virtual/2026/oral/10009150)
- **Official repo.** [P] SD3.5-M scripts only; no video. Multi-reward preset `sd3_multi_reward` covers GenEval, OCR, PickScore, ClipScore, HPSv2.1, Aesthetic, ImageReward and UnifiedReward (UnifiedReward-7B served via sglang). Default 8 GPUs. — [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)
- **Descendants and video use.** OmniNFT in verl-omni does audio-video post-training (LTX2.3 LoRA recipe, Oct 2026). [P] — [verl-omni](https://github.com/verl-project/verl-omni). MeanFlowNFT (arXiv 2607.15273) adapts NFT to few-step MeanFlow generators. [2nd] — [awesome-rm-for-video-generation](https://github.com/chrisliu298/awesome-rm-for-video-generation)

#### DGPO (Oct 2025; arXiv 2510.08425; ICLR 2026 poster) and other likelihood-free / ODE-native variants
- **DGPO.** Drops the policy-gradient framework and learns directly from group-level preferences. This allows ODE samplers, removes exploration noise, and avoids trajectory training, for a reported ~20× training speedup over Flow-GRPO. [S] — [arXiv](https://arxiv.org/abs/2510.08425); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007835). One reviewer notes limited theory on convergence and stability across group sizes. [2nd] — [Moonlight](https://www.themoonlight.io/review/reinforcing-diffusion-models-by-direct-group-preference-optimization)
- **Framework support.** AWM, DGPO, GRPO-Guard, DiffusionNFT, Flow-DPPO and CRD are all implemented in Flow-Factory; AWM, DGPO and Flow-DPPO are also in verl-omni. [P] — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory); [verl-omni](https://github.com/verl-project/verl-omni)

#### 2026 video-specific variants
- **Flash-GRPO** (arXiv 2605.15980, 15 May 2026; ICML 2026). Single-timestep GRPO for video with "iso-temporal grouping" (all group members are trained on the same timestep) and "temporal gradient rectification". Tested on 1.3B–14B models. The authors say a 14B model "typically demands hundreds of GPU days per experiment". Without rectification, reward collapses catastrophically between 300 and 400 GPU-hours. [S] — [arXiv](https://arxiv.org/abs/2605.15980); [ICML 2026](https://icml.cc/virtual/2026/poster/63629)
- **LingBot-Video** (arXiv 2607.07675, Jul 2026). Argues that in Flow-GRPO and DanceGRPO every step is stochastic but all steps share one trajectory-level advantage, which creates "a severe credit-assignment problem". It follows Flash-GRPO: exactly one stochastic step, with index k drawn uniformly from the first half of the schedule and shared by all G rollouts. [S] — [arXiv PDF](https://arxiv.org/pdf/2607.07675)
- **DiffusionART** (arXiv 2610.12572, 8 Oct 2026). On HunyuanVideo-13B it reaches the late-stage mean reward of DiffusionNFT and Flow-GRPO-Fast with 12.02× and 11.72× fewer GPU-hours (VQ and MQ rewards equally weighted). It needs no differentiable reward model, works with fewer samples per prompt, and supports pairwise reward models. [S] — [hyper.ai](https://hyper.ai/en/papers/2610.12572). Unverified secondary figures: 260.8 vs 883.3 GPU-hours in a pairwise-preference video test; Wan2.1 I2V reaches its motion target in 45.8 vs 220.3 GPU-hours. [2nd] — [LinkedIn repost](https://mer.vin/linkedin/diffusionart-faster-more-stable-rl-fine-tuning-for-diffusion-models-linkedin/)
- **LeanGRPO** (arXiv 2609.03528). Removes redundant recomputation in diffusion RL; benchmarks per-step time on Wan2.1-14B and compares against DanceGRPO on Wan2.1-1.3B. [S] — [arXiv PDF](https://arxiv.org/pdf/2609.03528)
- **Velocity matching** ("Scaling RL for Diffusion Models via Velocity Matching", arXiv 2608.23664). Fine-tunes Wan2.1 and reports per-iteration GPU-hour costs; numbers not retrieved. [S] — [arXiv PDF](https://arxiv.org/pdf/2608.23664)
- **OTCA** (arXiv 2604.19234). Splits coarse terminal video rewards into timestep-aware, per-objective credit along the trajectory. [S] — [arXiv HTML](https://arxiv.org/html/2604.19234v2)
- **One-line entries from a curated list.** [2nd] — [awesome-rm-for-video-generation](https://github.com/chrisliu298/awesome-rm-for-video-generation)
  - TAGRPO (2601.05729): GRPO on intermediate latents with shared initial noise.
  - Manifold-Aware Exploration (2603.21872): "curvature-corrected SDE sampler" that keeps exploration on the video manifold.
  - AR-CoPO (2603.17461): forks one autoregressive chunk and localizes credit to that pivot.
  - KVPO (2605.14278): ODE-native GRPO for streaming AR generators.
  - RAVEN (2605.15190): CM-GRPO for consistency samplers.
  - "Aligning Human Sense" (2608.21425): replaces the local KL in GRPO with a distributional term.
  - DM-Align (2609.04283).
  - V-GRPO (2604.23380): "Online RL for denoising generative models is easier than you think" ([arXiv PDF](https://arxiv.org/pdf/2604.23380)).
- **TMPO** (arXiv 2605.10983). Replaces reward maximization with trajectory-level reward-distribution matching (mode-covering, forward-KL-like), plus dynamic stochastic tree sampling to cut multi-trajectory cost. [S] — [alphaXiv listing](https://www.alphaxiv.org/abs/2605.10983)

#### Video-domain GRPO applications relevant to the app's reward dimensions
- **Identity-GRPO** (arXiv 2510.14256). Trains a video reward model on pairwise human-consistency annotations, then applies a GRPO variant for multi-human identity preservation. [S] — [arXiv](https://arxiv.org/abs/2510.14256)
- **IPRO** (CVPR 2026). Identity-preserving I2V with an ArcFace cosine-similarity reward and a multi-view scoring mechanism to prevent reward hacking. [S] — [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Identity-Preserving_Image-to-Video_Generation_via_Reward-Guided_Optimization_CVPR_2026_paper.pdf)
- **Camera-controlled video with a verifiable geometry reward** (arXiv 2512.02870). z-score group normalization; the reference model is initialized from the SFT model to reduce hacking. [S] — [arXiv PDF](https://arxiv.org/pdf/2512.02870)
- **VGGRPO** (arXiv 2603.26599). Latent-space GRPO with a 4D latent geometry reward plus KL to the reference. [S] — [arXiv PDF](https://arxiv.org/pdf/2603.26599)
- **InfLVG** (arXiv 2505.17574). Uses GRPO to train an inference-time context-selection policy for long video, with a face-identity + text–video + VLM-artifact reward. [2nd] — [Pith](https://pith.science/paper/2505.17574)
- **DS-GRPO.** Sharing initial noise across a group reduces reward variance and discourages hacking in video generation. [2nd] — [Emergent Mind topic](https://www.emergentmind.com/topics/ds-grpo)

### Inferences
- The app's design (Flow-GRPO with a MixGRPO window) matches a production recipe: HunyuanVideo 1.5's I2V online RL uses MixGRPO. It is no longer the most compute-efficient option, though. The 2026 video-specific evidence (Flash-GRPO, LingBot-Video, OTCA) points to:
  - a single stochastic step drawn from the first half of the schedule, shared across the group ("iso-temporal grouping");
  - per-timestep gradient or ratio normalization (GRPO-Guard RatioNorm, or Flash-GRPO rectification);
  - per-objective, timestep-aware credit.
- Likelihood-free methods are the most promising for a 14B MoE and for a long-sequence multi-shot DiT, where per-step log-probs over huge token counts are costly and numerically noisy. Candidates: DiffusionNFT/OmniNFT, DiffusionART, DGPO and AWM. They also let RL use the production sampler (few-step or distilled), which removes the train/serve sampler mismatch.
- Pref-GRPO's "illusory advantage" maps directly onto Likert-style VLM-judge scores, which are tightly clustered integers. Asking the judge for within-group pairwise comparisons (win rates), or reading out judge probabilities, is the likely fix.
- Shared initial noise within a group is a cheap and widely used variance-reduction trick (DanceGRPO `--use_same_noise`, flow_grpo `same_latent`, DS-GRPO, TempFlow seed grouping), and it is directly applicable.

### Gaps
- Exact MixGRPO window hyperparameters (window size, shift interval, stride) were not recoverable; the README omits them and arxiv.org could not be fetched.
- The Flow-GRPO paper's LoRA rank/alpha and default `clip_range` are not visible in the config. [BG] recollection of r = 32 / α = 64 and clip ≈ 1e-4 is unverified.
- BranchGRPO's branching schedule was not recovered, and the T2V vs I2V scope conflict is unresolved.
- DiffusionNFT hyperparameters (β/mixing, loss weighting) were not recovered; its config file was not shown.
- Flash-GRPO's absolute speedups and its backbone list were not retrieved.
- No paper was found that publishes GRPO-family results specifically on Wan 2.2 A14B (MoE); only framework recipes exist (see Q5).
- No multi-shot-specific GRPO paper was found. The closest are LongCat-Video-Avatar 1.5's temporally partitioned rewards (Q4), AR-CoPO's chunk-level credit, and identity-reward work.

---

## Q2. Offline / preference methods (Diffusion-/Flow-DPO, VideoDPO, Flow-RWR, Flow-NRG, DenseDPO, online/iterative DPO, KTO, RWR): when to prefer DPO on logged human pairs vs online GRPO

### Takeaway
DPO is still the most widely disclosed industrial alignment method for video: Step-Video, Seaweed, Kling-Omni, Alice v1, and HunyuanVideo 1.5's first RL stage. The best evidence-based refinements are:
- a fixed (not timestep-dependent) β (Flow-DPO);
- structurally aligned pairs with segment-level labels, which fixes DPO's bias toward low-motion clips (DenseDPO);
- real footage as the "chosen" sample (RealDPO, cIPO), which matters for photorealism.

Offline DPO is the right first stage when human pairs exist and the target attribute, such as fine motion or an "AI/CG look", is something reward models judge poorly. Online GRPO-family RL is the second stage, for pushing past the logged data distribution with cheap, reliable rewards. That is HunyuanVideo 1.5's order.

### Cited Findings

#### Flow-DPO / Flow-RWR / Flow-NRG (Kling team + collaborators, Jan 2025; arXiv 2501.13918; NeurIPS 2025)
- **VideoReward.** A multi-dimensional reward model (visual quality, motion quality, text alignment) trained on 182k (prompt, video A, video B) human-preference triplets. It significantly outperforms prior video reward models.
- **Algorithms.** Flow-DPO and Flow-RWR are training-time methods; Flow-NRG applies reward guidance directly to noisy videos at inference.
- **Key finding.** A timestep-dependent β makes Flow-DPO perform poorly on some tasks (uneven training across noise levels with shared weights). With a fixed β, Flow-DPO becomes the most effective method and beats Flow-RWR and SFT.

[S] — [arXiv HTML v2](https://arxiv.org/html/2501.13918v2); [project page](https://gongyeliu.github.io/videoalign); [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/76227feb18ea0ee40bd15cf02c33e18e-Abstract-Conference.html)

#### DenseDPO (Snap Research et al., Jun 2025; arXiv 2506.03517; NeurIPS 2025)
- **Problem.** Standard DPO compares videos from independent noise. That permits only coarse comparisons and biases annotators toward low-motion clips.
- **Fix.** Build pairs by denoising corrupted copies of one ground-truth video, which keeps motion structure aligned. Label preferences on short segments (for example 1-second slices).
- **VLM labels.** GPT (o3) predicts segment preferences about as well as task-tuned video reward models such as VisionReward. DenseDPO trained on those labels gets close to training on human labels.
- **Result.** With one third of VanillaDPO's labeled data, it significantly improves dynamic degree and matches VanillaDPO on text alignment, visual quality and temporal consistency (VideoJAM-bench, MotionBench).

[S] — [arXiv HTML v2](https://arxiv.org/html/2506.03517v2); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/117435); [2nd] — [Moonlight review](https://www.themoonlight.io/review/densedpo-fine-grained-temporal-preference-optimization-for-video-diffusion-models)

#### Real-footage-anchored preference methods (directly relevant to the photoreal / anti-3D goal)
- **RealDPO** (arXiv 2510.14955, Oct 2025). Real videos are the "win" samples and model outputs the "lose" samples; it uses the RealAction-5K human-action dataset. It avoids reward-model limitations and hacking, reports gains in quality, alignment and motion realism, and is bounded by the base model's expressiveness. [S] — [arXiv HTML v2](https://arxiv.org/html/2510.14955v2)
- **cIPO** (arXiv 2607.28058, 2026). "builds annotation-free pairs by treating a real clip as chosen and its noise-then-denoise reconstruction as rejected". [2nd] — [awesome-rm-for-video-generation](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **DDRL** (NVIDIA Cosmos, Dec 2025; arXiv 2512.04332). Replaces on-policy reverse-KL regularization with forward KL to an off-policy (real or synthetic) data distribution. It names on-policy regularization as the main cause of reward hacking, whose symptoms are "quality degradation, over-stylization, or reduced diversity". On Cosmos2.5 2B/14B T2V and I2V it improves rewards with less hacking and the highest human preference, using more than 1M GPU-hours and 10k double-blind human votes. Code is in cosmos-rl. [S] — [arXiv](https://arxiv.org/abs/2512.04332); [NVIDIA project page](https://research.nvidia.com/labs/cosmos-lab/ddrl/)

#### Online / iterative DPO and other 2026 offline-family variants
- **Framework support.** [P]
  - Flow-Factory implements offline Diffusion-DPO (`offline-dpo`) and online DPO (`dpo`) for every supported model type, including Wan2.1/2.2 T2V/I2V and LTX-2. — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
  - verl-omni has DPO for Qwen-Image and SD3.5 and says "offline DPO remains on v0 by design". — [verl-omni](https://github.com/verl-project/verl-omni)
- **One-line entries from a curated list.** [2nd] — [awesome-rm-for-video-generation](https://github.com/chrisliu298/awesome-rm-for-video-generation)
  - Diffusion-APO (2605.07503): "Trajectory-aware DPO that builds preference pairs online from a pairwise video-ranking oracle".
  - IPA (2605.07545): no pairs; maximizes likelihood of self-generated high-quality samples under a pretrained-prior penalty.
  - SHIFT (2603.17426): "advantage-weighted fine-tuning that unifies supervised and reward objectives" (an RWR-like approach).
- **MixGRPO's comparison.** The updated paper compares against offline Flow-DPO and online Flow-DPO on SD3.5-M LoRA; results not retrieved. [P] — [MixGRPO README](https://github.com/Tencent-Hunyuan/MixGRPO)

#### Industrial evidence on DPO vs online RL (details in Q4)
- **HunyuanVideo 1.5** runs offline DPO first because existing reward models cannot reliably separate fine-grained motion quality (paraphrased from a search summary), then online RL with MixGRPO. [S] — [arXiv HTML](https://arxiv.org/html/2511.18870v1)
- **Step-Video-T2V Video-DPO.** Humans label multiple seeds per prompt; training on the model's own outputs "ensur[es] smooth updates and improv[es] overall training stability". [S] — [arXiv HTML v3](https://arxiv.org/html/2502.10248v3)
- **Seedance 1.0** reports that direct multi-reward maximization beat DPO, PPO and GRPO. [S] — [arXiv HTML](https://arxiv.org/html/2506.09113v1)

#### Background (not re-verified)
- [BG] Diffusion-DPO (Wallace et al., Nov 2023, arXiv [2311.12908](https://arxiv.org/abs/2311.12908)) is the original DPO-for-diffusion loss on noised win/lose pairs relative to a frozen reference. Diffusion-KTO ([2404.04465](https://arxiv.org/abs/2404.04465)) is the unpaired, binary-feedback variant. VideoDPO ([2412.14167](https://arxiv.org/abs/2412.14167), Dec 2024) combines visual quality and semantic alignment into an "OmniScore" with pair re-weighting. These details were not retrieved this session.

### Inferences
- **Decision rule.**
  - Use offline DPO, with fixed β, on logged human pairs (and VLM-labeled DenseDPO-style pairs) when labels already exist, when the attribute is poorly captured by reward models (micro-motion, physics, "CG/plastic" look), or when you want a conservative stage with no rollout cost and no reward-model drift.
  - Use online GRPO/NFT when the reward is cheap and trustworthy, when you need on-policy exploration beyond the logged distribution, or when you are optimizing verifiable dimensions such as storyboard composition match and identity.
- **The anti-3D-animation goal is the strongest case for real-footage anchoring.** Options: RealDPO/cIPO real-vs-generated pairs, DenseDPO pairs built from corrupted real clips, and DDRL-style forward-KL to a curated live-action corpus. DDRL names "over-stylization" as a characteristic hacking mode of on-policy regularized RL; Pref-GRPO and LongCat give concrete drift examples.
- **Three ways to use logged pairwise human preferences:**
  - DPO directly;
  - train a Bradley–Terry / pairwise reward model (VideoReward-style) and feed it to Pref-GRPO or DiffusionART, which accept pairwise rewards;
  - keep a held-out slice to calibrate the VLM judge and detect reward hacking.

### Gaps
- No source compared offline DPO on human pairs against online GRPO with reward models on the same video model at matched compute. MixGRPO's image-only comparison was not retrieved.
- No "OnlineVPO" or video "IPO" paper surfaced; the names could not be confirmed.
- VideoDPO and Diffusion-KTO specifics, and any video-KTO results, were not retrieved.
- No β values for Flow-DPO were recovered.

---

## Q3. Direct reward backpropagation (ReFL, DRaFT, AlignProp, VADER, 2026 successors) vs RL for video

### Takeaway
Gradient-based reward fine-tuning is the most sample-efficient approach. ByteDance's Seedance 1.0 chose direct multi-reward maximization over DPO, PPO and GRPO, and VADER fits video reward backprop into 16 GB with truncated backprop. It needs differentiable, ideally latent-space, reward models and is the most exposed to adversarial reward hacking. A black-box VLM judge such as the app's 7-dimension judge cannot drive it directly, so the realistic pattern is reward backprop on in-house differentiable latent rewards plus RL or best-of-N for the VLM-judge dimensions.

### Cited Findings
- **VADER** (Jul 2024; arXiv 2407.08737). Backpropagates gradients from pre-trained reward models through denoising. Truncated backprop (1 step), LoRA, frame subsampling and gradient checkpointing make it fit on 16 GB VRAM. Reported to learn "much more efficiently in terms of reward queries and computation" than DDPO and DPO. [2nd] — [Moonlight review](https://www.themoonlight.io/review/video-diffusion-alignment-via-reward-gradients); [Emergent Mind](https://www.emergentmind.com/papers/2407.08737)
- **Diffusion-DRF** (Jan 2026; arXiv 2601.04153). Uses a frozen off-the-shelf VLM as a differentiable critic with structured, non-scalar feedback. Claims this gives "more stable gradient propagation and mitigates reward hacking or training collapse", and outperforms Flow-GRPO by 4.74% overall on unseen VBench-2.0. It argues existing video reward models are not robust enough to prevent hacking and collapse. [S] — [arXiv HTML v2](https://arxiv.org/html/2601.04153v2)
- **Seedance 1.0 "Base Model Feedback Learning".** Simulates the inference pipeline during training, predicts the clean video directly, scores it with reward models, and "directly maximizes the composite rewards from multiple RMs". Reported as the most efficient and effective approach compared with DPO, PPO and GRPO. Multi-round iterative learning between the diffusion model and the reward models is "more stable and controllable than dynamic update of the RM". The same approach is used for refiner RLHF. [S] — [arXiv HTML](https://arxiv.org/html/2506.09113v1); [Seed blog](https://seed.bytedance.com/en/blog/tech-report-of-seedance-1-0-is-now-publicly-available)
- **Latent / differentiable reward infrastructure (2026).**
  - verl-omni added "FlowGRPO with DiNa-LRM", which skips VAE decoding (Jul 2026). [P] — [verl-omni](https://github.com/verl-project/verl-omni)
  - Reward Lightning (2607.03960): a latent-space reward model sharing its backbone with adversarial distillation. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
  - Seedance 1.0 uses a "latent-space motion" reward model. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **PISCES** (2602.01624). Annotation-free post-training via direct reward backprop and RL. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **GigaVideo-1** (Jun 2025; arXiv 2506.10639). Title claims "Advancing Video Generation via Automatic Feedback with 4 GPU-Hours Fine-Tuning"; details not retrieved. [S] — [arXiv HTML](https://arxiv.org/html/2506.10639v1)
- **Contrast with RL.** DiffusionART explicitly "requires no differentiable reward model". [S] — [hyper.ai](https://hyper.ai/en/papers/2610.12572)
- [BG] ReFL (ImageReward, 2023), DRaFT (2023) and AlignProp (2023) are the image-domain originals (truncated gradients through the last K sampling steps). No video-specific comparison of them surfaced.

### Inferences
- **Pros vs RL for video.** Far fewer samples per gain, since there is no group of 8–24 rollouts per prompt. It optimizes exactly the decoded output, and it pairs naturally with refiner or super-resolution stages, as Seedance did for its refiner.
- **Cons.**
  - Needs differentiable rewards: the external VLM judge is not differentiable.
  - Memory grows with video length (decode plus reward-model backward pass), which pushes you to truncation or frame subsampling.
  - Gradients can find adversarial directions in the reward model, so hacking risk is higher.
  - Seedance mitigates this with several reward models and periodic reward-model retraining between rounds rather than live updates.
- **Pragmatic split for the app.**
  - Train small differentiable latent rewards (photorealism vs CG classifier, identity embedding similarity across shots, composition match against the storyboard layout) and use them with ReFL/DRaFT-style truncated backprop, or as dense signals.
  - Keep the 7-dimension VLM judge for GRPO, NFT, ART or best-of-N.
  - Diffusion-DRF suggests VLM critics can be made differentiable if the judge model is self-hosted.

### Gaps
- No head-to-head memory or GPU-hour comparison of VADER/DRaFT/AlignProp vs GRPO at 14B video scale was found.
- No AlignProp or DRaFT video results surfaced.
- Seedance's reward weights and schedule were not retrieved.

---

## Q4. What industrial technical reports disclose about video RLHF

### Takeaway
Chinese labs disclose the most. Common patterns:
- multi-dimensional, often VLM-based reward models with a separate motion reward model;
- DPO as the most common algorithm;
- online GRPO in HunyuanVideo 1.5 (MixGRPO), LongCat-Video (multi-reward GRPO), LingBot-Video (Flash-GRPO, 2026) and an April 2026 "systematic post-train" report (Flow-GRPO, about +31% GSB);
- ByteDance Seed's direct reward maximization with iterative reward-model refresh, applied to both base model and refiner.

No retrieved source discloses an RLHF algorithm for Seedance 2.0, Veo 3, Sora 2, Wan 2.x, MAGI-1 or LTX-2.

### Cited Findings

#### ByteDance Seedance
- **Seedance 1.0** (Jun 2025; arXiv 2506.09113). SFT then RLHF with three reward models: Foundational (VLM-based; image–text alignment and structural stability), Motion, and Aesthetic. Training "directly maximizes the composite rewards from multiple RMs", was found best against DPO, PPO and GRPO, and uses multi-round iterative learning between model and reward models. A separate super-resolution refiner RLHF stage maximizes a linear combination of the rewards. Reward curves show "stable and consistent upward trend" for both base and refiner. [S] — [arXiv HTML](https://arxiv.org/html/2506.09113v1); [Seed blog](https://seed.bytedance.com/en/blog/tech-report-of-seedance-1-0-is-now-publicly-available). A curated list describes the reward models as "VLM-based foundational, latent-space motion, keyframe aesthetic". [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **Seedance 1.5 pro** (Dec 2025; arXiv 2512.13507). Audio-video joint model; SFT then RLHF with multi-dimensional reward models.
  - Secondary sources disagree on the dimensions. One lists motion, audio fidelity, AV sync and aesthetics. One lists motion, aesthetics, audio fidelity and prompt adherence. The curated list says three reward models: AV alignment, motion dynamics, aesthetics.
  - Reported about 3× faster RLHF training from infrastructure work, and more than 10× inference acceleration.

  [S]/[2nd] — [arXiv](https://arxiv.org/abs/2512.13507); [Emergent Mind topic](https://www.emergentmind.com/topics/seedance-1-5-pro); [ChatPaper](https://chatpaper.com/paper/219064); [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **Seedance 2.0** (released in China early Feb 2026; arXiv 2604.14148, 15 Apr 2026). Unified multi-modal audio-video architecture; 4–15 s output at native 480p/720p; text, image, audio and video inputs (up to 3 video clips, 9 images and 3 audio clips on the open platform); a "Fast" low-latency variant. No RLHF or reward-model details appeared in any retrieved source. [S] — [arXiv abstract](https://arxiv.org/abs/2604.14148); [2nd] — [paper_notes #5233](https://github.com/AkihikoWatanabe/paper_notes/issues/5233)

#### Tencent HunyuanVideo 1.5 (Nov 2025; arXiv 2511.18870)
- Post-training runs continuing training, then SFT, then RLHF, with separate T2V and I2V strategies aimed mainly at artifacts and motion quality.
- I2V uses online RL that varies random seeds and CFG scales and uses the hybrid ODE–SDE MixGRPO to enrich exploration while keeping sampling quality (paraphrased from a search summary).
- Because reward models cannot separate fine-grained motion quality, an offline DPO stage comes first. Secondary sources say T2V uses DPO followed by online RL; the order is not explicit per task in the snippet.

[S] — [arXiv HTML](https://arxiv.org/html/2511.18870v1); [2nd] — [Emergent Mind](https://www.emergentmind.com/papers/2511.18870)
- The report trains its own VLM-based video reward model scoring text alignment, image alignment, visual quality and motion. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)

#### Meituan LongCat-Video (Oct 2025; arXiv 2510.22200; 13.6B) and LongCat-Video-Avatar 1.5 (2026)
- **Reward terms.** Multi-reward GRPO with four terms:
  - HPSv3-general: mean over all frames with the prompt "A high-quality image";
  - HPSv3-percentile: scored against the caption, averaging the top 30% of frames;
  - VideoAlign-based motion quality;
  - VideoAlign-based text alignment.

  All weights are 1. LoRA is applied to self-attention, cross-attention and FFN. The motion reward offsets the "static tendency" that HPSv3 alone induces. [S] — [hyper.ai](https://hyper.ai/en/papers/2510.22200); [arXiv HTML v2](https://arxiv.org/html/2510.22200v2)
- **Other details.** The motion reward model uses grayscale input "to suppress appearance shortcuts". [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation). An ablation shows gradient reweighting matters for GRPO stability. [2nd] — [alphaXiv overview](https://alphaxiv.org/overview/2510.22200v2)
- **LongCat-Video-Avatar 1.5** (arXiv 2605.26486). Extends multi-reward GRPO with per-frame and temporally partitioned rewards that localize motion inconsistency, hand deformation and structural collapse. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)

#### Other 2026 reports
- **LingBot-Video** (arXiv 2607.07675, Jul 2026; MoE video pretraining for embodied AI). Six reward signals combined through decoupled per-reward normalization; Flash-GRPO-style single stochastic step. [S] — [arXiv PDF](https://arxiv.org/pdf/2607.07675)
- **"A Systematic Post-Train Framework for Video Generation"** (arXiv 2604.25427, Apr 2026; HKU + collaborators, first author Xue; internal model; no company named).
  - Stages: SFT, then GRPO RLHF adapting Flow-GRPO with four reward models (video aesthetics, image aesthetics, motion quality, text–video alignment) in two stages, then a GRPO-trained prompt-enhancement LLM, then autoregressive distillation.
  - About +31% overall GSB, mainly visual and motion quality; text alignment improved less.

  [S]/[2nd] — [arXiv PDF](https://arxiv.org/pdf/2604.25427); [Moonlight](https://www.themoonlight.io/review/a-systematic-post-train-framework-for-video-generation)
- **Alice v1** (arXiv 2605.08115, 2026). Generation-specific verifiers for hard-example mining, "then DPO on ~10K human A/B video-preference pairs". [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **NVIDIA Cosmos 2.5 + DDRL.** Forward-KL data-regularized RL on 2B/14B T2V and I2V (see Q2). [S] — [project page](https://research.nvidia.com/labs/cosmos-lab/ddrl/)

#### Kuaishou Kling
- **VideoAlign / Flow-DPO paper** is a Kling team collaboration (see Q2). [S] — [arXiv](https://arxiv.org/html/2501.13918v2)
- **Kling-Omni** (arXiv 2512.16776). Progressive training from instruction pre-training through SFT to RL. Sources conflict on what the RL stage covers.
  - A search summary says the RL reward targets "factual correctness, content richness, and semantic plausibility", which would be prompt processing.
  - The curated list says "Multi-round DPO on human preference pairs over sampled video variants".

  [S] — [arXiv HTML](https://arxiv.org/html/2512.16776v1); [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **Versions.** Kling 3.0 launch is confirmed by Kuaishou IR; Wikipedia lists a Kling 4.0 release in Oct 2026, which was not reconciled with official sources. — [Kuaishou IR](https://ir.kuaishou.com/news-releases/news-release-details/kling-ai-launches-30-model-ushering-era-where-everyone-can-be); [Wikipedia](https://en.wikipedia.org/wiki/Kling_AI)

#### StepFun, ByteDance Seaweed, others
- **Step-Video-T2V** (Feb 2025). Video-DPO to reduce artifacts. Annotators label multiple seeds per prompt; pairs are on-policy for stability. [S] — [arXiv HTML v3](https://arxiv.org/html/2502.10248v3). It also "trains an in-house human-feedback reward model that scores on-policy samples online and is periodically refreshed". [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **Seaweed-7B** (Apr 2025). "Video-DPO on annotator best/worst picks from four generations per prompt". [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)
- **Waver** (Aug 2025) uses a data-filtering quality model, not RLHF. **LTX-Video** (Jan 2025) trained "a Siamese aesthetic ranker on human-tagged pairs" for data filtering. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)

#### Labs with no algorithmic disclosure found
- **Veo 3.** The tech report and model card (23 May 2025) describe latent diffusion, safety mitigations in pre- and post-training data (including SynthID), and human-rater preference evaluations. No RLHF algorithm is disclosed. [S] — [Veo 3 Tech Report](https://storage.googleapis.com/deepmind-media/veo/Veo-3-Tech-Report.pdf); [Veo 3 Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Veo-3-Model-Card.pdf)
- **Sora 2.** No system-card post-training disclosure surfaced (gap).
- **MAGI-1** (arXiv 2505.13211). Retrieved material covers only the three-stage pretraining and chunk-wise autoregressive conditioning; no RL found. [S] — [arXiv PDF](https://arxiv.org/pdf/2505.13211)
- **Wan 2.x.** No official Alibaba disclosure of RL or RLHF for Wan 2.2/2.5/2.6 was found; only product pages. [S] — [search result pages, e.g. OpenRouter Wan 2.6](https://openrouter.ai/ko-KR/alibaba/wan-2.6). The Wan 2.2 A14B card describes a high-noise expert (layout) and a low-noise expert (detail refinement), each about 14B, switching at an SNR threshold t_moe set to half of SNR_min. [S] — [Wan2.2-T2V-A14B model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- **LTX-2.** No RL disclosure found in its report. Open frameworks do support RL on it:
  - Flow-Factory: LTX-2 (19B) and LTX-2.3 (22B) T2AV/I2AV since 2026-04-25.
  - verl-omni: LTX2.3 FlowGRPO (Aug 2026) and an OmniNFT LTX2.3 LoRA recipe (Oct 2026).

  [P] — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory); [verl-omni](https://github.com/verl-project/verl-omni)

### Inferences
- **Common production pattern.** Multi-dimensional rewards with a dedicated motion reward model; an aesthetics or quality reward that on its own pushes toward static or over-stylized output (LongCat, DDRL); and stage-wise RLHF: SFT, then DPO or reward-feedback learning, then online GRPO, with RLHF also applied to refiners and to prompt rewriters.
- For a story-to-video app, the systematic post-train report's separate GRPO-trained prompt enhancer and Seedance's refiner RLHF are low-risk extra levers that don't touch the base generator.
- Using several reward models with equal or decoupled weights, plus reward-model refresh between rounds rather than online reward-model updates, is the disclosed industrial hedge against hacking.

### Gaps
- No RLHF details for Seedance 2.0, Sora 2, Veo 3, Wan 2.2/2.5/2.6, MAGI-1, LTX-2, MiniMax Hailuo or Vidu.
- Kling 2.x/3.0 RL specifics are unclear (Kling-Omni descriptions conflict).
- The Seedance 1.0 reward weights and number of iterative rounds were not retrieved.
- HunyuanVideo 1.5's DPO pair counts and RL hyperparameters were not retrieved.

---

## Q5. Practical recipes: LoRA vs full FT, timestep selection, CFG, mixed ODE-SDE, multi-reward normalisation, diversity, frameworks and hardware

### Takeaway
The defensible 2026 default for a 14B video model is:
- LoRA on attention and FFN (LongCat, flow_grpo, Flow-Factory), with learning rate about 1e-4 for LoRA or 5e-6–2e-5 for full fine-tuning;
- bf16 (required for Wan);
- shared initial noise per group;
- one or a few SDE steps in the high-noise half of the schedule (MixGRPO / Flash-GRPO / CPS), with ODE elsewhere;
- a small clip (1e-3 down to 1e-5) plus RatioNorm-style normalization, and β about 0.004 KL or data regularization;
- decoupled per-reward advantage normalization, with pairwise or win-rate signals for VLM judges;
- CFG either disabled during RL (which acts as CFG distillation) or randomized for exploration.

Open frameworks that support Wan 2.2 A14B RL as of Nov 2026: Flow-Factory (Wan2.2 T2V/I2V-A14B, TI2V-5B; GRPO, NFT, DGPO, AWM, DPO; LoRA) and verl-omni (Wan2.2 DanceGRPO; fully async v0.3.0 of Oct 2026). The Miles docs also describe a Wan2.2-A14B Flow-GRPO recipe, with expert routing by timestep (boundary 0.875).

### Cited Findings

#### LoRA vs full fine-tuning
- **flow_grpo.** LoRA is the default (`use_lora = True`) for SD3, Wan2.1 and Qwen-Image. Full BAGEL training "requires at least 8 × 80GB GPUs", with a LoRA fallback for out-of-memory cases. [P] — [flow_grpo README](https://github.com/yifan123/flow_grpo); [config](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- **DanceGRPO.** Main recipes are full fine-tunes on FSDP with learning rate 5e-6–2e-5 (1e-5 for HunyuanVideo). The FLUX LoRA variant uses about 20 GB VRAM per GPU on one 8-GPU node. [P] — [DanceGRPO README](https://github.com/XueZeyue/DanceGRPO); [Hunyuan script](https://github.com/XueZeyue/DanceGRPO/blob/main/scripts/finetune/finetune_hunyuan_grpo.sh)
- **LongCat-Video** applies LoRA to self-attention, cross-attention and FFN for its production GRPO. [S] — [hyper.ai](https://hyper.ai/en/papers/2510.22200)
- **Flow-Factory.** All example recipes are LoRA (for example `examples/grpo/lora/flux1/default.yaml`). [P] — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
- **Miles docs.** Wan2.2-T2V-A14B "Flow-GRPO + PickScore recipe" with both full-fine-tune and LoRA variants, single- and multi-node. [2nd] — [Miles Wan2.2 doc](https://radixark.mintlify.app/diffusion/models/wan/wan2-2)

#### Timestep selection, fractions and credit assignment
- **flow_grpo.**
  - `timestep_fraction` 0.99 in SD3 and Wan configs.
  - Fast or windowed configs train an SDE window of 3 steps within the first 5 (SD3) or first 3 (FLUX); Qwen-Image uses a window of 2 within the first 5.
  - Log-prob errors grow at low-noise steps, so training only high-noise steps works better.

  [P] — [config](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py); [README](https://github.com/yifan123/flow_grpo)
- **DanceGRPO** trains a `timestep_fraction` of 0.6 for HunyuanVideo. [P] — [script](https://github.com/XueZeyue/DanceGRPO/blob/main/scripts/finetune/finetune_hunyuan_grpo.sh)
- **MixGRPO** cuts policy forward passes from 14 to 4 through its window. [S] — [arXiv HTML v3](https://arxiv.org/html/2507.21802v3)
- **Flash-GRPO / LingBot-Video** use one stochastic step drawn uniformly from the first half of the schedule and shared by the group. [S] — [LingBot-Video](https://arxiv.org/pdf/2607.07675); [Flash-GRPO](https://arxiv.org/abs/2605.15980)
- **TempFlow-GRPO** gives early, high-noise steps larger weight. **GRPO-Guard** reweights gradients across steps. **OTCA** assigns timestep-aware, per-objective credit. [S] — [TempFlow](https://arxiv.org/html/2508.04324v2); [GRPO-Guard](https://arxiv.org/pdf/2510.22319); [OTCA](https://arxiv.org/html/2604.19234v2)

#### CFG handling during RL
- **flow_grpo.** SD3 and Wan configs sample with CFG 4.5 during RL. The "nocfg" configs use guidance 1 in both training and evaluation, which "effectively perform[s] CFG distillation" and is listed as a speed tip. [P] — [README](https://github.com/yifan123/flow_grpo); [config](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- **DanceGRPO HunyuanVideo** uses `cfg 0.0` on a guidance-distilled model. With CFG enabled, the authors use a large gradient accumulation (num_generations × 20) so the gradient updates once per rollout. [P] — [README](https://github.com/XueZeyue/DanceGRPO)
- **DiffusionNFT** is CFG-free and reaches GenEval 0.98 without CFG. [S] — [ICLR abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8e68ddfe22520b45b8fb8d5cbde5a21-Abstract-Conference.html)
- **HunyuanVideo 1.5** randomizes CFG scales (and seeds) during online RL to enrich exploration. [S] — [arXiv HTML](https://arxiv.org/html/2511.18870v1)
- **Flow-Factory** runs MiniMax H3 audio-video RL with "batch size 1, no CFG, and separate video/audio trajectories". [P] — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)

#### Mixed ODE–SDE sampling and exploration samplers
- **Options.**
  - MixGRPO window; MixGRPO-Flash with DPM-Solver++ outside the window. [S] — [arXiv HTML v3](https://arxiv.org/html/2507.21802v3)
  - Coefficients-Preserving Sampling (CPS), with `noise_level` 0.8 "typical" and working without tuning, supported in flow_grpo (Oct 2025) and MixGRPO (Feb 2026). [P] — [flow_grpo README](https://github.com/yifan123/flow_grpo); [MixGRPO README](https://github.com/Tencent-Hunyuan/MixGRPO)
- **Manifold-Aware Exploration** (2603.21872) uses a curvature-corrected SDE sampler to keep video exploration on the learned manifold. [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation)

#### Precision, debugging and stability guards
- **flow_grpo.**
  - Prefer fp16, but FLUX and Wan need bf16 because fp16 gives invalid images or videos.
  - With `num_batches_per_epoch = 1` and gradient accumulation 1, verify that the ratio is exactly 1.
  - Check base SDE quality with `noise_level = 0`.
  - Match the training batch size to the collection batch size (SD3 output depends on batch size).

  [P] — [flow_grpo README](https://github.com/yifan123/flow_grpo)
- **DanceGRPO.** Print ratio, reward and advantage per sample; lower `max_grad_norm` on reward collapse (issue #55); use FSDP; EMA 0.995–0.999. [P] — [DanceGRPO README](https://github.com/XueZeyue/DanceGRPO)
- **Flash-GRPO.** Without temporal gradient rectification, reward collapses between 300 and 400 GPU-hours. [S] — [arXiv](https://arxiv.org/abs/2605.15980)

#### Reward normalisation across multiple rewards
- **flow_grpo.** Per-prompt statistics tracking (`per_prompt_stat_tracking = True`); FLUX configs use `global_std = True`. Multi-reward is a weighted dict, for example {pickscore 0.5, ocr 0.2, aesthetic 0.3}. [P] — [README](https://github.com/yifan123/flow_grpo); [config](https://github.com/yifan123/flow_grpo/blob/main/config/grpo.py)
- **LingBot-Video.** Six rewards "with different scales and variances" are each normalized independently, then fused with weights ("decoupled per-reward normalization"). GDPO additionally applies batch-wise advantage normalization so advantage magnitude does not grow with the number of rewards. [S] — [LingBot-Video](https://arxiv.org/pdf/2607.07675) (GDPO described in the same search result)
- **LongCat-Video.** Four reward terms, all weight 1; the motion reward counters HPSv3's static bias. [S] — [hyper.ai](https://hyper.ai/en/papers/2510.22200)
- **DanceGRPO.** VideoAlign de-normalization constants, with `vq_coef` 1.0 and `mq_coef` 0.0 in the HunyuanVideo script; the TA reward was unstable and dropped. [P]/[S] — [README](https://github.com/XueZeyue/DanceGRPO); [arXiv HTML v4](https://arxiv.org/html/2505.07818v4)
- **Pref-GRPO.** Group-normalizing tightly clustered pointwise scores creates an "illusory advantage"; use pairwise win rates. [S] — [alphaXiv abstract](https://alphaxiv.org/abs/2508.20751)
- **Reward serving.** [P]
  - Flow-Factory: VLM judges via OpenAI-compatible APIs (`vllm_evaluate`, `rational_rewards_*`), custom and remote reward servers, async rewards with reward/optimization overlap. — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
  - verl-omni: multi-reward serving (HPSv3, GenRM-OCR, UnifiedReward, HTTP scorer) and asynchronous reward computation overlapping rollout. — [verl-omni](https://github.com/verl-project/verl-omni)

#### Mode collapse, diversity loss and reward hacking
- **Observed failure modes.**
  - Proxy score rises while gold score falls. [P] — [flow_grpo](https://github.com/yifan123/flow_grpo)
  - Drift to an "unnaturally dark style". [S] — [Pref-GRPO](https://alphaxiv.org/abs/2508.20751)
  - Static videos from an aesthetics-only reward. [S] — [LongCat](https://hyper.ai/en/papers/2510.22200)
  - Annotator bias toward low motion. [S] — [DenseDPO](https://arxiv.org/html/2506.03517v2)
  - "Over-stylization, or reduced diversity". [S] — [DDRL](https://arxiv.org/abs/2512.04332)
- **GARDO.** Multiplicative advantage shaping against mode collapse; reports that KL-free baselines suffer severe reward hacking. [2nd] — [alphaXiv overview 2512.24138](https://alphaxiv.org/overview/2512.24138v1)
- **DiverseGRPO** (arXiv 2512.21514): diversity-aware GRPO for image generation. [S] — [arXiv PDF](https://arxiv.org/pdf/2512.21514). **"Beyond the Dirac Delta"** (arXiv 2601.12401) addresses diversity collapse in reinforcement fine-tuning. [S] — [fugumt mirror](https://fugumt.com/fugumt/paper_check/2601.12401v1_enmode). **TMPO** (2605.10983) uses mode-covering trajectory distribution matching. [S] — [alphaXiv listing](https://www.alphaxiv.org/abs/2605.10983)
- **Entropy is not a reliable diversity signal.** A 2026 flow-RLHF paper reports diversity collapse while policy entropy stays constant, which it attributes to mode-seeking online RL. Mapping to arXiv 2605.12112 is from a search summary and uncertain. [S] — [arXiv PDF](https://arxiv.org/pdf/2605.12112)
- **Identity-reward hacking.** IPRO's multi-view scoring guards against it. [S] — [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Identity-Preserving_Image-to-Video_Generation_via_Reward-Guided_Optimization_CVPR_2026_paper.pdf). **Reference model from the SFT checkpoint** reduces hacking in camera-control GRPO. [S] — [arXiv 2512.02870](https://arxiv.org/pdf/2512.02870)

#### Wan 2.2 (A14B MoE) specifics
- **Expert routing in RL.** Miles' Wan2.2 recipe uses `boundary_ratio = 0.875`: training timesteps at or above 0.875 update the high-noise expert, the rest update the low-noise expert, and "which expert a training pair updates depends only on its timestep". [2nd] — [Miles doc](https://radixark.mintlify.app/diffusion/models/wan/wan2-2)
- **Expert roles.** The high-noise expert handles overall layout in early steps; the low-noise expert refines video details. [S] — [Wan2.2 model card](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)
- **Framework coverage.** [P]
  - Flow-Factory: Wan2.1-T2V-1.3B/14B, Wan2.2-TI2V-5B, Wan2.2-T2V-A14B, Wan2.1-I2V-14B (480P/720P), Wan2.2-I2V-A14B, Wan2.1-FLF2V-14B-720P. — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory)
  - verl-omni: Wan2.2 T2V with DanceGRPO (added Jun 2026). — [verl-omni](https://github.com/verl-project/verl-omni)

#### Frameworks and hardware (as of Nov 2026)
- **flow_grpo** (Flow-GRPO reference). SD3.5, FLUX, FLUX-Kontext, Qwen-Image(-Edit), Wan2.1, BAGEL; GRPO-Guard; CPS. SD3 configs target 32 GPUs, with 4- and 1-GPU variants. [P] — [flow_grpo](https://github.com/yifan123/flow_grpo)
- **DanceGRPO repo.** SD, FLUX, HunyuanVideo, SkyReels-I2V, Wan-2.1, Qwen-Image(-Edit). H800 requirements: 8 (SD, Wan-2.1, Qwen), 16 (FLUX), 16–32 (HunyuanVideo), 32 (SkyReels-I2V). [P] — [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- **MixGRPO repo.** FLUX.1-dev, 32 GPUs by default. [P] — [MixGRPO](https://github.com/Tencent-Hunyuan/MixGRPO)
- **DiffusionNFT repo.** SD3.5-M, 8 GPUs by default. [P] — [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)
- **Flow-Factory** (X-GenGroup; paper arXiv 2602.12529, Feb 2026; 746 stars).
  - Algorithms: SFT, offline DPO, online DPO, GRPO (Flow-/Dance-GRPO), DiffusionNFT, AWM, DGPO, GRPO-Guard, Flow-DPPO, CRD; distillation (DiffusionOPD, DMD2, TDM, TDM-R1).
  - Models: SD3.5, FLUX.1/2/3, Qwen-Image, Z-Image, BAGEL, Wan2.1/2.2, LTX-2/2.3, MiniMax H3.
  - Infrastructure: YAML and registry design; embedding caching; `torch.compile`; tensor and sequence parallelism; optional DeepSpeed; PyTorch ≥ 2.10. No VRAM figures given.

  [P] — [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory). A summary reports about 13% GPU-memory reduction and 1.74× training speed from caching and offloading. [2nd] — [arXiv HTML](https://arxiv.org/html/2602.12529)
- **verl-omni** (built on verl; Apache-2.0).
  - Releases: v0.1.0 (Qwen-Image recipes for FlowGRPO, Flow-DPPO, MixGRPO, GRPO-Guard, DiffusionNFT, DPO); v0.2.0 (Aug 2026; LTX2.3 FlowGRPO, MiniMax-H3); v0.3.0 (Oct 2026; fully async diffusion RL with TransferQueue and ReplayBuffer; v1 trainer as default).
  - Stack: vLLM-Omni rollout, FSDP2 / VeOmni trainers, USP/TP/DP parallelism, Ascend A2/A3/A5 NPU support.
  - About 25% higher end-to-end throughput than the diffusers-based flow_grpo on the reference Qwen-Image FlowGRPO setup.

  [P] — [verl-omni GitHub](https://github.com/verl-project/verl-omni); [v0.1.0 release](https://github.com/verl-project/verl-omni/releases/tag/v0.1.0)
- **Other infrastructure.**
  - Miles (radixark): Wan2.2-A14B Flow-GRPO recipe. [2nd] — [Miles doc](https://radixark.mintlify.app/diffusion/models/wan/wan2-2)
  - cosmos-rl (NVIDIA): DDRL code. [S] — [DDRL page](https://research.nvidia.com/labs/cosmos-lab/ddrl/)
  - Spotlight (Jun 2026; built on ROLL): runs seed exploration on idle spot GPUs, which cost 69–77% less, while the reserved cluster trains. It relies on stale-weight exploration preserving seed rankings and uses elastic sequence parallelism. Reaches the target score 4× faster with 1.4–6.4× lower total cost on Qwen-Image. [S] — [arXiv PDF](https://arxiv.org/pdf/2606.19004)

#### Compute-cost data points (sparse)
- **Image runs.** DanceGRPO FLUX: 200 iterations ≈ 12 h on 16 H800, about 192 GPU-hours. [P] — [DanceGRPO README](https://github.com/XueZeyue/DanceGRPO)
- **HunyuanVideo.** DanceGRPO trains on 32 H800 for 202 steps at 480×480×53 frames (wall-clock not stated). [P] — [script](https://github.com/XueZeyue/DanceGRPO/blob/main/scripts/finetune/finetune_hunyuan_grpo.sh)
- **14B video.** "Hundreds of GPU days per experiment" with standard GRPO. [S] — [Flash-GRPO](https://arxiv.org/abs/2605.15980)
- **DiffusionART vs baselines.** On HunyuanVideo-13B, 12.02× / 11.72× fewer GPU-hours than DiffusionNFT / Flow-GRPO-Fast. [S] — [hyper.ai](https://hyper.ai/en/papers/2610.12572). Unverified: 260.8 vs 883.3 GPU-hours, and 45.8 vs 220.3 GPU-hours for Wan2.1 I2V. [2nd] — [LinkedIn repost](https://mer.vin/linkedin/diffusionart-faster-more-stable-rl-fine-tuning-for-diffusion-models-linkedin/)
- **Whole-project scale.** DDRL used more than 1M H100 GPU-hours across all experiments. [S] — [arXiv](https://arxiv.org/abs/2512.04332)
- **Pref-GRPO.** 64 H20 GPUs. [2nd] — [Liner](https://liner.com/review/prefgrpo-pairwise-preference-rewardbased-grpo-for-stable-texttoimage-reinforcement-learning)

### Inferences

#### Candidate design for Prompt-to-Scene Creator (synthesis, not a sourced claim)
1. **Reward hygiene first.**
   - Treat each of the 7 judge dimensions as a separate reward.
   - Normalize each within its group (z-score, with a std floor), then do a weighted sum, then a batch renormalization (GDPO-style).
   - Use the judge in pairwise or win-rate mode (Pref-GRPO) or read its token probabilities, not integer Likert scores.
   - Add a dedicated motion reward model (VideoAlign-MQ-like; LongCat-style grayscale input against appearance shortcuts), an identity reward (ArcFace-like cosine with multi-view scoring), and a real-vs-CG photorealism classifier.
   - Keep a held-out slice of logged human pairs as the "gold" metric so over-optimization is detected (flow_grpo / GRPO-Guard).
2. **SFT** on curated live-action footage.
3. **Offline DPO.**
   - Fixed β.
   - Pair sources: logged human pairs; DenseDPO-style aligned pairs labeled per segment by the VLM judge; RealDPO/cIPO real-vs-generated pairs aimed at the anti-3D look.
4. **Online RL.**
   - Keep the Flow-GRPO code path, but use a single (or ≤ 3) stochastic step drawn from the high-noise half of the schedule, iso-temporal grouping, CPS noise about 0.8, and shared initial noise.
   - Group size 8–24; clip about 1e-3 for video (flow_grpo's Wan value), down to 1e-5 for 1–2-step variants; RatioNorm and per-step gradient reweighting.
   - KL β about 0.004 (flow_grpo Wan), or a DDRL forward-KL term on real footage.
   - LoRA on attention and FFN, learning rate about 1e-4.
   - Run DiffusionNFT or DiffusionART on the same rewards as an A/B, since both report large compute savings and are CFG-free or sampler-agnostic.
5. **Wan 2.2 MoE.** Training timesteps at or above 0.875 hit only the high-noise expert, which governs layout, composition and coarse motion. A window placed in the high-noise half therefore mostly trains that expert. Photoreal texture, the "never 3D" requirement, lives largely in the low-noise expert, where Flow-GRPO log-probs are least reliable. That argues for training the low-noise expert with a likelihood-free objective (NFT/AWM), DPO, or truncated reward backprop on a photorealism reward. This is an inference from flow_grpo's low-noise log-prob warning and Wan's expert roles.
6. **Multi-shot DiT.**
   - Use temporally partitioned, per-shot rewards (LongCat-Avatar 1.5) plus cross-shot identity rewards.
   - Use chunk- or shot-level credit (AR-CoPO-style forking at a shot boundary).
   - Evaluate on EntityBench or MSVBench.
7. **Refiner / SR stage RLHF** on photorealism and texture rewards (Seedance), and a separate GRPO-trained prompt rewriter (systematic post-train report).

### Gaps
- No primary GPU-hour figures for Flow-GRPO or MixGRPO on any video model, and no VRAM requirements for Wan2.2-A14B RL in any framework README.
- Miles' recipe hyperparameters were not retrieved.
- The DanceGRPO paper's exact multi-reward advantage aggregation was not verified; it is only referenced via GitHub issue #19.
- No published study tests per-expert RL strategies for Wan 2.2.

---

## Q6. Test-time alternatives (best-of-N, reward-guided sampling, verifier-based inference-time scaling) vs training

### Takeaway
Test-time search is a complement, not a replacement:
- Video-T1's Tree-of-Frames and multi-verifier ranking, and EvoSearch's evolutionary noise search on HunyuanVideo and Wan, keep improving with compute.
- Reward guidance (Flow-NRG) and VLM-driven segment repair (AESR, VQQA) fix individual outputs.

Their cost scales linearly per delivered clip, while RL is a one-off amortized cost. The same verifiers and reward models can serve both training rewards and test-time selection.

### Cited Findings
- **Video-T1** (Mar 2025; ICCV 2025). Treats test-time scaling as search over noise-to-video trajectories, presented as an alternative to retraining. Tree-of-Frames is cheaper than full-step denoising of every candidate, and best-of-N is used to validate scaling laws. Candidates are scored by a weighted mix of verifiers and the top candidate is picked by averaged rank, which is more robust than a single reward. [S] — [arXiv PDF](https://arxiv.org/pdf/2503.18942); [project page](https://liuff19.github.io/Video-T1). Complexity is claimed as O(N + T) vs O(TN) for random linear search. [2nd] — [summary](https://papers.lunadong.com/paper/16722)
- **EvoSearch** (Kuaishou / HKUST, May 2025; arXiv 2505.17618). Evolutionary test-time search, of which best-of-N is a special case. It keeps improving with more compute where baselines plateau, with gains on image and video models including HunyuanVideo and Wan. [2nd] — [hyper.ai](https://hyper.ai/fr/papers/2505.17618)
- **Flow-NRG.** Inference-time reward guidance applied directly to noisy videos, from the VideoAlign paper. [S] — [arXiv HTML v2](https://arxiv.org/html/2501.13918v2)
- **Other 2026 inference-time methods.**
  - "Test-Time Scaling for Video Diffusion Models via Diagnosis-Guided Candidate Recycling" (arXiv 2608.29322). [S] — [arXiv PDF](https://arxiv.org/pdf/2608.29322)
  - From the curated list [2nd] — [awesome list](https://github.com/chrisliu298/awesome-rm-for-video-generation):
    - LatSearch (2603.14526): latent-space search and pruning with reward estimates.
    - WMReward (2601.10553): physics verifier used purely at inference.
    - VIGOR (2603.16271): usable for both post-training and inference-time selection.
    - Euphonium (2602.04928): process rewards with stochastic flow guidance.
    - AESR (2608.20749): a VLM localizes bad segments and drives targeted frame-level repair.
- **VQQA** (arXiv 2603.12310). Agentic video evaluation and quality improvement. It notes that gradient-based or search methods such as Video-TTT and EvoSearch need white-box model access and cannot be used with commercial APIs. [S] — [arXiv PDF](https://arxiv.org/pdf/2603.12310)
- **DanceGRPO** exposes a `--bestofn 8` flag in its HunyuanVideo RL script. Its exact role (training-sample selection vs inference) was not verified. [P] — [script](https://github.com/XueZeyue/DanceGRPO/blob/main/scripts/finetune/finetune_hunyuan_grpo.sh)
- **InfLVG** applies GRPO to an inference-time context-selection policy for long, consistent video. [2nd] — [Pith](https://pith.science/paper/2505.17574)

### Inferences
- **Cost comparison.** Best-of-N multiplies generation cost by N and adds N judge calls for every shot delivered. For a story with dozens of shots per user, N = 4 means about 4× serving GPU per story. RL costs hundreds to thousands of GPU-hours once, per the Flash-GRPO and DiffusionART figures, and then shifts the whole output distribution at no extra serving cost.
- **Practical split for the app.**
  - RL (or DPO) to move the base distribution toward photoreal live action and storyboard adherence.
  - Small-N best-of-N (2–4) with the same VLM judge and identity reward for hero shots or failed QA checks.
  - Segment-level repair (AESR-style) or noise re-search for long multi-shot sequences, where errors in early shots propagate.
  - Because Wan 2.2 is open-weights, white-box search (EvoSearch, Video-T1) is usable.
- **Hacking audit.** Test-time selection with the same judge that is used as the RL reward compounds Goodhart effects. Keep at least one verifier for selection that is not used in training.

### Gaps
- No source gave a quantified dollar or GPU-hour comparison of test-time scaling vs RL training for video, or best-of-N gains specifically for photorealism or identity continuity.
- Video-T1 and EvoSearch quantitative gains were not retrieved; arxiv.org and hyper.ai full text were not read.
