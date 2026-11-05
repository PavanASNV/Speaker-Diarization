# Feedback-driven optimization of the LLM / prompt layer in generative-media pipelines (state as of November 2026)

*Compiled 2026-11-05 for "Prompt-to-Scene Creator" (Claude as director + prompt writer for Seedance 2.0 / Veo 3.1 / Kling 3.0 / Sora 2 / Wan 2.2). How the sources were gathered: WebFetch to arxiv.org, alphaxiv.org and dspy.ai failed (network policy), and the shared web-search budget ran out partway through. Findings therefore come from three places. (a) Search-result summaries; those taken from aggregator or review sites (themoonlight, liner, emergentmind, newx, hyper.ai, fugumt) are marked "secondary". (b) Primary GitHub READMEs and source code, read directly via `git clone` and marked "primary repo". The DSPy checkout is at commit 3f708b1, dated 2026-11-04. (c) A few canonical abstract-level facts marked "[recalled, not re-fetched]". These come from prior knowledge of the primary source, and the report writer may want to re-verify them.*

## 1. Automatic prompt optimization (APO) methods: evidence, sample efficiency when each evaluation is expensive, and handling of noisy rewards

### Takeaway
The 2025–2026 evidence on sample efficiency favors reflective optimizers that learn from rich textual feedback. GEPA reports needing 100–500 evaluations where GRPO needs 5,000–25,000+, and it beats MIPROv2 by more than 10 pp. ACE reports 75% fewer rollouts than GEPA in one setting. Label-free optimizers that work from pairwise comparisons (SPO, PDO) cost roughly 1–6% of what earlier methods cost. Classic population search (OPRO, PromptBreeder, EvoPrompt) needs far more evaluations. The robust ways to handle noise all separate two jobs: cheap minibatch screening proposes candidates, and a statistically principled step selects among them. That step can be minibatch-then-validate with Pareto retention (GEPA), a surrogate model with periodic full evaluation (MIPROv2), best-arm identification (ProTeGi, TRIPLE), or dueling bandits (PDO). None of these methods trusts a single noisy score.

### Cited Findings
**Classic methods (2022–2024; dated)**
- APE (Zhou et al., ICLR 2023) has an LLM propose instructions and selects them by score. Its prompts matched or beat human-written instructions on 24/24 Instruction-Induction tasks and 17/21 BIG-Bench tasks [recalled, not re-fetched] — [arXiv 2211.01910](https://arxiv.org/abs/2211.01910)
- OPRO (Yang et al., ICLR 2024) puts a meta-prompt containing (prompt, score) history in front of the LLM optimizer. Its prompts beat human-designed prompts by up to 8% on GSM8K and up to 50% on Big-Bench Hard [recalled, not re-fetched] — [arXiv 2309.03409](https://arxiv.org/abs/2309.03409)
- EvoPrompt (Guo et al., ICLR 2024) uses an LLM as the GA/DE crossover and mutation operator. It reports up to 25% gains on BBH over human prompts [recalled, not re-fetched] — [arXiv 2309.08532](https://arxiv.org/abs/2309.08532)
- ProTeGi (Pryzant et al., EMNLP 2023) computes "textual gradients" (LLM critiques of errors), edits prompts against them, and runs beam search, using bandit best-arm identification (UCB, successive rejects) to save evaluations. It improved the initial prompt by up to 31% [recalled, not re-fetched] — [arXiv 2305.03495](https://arxiv.org/abs/2305.03495)
- PromptBreeder (Fernando et al., DeepMind 2023) evolves task-prompts and also the mutation-prompts that rewrite them ("self-referential" self-improvement). It outperformed Chain-of-Thought and Plan-and-Solve on arithmetic and commonsense benchmarks [recalled, not re-fetched] — [arXiv 2309.16797](https://arxiv.org/abs/2309.16797)
- TextGrad ("automatic differentiation via text") was published in *Nature* in 2025 (primary repo) — [TextGrad repo](https://github.com/zou-group/textgrad); [Nature article](https://www.nature.com/articles/s41586-025-08661-4). The abstract reports GPT-4o zero-shot GPQA rising from 51% to 55% and a 20% relative gain on LeetCode-Hard [recalled, not re-fetched] — [arXiv 2406.07496](https://arxiv.org/abs/2406.07496)
- MIPROv2 (DSPy) works in three steps: (1) bootstrap few-shot demo candidates by keeping examples the program got right; (2) propose instructions from a dataset summary, a program summary, the demos, and a randomly sampled "tip" (e.g., "be creative"); (3) search instruction × demo combinations with Bayesian optimization. Each trial is evaluated on a minibatch, and the best-averaging configuration gets a full-validation evaluation every `minibatch_full_eval_steps` (primary repo) — [DSPy MIPROv2 docs](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md)
- MIPROv2 auto presets in the current source: light = 6 candidates / 100 validation examples; medium = 12 / 300; heavy = 18 / 1,000. The trial count is `max(2·num_vars·log2(n), 1.5·n)` (primary repo) — [mipro_optimizer_v2.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py). The MIPROv2 paper reports up to 13% accuracy gains over baseline optimizers with Llama-3-8B [recalled, not re-fetched] — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695)
- "Are LLMs Good Prompt Optimizers?" (Ma et al., Feb 2024) found that reflection-based optimizers struggle to identify the true causes of errors and lean on their own prior knowledge. They produced similar feedback whatever the error distribution, and performance was often comparable whether the optimizer saw real or pseudo errors. The authors also found that even sound reflections often failed to yield a good prompt in a single refinement step — [arXiv 2402.02101](https://arxiv.org/abs/2402.02101v1); [secondary review](https://themoonlight.io/review/are-large-language-models-good-prompt-optimizers)

**2025–2026 successors**
- GEPA (Agrawal et al.; ICLR 2026 oral) mutates every prompt in a system using natural-language reflection on new rollouts, and keeps a Pareto front of candidates rather than only the global best. On Qwen3-8B across six tasks it beats GRPO (run with 24,000 rollouts) by 6 pp on average and by up to 19 pp, with up to 35× fewer rollouts. The arXiv v1 text claimed a 10% average gain over four tasks and up to 20%, so the averages depend on the version. GEPA also beats MIPROv2 by more than 10 pp (e.g., +12 pp on AIME-2025) — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009493); [ICLR oral](https://www.iclr.cc/virtual/2026/oral/10009494); [arXiv](https://arxiv.org/abs/2507.19457)
- The GEPA README (maintainers' claims) states "100–500 evaluations vs. 5,000–25,000+ for GRPO" and lists "expensive rollouts" as a target use case ("GEPA needs 100–500 evals vs 10K+ for RL"). It also reports GPT-4.1 Mini going from 46.6% to 56.6% on AIME 2025 and "50+ production uses" (Shopify, Databricks, Dropbox, OpenAI, Pydantic, MLflow, Comet ML) (primary repo, self-reported) — [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- GEPA mechanics in DSPy:
  - Reflection uses a minibatch whose default size is 3 (`reflection_minibatch_size=3`). A new candidate that improves on that minibatch is then evaluated on the Pareto validation set.
  - The Pareto frontier is "the set of candidates which achieve the highest score on at least one evaluation instance". Parents are sampled in proportion to coverage.
  - The frontier type can be per-instance, per-objective, hybrid or cartesian. Metrics can return `objective_scores` plus textual `feedback`.
  - Reflection is skipped only when the whole minibatch scores perfectly. `dspy.Image` inputs reach the reflection LM as structured content.
  - The docs recommend a strong reflection LM (example: `gpt-5`, temperature 1.0, 32k tokens).
  - An "inference-time search" mode sets the validation set to the evaluation batch and returns the best outputs found per task (primary repo) — [GEPA overview](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md); [GEPA advanced](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/GEPA_Advanced.md); [gepa.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py)
- GEPA's `auto` budget in DSPy is computed as `V + 5n + N·35 + (⌊(N+1)/5⌋+1)·V` metric calls, where N = `max(4·num_components·log2(n), 1.5n)`, n is 6/12/18 for light/medium/heavy, and V is the validation-set size (primary repo) — [gepa.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py)
- GEPA's `optimize_anything` API (blog post of 2026-02-18; paper arXiv 2605.19633) generalizes it to any text artifact (code, configurations, agent architectures) given an evaluator. Adapters exist for DSPy, LangChain, MCP, RAG and generic system prompts (primary repo) — [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- ACE (Agentic Context Engineering; Zhang et al.; ICLR 2026) treats context as an evolving "playbook" maintained by Generator, Reflector and Curator roles. It identifies two failure modes of earlier optimizers: *brevity bias*, where prompts drift toward short generic text, and *context collapse*, where wholesale LLM rewriting erodes accumulated detail. Its fix is itemized "delta" updates with grow-and-refine — [ICLR 2026](https://iclr.cc/virtual/2026/10018667); [arXiv HTML](https://arxiv.org/html/2510.04618v1)
- The ACE README reports +10.6% on agent tasks (AppWorld) and +8.6% on finance. On offline AppWorld it cites −82.3% latency and −75.1% rollouts versus GEPA, and on online FiNER −91.5% latency and −83.6% token cost versus Dynamic Cheatsheet. The Curator "converts lessons into structured delta updates with helpful/harmful counters, using deterministic merging with de-duplication and pruning" (primary repo) — [ace-agent/ace](https://github.com/ace-agent/ace)
- SPO (Self-Supervised Prompt Optimization; Xiang et al., EMNLP 2025 Findings) needs no ground truth: an LLM judge compares outputs pairwise (output-vs-output) and an LLM optimizer revises the prompt. The paper reports 1.1–5.6% of the cost of state-of-the-art methods — [ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-emnlp.479/); [emergentmind (secondary)](https://www.emergentmind.com/papers/2502.06855). The README claims "17.8–90.9× higher cost efficiency" and "$0.15 per task optimization", uses "typically 3" QA pairs per iteration, and defaults to `max_rounds=10` (primary repo) — [FoundationAgents/SPO](https://github.com/FoundationAgents/SPO)
- PDO (Prompt Duel Optimizer; ACL 2026 Findings) treats label-free prompt selection as a dueling bandit with an LLM judge. Double Thompson Sampling picks the most informative prompt pairs, the target is a Copeland-optimal prompt, and top-performer-guided mutation generates new candidates. It beats label-free baselines under fixed judge budgets on BBH and MS MARCO — [ACL Findings 2026](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.490/); [arXiv v2](https://arxiv.org/html/2510.13907v2)
- Training-Free GRPO (Tencent Youtu, Oct 2025) moves "policy optimization" into context space. The LLM introspects on a group of rollouts to produce a semantic advantage, then updates an experience library with add/delete/modify/keep operations. On DeepSeek-V3.1-Terminus, AIME24 rose from 80.0% to 82.7% and AIME25 from 67.9% to 73.3%, using "a few dozen" training samples at about $18 — [HF papers](https://huggingface.co/papers/2510.08191)
- Other 2026 sample-efficiency work (search snippets; no numbers retrieved):
  - GFlowPO (Feb 2026) treats prompt search as posterior inference with GFlowNets and adds off-policy replay so past evaluations are reused — [arXiv 2602.03358](https://arxiv.org/pdf/2602.03358)
  - PIAST (ACL 2026) estimates in-context-example utility with Monte Carlo Shapley values, subsampling and a replay buffer. It concludes that "optimizing examples is the dominant lever" for fast, data-efficient prompt engineering.
  - Adaptive Prompt Structure Factorization (ACL 2026) splits prompts into components so credit assignment is clearer.
  - Sources for the last two: [ACL 2026 long.515](https://aclanthology.org/2026.acl-long.515/); [ACL 2026 long.537](https://aclanthology.org/2026.acl-long.537/). Which ID belongs to which paper was not confirmed.

### Inferences
- **Translate optimizer budgets into video cost before choosing one.** Using DSPy's formula, GEPA `auto="light"` on a single-component program costs about 420, 460 or 580 metric calls for V = 10, 20 or 50 validation items. Medium costs 740–940 and heavy 1,105–1,385 (my computation from the source). If each metric call is one judged video clip costing C, a light run costs roughly 420–580·C for each context. Running it per (target × style × mode) context is therefore impractical. For video, set `max_metric_calls` explicitly (roughly 60–150) with a small validation set of about 8–12 scenes and the default minibatch of 3. Screen most candidates on cheap keyframe/T2I proxies, and spend video generations only on finalists, in the spirit of HbBoPs-style multi-fidelity selection (see Q3).
- **Ranking for the expensive-evaluation regime.** This ranking is an inference from the reported budgets; no study compares the methods head-to-head with video in the loop.
  1. Context/playbook learners that update online from production traffic: ACE, Training-Free GRPO, ReasoningBank (Q4). They need no dedicated evaluation budget.
  2. GEPA or SPO/PDO-style reflective or pairwise optimizers with minibatches of about 3.
  3. MIPROv2. Its defaults assume validation sets of 100–1,000.
  4. OPRO, EvoPrompt and PromptBreeder. These are population searches that score every candidate on a dev set.
- **The app's current design is OPTS + GEPA/ACE in all but name.** Thompson-sampled strategies plus Claude reflection that rewrites or retires them correspond to OPTS (Thompson sampling over strategies, see Q3) plus GEPA/ACE (reflection proposes edits). The evidence suggests four upgrades:
  1. Feed reflection structured, per-dimension VLM critiques and failure traces, which GEPA calls "actionable side information", not only scalar stars.
  2. Make reflection edit strategies at bullet level, with helpful/harmful counters per bullet (ACE), rather than rewriting whole strategies. This avoids context collapse and brevity bias.
  3. Retain any strategy that is best for some sub-context (Pareto retention), instead of retiring it on its pooled mean.
  4. Keep selection and retirement statistical rather than LLM-judged, given Ma et al.'s finding that LLM reflections are biased toward the model's priors.
- **Handle self-referential mutation cautiously.** Evolving the reflection meta-prompt itself, PromptBreeder-style, has the weakest cost-efficiency evidence. GEPA's results suggest a fixed, well-designed reflection prompt fed with rich feedback is enough. If the meta-prompt is evolved at all, do it offline against logged data, not live.

### Gaps
- No study was found that compares APO methods when each evaluation is a video generation. All sample-efficiency figures come from text, math and agent tasks.
- GEPA's per-benchmark rollout tables could not be retrieved because alphaxiv was blocked. The README's "100–500" figure is the maintainers' summary.
- The evaluation budgets of OPRO, PromptBreeder and EvoPrompt (population sizes, dev-set sizes) were not retrieved.
- Numbers for PIAST, APSF and GFlowPO were not retrieved, nor details of DSPy's SIMBA optimizer.

## 2. Prompt enhancers and rewriters for text-to-video and text-to-image trained with feedback, and what vendors disclose about their own rewriting

### Takeaway
Every major video stack rewrites user prompts toward the caption distribution it was trained on:
- **Veo:** a Gemini rewriter, `enhancePrompt`, is on by default.
- **HunyuanVideo 1.5:** a Qwen3-235B rewriter is on by default.
- **Wan 2.2:** Qwen prompt extension is recommended.
- **Seedance 1.0:** a fine-tuned Qwen2.5-14B "PE" module, per secondary sources.
- **Sora:** GPT recaptioning [recalled].

Rewriters trained with feedback from image or video rewards (Prompt-A-Video, VPO, RAPO, RePrompt, PromptEnhancer) report consistent gains. Optimizing for aesthetics alone can erode prompt adherence, though: Promptist lowered consistency by 3–4%, and HunyuanVideo's "Master mode" can lose semantic details.

### Cited Findings
**Vendor and production rewriters**
- **Veo (Vertex AI).** `enhancePrompt` is a boolean that defaults to true and uses Gemini to rewrite the prompt. The rewriter can add scene description, camera movement, dialogue transcription and sound effects. The rewritten prompt is returned in the response only when the original has fewer than 30 words. The docs warn that turning it off may reduce video quality and prompt adherence. The English page's examples use `veo-2.0-generate-001`, while localized pages list Veo 3 options, so the default for Veo 3.x is unconfirmed — [Vertex AI: turn off Veo prompt rewriter](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/video/turn-the-prompt-rewriter-off)
- **HunyuanVideo (v1).** Tencent fine-tuned Hunyuan-Large as a rewrite model "to adapt the original user prompt to model-preferred prompt". It has two modes. Normal mode is for intent comprehension. Master mode enhances "composition, lighting, and camera movement… However, this emphasis may occasionally result in the loss of some semantic details." The weights are released (primary repo) — [Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo); [HF PromptRewrite](https://huggingface.co/tencent/HunyuanVideo-PromptRewrite)
- **HunyuanVideo 1.5 (primary repo):**
  - Rewriting is on by default (`--rewrite true`), and disabling it "may result in lower quality video generation".
  - T2V rewriting uses Qwen3-235B-A22B-Thinking-2507 and I2V rewriting uses Qwen3-VL-235B-A22B-Instruct, both served via vLLM endpoints.
  - The rewrite system prompt ("Hunyuan Video Rewriter 1.5") demands objective, structured, detailed prompts ordered from general to specific:
    1. Subject and scene introduction.
    2. Subject details.
    3. A chronological action sequence with explicit connectives ("Initially… Then… Next… As… Finally").
    4. Environment and background.
    5. A closing technical and style summary covering camera movement, shot type, atmosphere and lighting.
  - The README's rewritten examples typically end with "The overall video presents a cinematic photography realistic style."
  - Sources — [HunyuanVideo-1.5 README](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5); [t2v_prompt.py](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/hyvideo/utils/rewrite/t2v_prompt.py)
- **Wan 2.2 (primary repo).** The README says "Extending the prompts can effectively enrich the details in the generated videos, further enhancing the video quality. Therefore, we recommend enabling prompt extension." Extension runs either through the DashScope API (`qwen-plus` for T2V, `qwen-vl-max` for I2V) or a local model (Qwen2.5-14B/7B/3B-Instruct for T2V, Qwen2.5-VL-7B/3B for I2V), and larger rewriters give better extensions. Its efficiency benchmarks were run without `--use_prompt_extend` — [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- **Seedance 1.0 (ByteDance, June 2025).**
  - Secondary summaries describe a prompt-engineering (PE) module: a Qwen2.5-14B fine-tuned with SFT and then RL (DPO) to convert user prompts into the dense caption format the DiT was trained on.
  - The official blog confirms only a purpose-trained precise caption model and a video-tailored RLHF stage, so the DPO-for-PE detail is unconfirmed.
  - Secondary sources also say the RLHF stage uses foundational, motion and aesthetic reward models.
  - Sources — [Seed blog](https://seed.bytedance.com/en/blog/tech-report-of-seedance-1-0-is-now-publicly-available); [secondary review](https://www.themoonlight.io/review/seedance-10-exploring-the-boundaries-of-video-generation-models); [arXiv HTML](https://arxiv.org/html/2506.09113v2)
- **Sora (Feb 2024 report).** "Similar to DALL·E 3, we also leverage GPT to turn short user prompts into longer detailed captions that are sent to the video model" [recalled, not re-fetched] — [OpenAI, Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/). DALL·E 3 introduced this kind of GPT-4 caption "upsampling" at inference [recalled, not re-fetched] — [DALL·E 3 paper](https://cdn.openai.com/papers/dall-e-3.pdf)

**Research rewriters and loops trained or driven by feedback**
- **Promptist** (Microsoft, NeurIPS 2023) trained an LM rewriter with SFT followed by PPO, rewarding CLIP relevance plus an aesthetic score on Stable Diffusion [recalled, not re-fetched] — [arXiv 2212.09611](https://arxiv.org/abs/2212.09611). OPT2I later measured that Promptist *decreases* T2I consistency by 3.56% on SD-1.4 and 3.29% on LDM-2.1, while OPT2I with Llama-2 raises it by 14.16% and 11.21% — [OPT2I (HF papers)](https://huggingface.co/papers/2403.17804)
- **OPT2I** (Meta; TMLR 2024, ICLR 2025 poster) runs a training-free, OPRO-style loop in which an LLM iteratively revises the prompt against a consistency score from the generated images. It improves DSG consistency by up to 24.9% while preserving FID, evaluated on MSCOCO and PartiPrompts. Optimizing against the finer DSG metric beat optimizing against decomposed CLIPScore — [HF papers](https://huggingface.co/papers/2403.17804); [TMLR](https://mlanthology.org/tmlr/2024/manas2024tmlr-improving)
- **Prompt-A-Video** (ICCV 2025) works in two stages:
  1. Reward-guided prompt evolution builds SFT data. GPT-4o or GLM-4 rewrites prompts, a T2V model (Open-Sora 1.2 or CogVideoX) generates videos, and VideoScore, MPS and an aesthetic predictor score them. A Llama-3-Instruct LoRA booster is then fine-tuned on the result.
  2. DPO on pairs built from the multi-dimensional rewards.

  The authors explicitly criticize generic refiners as "Model-Unaware" (primary repo + paper) — [repo](https://github.com/jiyt17/Prompt-A-Video); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Prompt-A-Video_Prompt_Your_Video_Diffusion_Model_via_Preference-Aligned_LLM_ICCV_2025_paper.html)
- **VPO** (Tsinghua/Zhipu, ICCV 2025) optimizes for harmlessness, accuracy and helpfulness. It runs SFT on curated data and then preference learning from both text-level and video-level feedback. On CogVideoX the reported human evaluation gives a "37.5% win rate" over original prompts and 14% over the official prompt optimizer. Whether these are win rates or net wins is unclear in the summary (secondary). It generalizes across video models and can be combined with RLHF — [HF papers](https://huggingface.co/papers/2503.20491); [ICCV poster](https://iccv.thecvf.com/virtual/2025/poster/694). Its released template tells the rewriter to "Keep the scene feasible and well-suited for a brief duration, avoiding unnecessary complexity or unrealistic elements not mentioned in the query" (primary repo) — [thu-coai/VPO](https://github.com/thu-coai/VPO)
- **RAPO** (CVPR 2025) runs two branches and keeps the better result. One augments the prompt with modifiers retrieved from a learned relation graph and then refactors it with a fine-tuned LLM. The other has a pre-trained LLM rewrite it under fixed instructions. A selection module picks the better prompt. It was applied to LaVie and Latte and tested on VBench, EvalCrafter and T2V-CompBench — [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_The_Devil_is_in_the_Prompts_Retrieval-Augmented_Prompt_Optimization_for_CVPR_2025_paper.html)
- **RePrompt** (Microsoft; ICLR 2026) uses RL to train a reasoning reprompter on image-level rewards for human preference, semantic alignment and composition, without human annotation. On GenEval it reports relative gains over Qwen2.5-3B-enhanced baselines of +77.1% (FLUX), +78.8% (SD3) and +122.2% (PixArt-Σ) — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10010403); [arXiv](https://arxiv.org/abs/2505.17540)
- **PromptEnhancer** (Tencent Hunyuan, Sept 2025) is a chain-of-thought rewriter trained with SFT and then GRPO. Its reward comes from "AlignEvaluator", which scores the image against 24 key points in 6 categories rather than giving one holistic score. The T2I model (HunyuanImage 2.1) stays frozen and the method is model-agnostic — [arXiv](https://arxiv.org/abs/2509.04545); [hyper.ai summary (secondary)](https://hyper.ai/fr/papers/2509.04545)

### Inferences
- **The app is optimizing a pre-prompt.** Veo rewrites by default and the other stacks recommend or ship rewriters, so the app's prompt is not the final conditioning text. Vendor rewriting adds noise and can wash out differences between strategies. Log whether rewriting was on for every generation. Where a toggle exists (Veo `enhancePrompt`), treat "rewriter on/off" as a bandit arm, not a fixed choice. For short prompts, capture Veo's returned rewritten prompt as training signal.
- **Public vendor rewriter prompts are data, not heuristics.** The HunyuanVideo 1.5 rewrite system prompt is effectively a published map of a DiT's caption distribution. Claude can ingest such material as context to *propose* initial per-target strategies. That keeps every creative decision model-driven while starting closer to each model's preferred style. Prompt-A-Video's "Model-Unaware" critique supports keeping strategies per target.
- **Photorealism is a separate objective to enforce.** HunyuanVideo 1.5's rewriter picks a style per content: realistic prompts get "cinematic photography realistic style", but an ink-painting prompt got "dynamic Chinese ink wash animation style". Vendor rewriters can therefore pull fantastical or surreal content toward animation looks. The app should state live-action photographic intent explicitly in the final prompt. It should also track "photographic live action vs CGI/3D/animated look" as its own judge objective (a Pareto objective, see Q6), because OPT2I and HunyuanVideo's Master mode both show that optimizing one quality axis can cost another.
- **Rewriter training is a later-stage option.** Trained rewriters (VPO, Prompt-A-Video, RePrompt, PromptEnhancer) assume large generation budgets and open or controllable models. For the app this only becomes realistic once enough logged preference data exists to distill a small target-specific rewriter (see Q5). Until then, Claude plus learned context is the appropriate rewriter.

### Gaps
- No public details were retrieved on prompt rewriting in Kling 3.0, Seedance 2.0, Sora 2 or Veo 3.1, including whether Veo 3.x lets callers disable `enhancePrompt` or how it rewrites.
- Headline numbers for Prompt-A-Video, RAPO and PromptEnhancer were not retrieved, and VPO's win-rate definition is unclear.
- No figures were retrieved on the data or compute behind these rewriters (number of generated videos, rater counts).
- PAE (dynamic prompt editing, CVPR 2024), PhyT2V (LLM-guided iterative T2V refinement) and Wan's own prompt-extension system prompts were not retrieved.

## 3. Bandits and Bayesian optimization for choosing prompts, strategies, models/providers and parameters under cost

### Takeaway
Thompson sampling over LLM-generated strategies is directly supported: in OPTS, Thompson-sampling selection beat the other selectors and improved EvoPrompt by up to 50%. The most valuable additions for this app are:
1. Hierarchical or empirical-Bayes priors that pool evidence across target × style × mode contexts (HierTS, ebmTS).
2. Budgeted best-arm identification and multi-fidelity evaluation for promotion decisions (TRIPLE, HbBoPs).
3. Dueling bandits for pairwise feedback (APOHF, PDO's Double TS).
4. Contextual, cost-aware routing of each shot to a provider (PAK-UCB, PILOT).
5. LLM priors as warm starts (LLAMBO).

No retrieved source addresses non-stationarity for prompt bandits specifically, so the handling proposed below is inference.

### Cited Findings
- **OPTS** (ACL 2025 Findings) gives each prompt-design strategy (CoT, role prompting, etc.) an arm and adds an "inaction" arm, each with a Beta posterior. The reward is whether the new prompt beats the best score so far. It was integrated into EvoPrompt and tested with Llama-3-8B-Instruct and GPT-4o mini on BBH. Thompson sampling was the best of the three selectors compared (TS, uniform, and APET's implicit LLM choice), improving EvoPrompt by up to 50% with GPT-4o mini. The authors note they did not test contextual bandits and that the reward formulation could be improved — [ACL Anthology](https://aclanthology.org/2025.findings-acl.1070); [shiralab/OPTS (primary repo)](https://github.com/shiralab/OPTS)
- **TRIPLE** (NeurIPS 2024) frames prompt selection under a fixed budget as fixed-budget best-arm identification, using Sequential Halving or Continuously Reject. It adds embedding-based variants (CLST, GSE) for large pools and extends to selecting few-shot examples — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b46bc1449205888e1883f692aff1a252-Abstract.html)
- **HbBoPs** (Amazon; ICML 2025) uses a structure-aware deep-kernel Gaussian process, embedding instructions and exemplars separately, with Hyperband as a multi-fidelity scheduler over how many validation instances each prompt gets. It beat MIPROv2, EASE and TRIPLE-SH/GSE across 10 benchmarks and 3 LLMs, on both performance and efficiency — [PMLR v267](https://proceedings.mlr.press/v267/schneider25b.html)
- **APOHF** (ICML 2024 workshop, oral) optimizes prompts from pairwise human preferences only. A neural-network-based, dueling-bandit-inspired rule chooses which pair of prompts to show the user. Applications include text-to-image prompt optimization, and baselines include random search, linear dueling bandits and Double TS. It finds good prompts "using a small number of preference feedback instances" — [arXiv HTML](https://arxiv.org/html/2405.17346v1); [ICML workshop](https://icml.cc/virtual/2024/37753); [xqlin98/APOHF](https://github.com/xqlin98/APOHF)
- **PDO** uses Double Thompson Sampling over prompt pairs judged by an LLM, targeting a Copeland winner under a fixed judge-query budget (see Q1) — [arXiv v2](https://arxiv.org/html/2510.13907v2)
- **PAK-UCB** (ICML 2025) treats the prompt as context and the candidate generators as arms. It predicts each model's per-prompt score with kernel ridge regression, so it learns that different generators win on different prompt types instead of using one global ranking. An RFF-UCB variant speeds it up with an Õ(√T) regret bound. Experiments cover T2I and image captioning. A secondary summary cites +12.7–23.4% selection accuracy and an 18.3–31.6% cut in query cost (unverified) — [ICML 2025](https://icml.cc/virtual/2025/poster/44140); [arXiv HTML](https://arxiv.org/html/2410.13287v2); [code](https://github.com/yannxiaoyanhu/dgm-online-select); [secondary numbers](https://newx.sg/paper/detail/e0fc0f95-f822-11f0-b84c-00163e10baa7)
- **Cost-aware LLM routing as a bandit:**
  - **PILOT** (EMNLP 2025 Findings) is a contextual bandit with a multi-choice-knapsack budget policy. Per-model cost-eligibility thresholds come from estimated reward, reward-to-cost ratio and current budget use, and the best-expected-reward model is chosen among the eligible ones — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.1301); [arXiv HTML](https://arxiv.org/html/2508.21141v1)
  - **Poon et al.** (2025; AAAI) use a LinUCB-based multi-LLM selector with budget-aware and position-aware extensions for multi-step query refinement — [arXiv 2506.17670](https://web3.arxiv.org/abs/2506.17670); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39672/43633)
- **Sharing statistical strength across contexts.**
  - **Hierarchical Bayesian Bandits** (HierTS; Hong et al., AISTATS 2022) unify meta-, multi-task and federated bandits in one hierarchical model. Their regret bounds improve as the shared prior becomes more informative, and experiments show the hierarchy shares knowledge across tasks — [PMLR v151](https://proceedings.mlr.press/v151/hong22c.html)
  - **Empirical Bayesian Multi-Bandit Learning** (ebmTS/ebmUCB, Oct 2025) estimates the prior covariance across bandits from data instead of fixing it — [arXiv 2510.26284](https://arxiv.org/pdf/2510.26284v2)
- **LLM-guided Bayesian optimization.** LLAMBO (ICLR 2024) does zero-shot warm-starting, uses the LLM as a surrogate and as a candidate sampler, all in context with no fine-tuning. Its gains are strongest early in a search, when observations are sparse — [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/84b8d9fcb4e262fcd429544697e1e720-Abstract-Conference.html)
- **Lightweight bandit-like statistics in memory systems.** ACE keeps helpful/harmful counters on every playbook bullet (primary repo) — [ace-agent/ace](https://github.com/ace-agent/ace). ExpeL refines its insights by adding, editing and voting on their importance — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29936)

### Inferences
- **Recommended selection model: hierarchical Beta/logit model with backoff.**
  - Treat each (strategy s, context c) as a cell. A minimal form uses a logit-normal effect, `logit p(s,c) = μ_s + b(s,target) + b(s,style) + b(s,mode)`, fitted by empirical Bayes in the style of HierTS/ebmTS.
  - A simpler "Beta backoff" also works: initialize each cell's prior from its parent cell's posterior (global → target → target × style → target × style × mode), capped at an effective sample size of about 5–10 pseudo-observations.
  - Thompson-sample from the cell posterior.
  - Pooling is essential. The paired-test calculation in Q6 shows roughly 190 observations are needed to separate 60% from 50%. A single context with sparse traffic cannot learn on its own.
- **Use Claude's opinion as a weak prior (LLAMBO-style).** When reflection proposes a new or edited strategy, ask Claude for an a priori success estimate and convert it into a small Beta prior, e.g. 2–4 pseudo-counts, centred on the parent's posterior mean shifted by Claude's predicted delta. Data then quickly overrides it. This keeps the system model-driven while making cold starts efficient.
- **Keep a control arm** (the OPTS "inaction" arm): a frozen baseline strategy (or no strategy) in every context. Strategies that hurt are then detectable as "worse than control" rather than only "worse than the best". It also helps detect drift when vendors update models.
- **Define rewards robustly.** Ratings and judge scores need converting before they feed a Beta posterior:
  - Convert star ratings to within-user relative outcomes, e.g. above that user's median, or the pairwise winner among takes of the same shot. Raw stars vary by rater.
  - Fractional Beta updates (α += r, β += 1−r) for judge scores r in [0,1] are a pragmatic option.
  - When several takes exist for one shot, prefer pairwise outcomes and a dueling-bandit treatment (Double TS / Copeland, as in PDO and APOHF).
- **Make the bandit cost-aware.** Sample θ_k and choose `argmax θ_k − λ·cost_k`, or apply PILOT-style eligibility thresholds based on reward-to-cost and remaining budget. Here cost_k is the provider's per-second price × duration plus expected regenerations.
- **Route each shot to a provider with a contextual bandit.** A PAK-UCB/RFF-UCB-style contextual bandit over shot-description embeddings fits Seedance/Veo/Kling/Sora/Wan routing. The kernel lets evidence from similar shots (night exteriors, crowds, close-up faces) transfer across projects.
- **Handle non-stationarity explicitly.** Vendor version changes (Veo 3 → 3.1, Kling 2.x → 3.0, Seedance 1.x → 2.0) and silent server-side rewriter changes invalidate old evidence. Three mitigations (standard practice, not sourced here):
  1. Key contexts by model version, and inherit priors from the previous version's posterior with deflated weight.
  2. Exponentially discount Beta counts (multiply α and β by γ, e.g. 0.98–0.995, per update or per day).
  3. Trigger re-exploration when the control arm's rate shifts.
- **Promote and retire with budgeted best-arm logic.** Use TRIPLE/Sequential-Halving logic and multi-fidelity screening (HbBoPs): keyframes first, video for finalists. Retire a strategy only when it is (a) worse than control with high posterior probability, *and* (b) not on the Pareto frontier of any sub-context or objective (GEPA).

### Gaps
- No retrieved study compares TS, UCB and contextual bandits specifically for prompt or strategy selection. The classic empirical evidence that TS is robust to delayed and batched feedback (Chapelle & Li, 2011) was not re-verified this session.
- No source was found on prompt bandits under generator version changes, nor on how quickly prompt-strategy rankings decay after a vendor update.
- Neural bandits (NeuralUCB/NeuralTS) for prompts were not retrieved beyond APOHF's use of a neural network.
- PILOT's experimental numbers were not verified. MixLLM (said to be a linear contextual bandit over weighted accuracy/cost), LLM Bandit (Feb 2025) and BaRP (Oct 2025) appeared only as names in search snippets, with no primary page retrieved.

## 4. Agentic self-improvement in creative pipelines: memory, critique-and-revise with VLM feedback, exemplar retrieval, and evidence on human-rated quality

### Takeaway
Test-time critique-and-revise with multimodal judges does raise human-rated video quality when the generator is strong. VISTA on Veo 3 reports up to a 60% pairwise win rate against baselines and 66.4–68% human preference. Multi-agent story planners (MovieAgent, FilmAgent, VideoGen-of-Thought, Anim-Director) report better consistency and coherence in small, author-run studies. The best evidence for *persistent* learning across episodes comes from agent-memory methods: ExpeL, ReasoningBank, ACE and Training-Free GRPO. Critique without grounded external signals is unreliable: Ma et al. 2024 show reflection biased by the model's priors, Huang et al. [recalled] show intrinsic self-correction failing, and Q6 shows weak VLM–human agreement.

### Cited Findings
- **VISTA** (Google + NUS; CVPR 2026) is a black-box, test-time prompt optimizer around Veo 3:
  - It decomposes the user idea into a structured temporal (multi-scene) plan.
  - It picks the best candidate video through a pairwise tournament judged by a multimodal LLM.
  - A multi-agent critic panel scores the videos on visual, audio and context dimensions.
  - A reasoning agent rewrites the prompt for the next round.
  - Results: up to 60% pairwise win rate against state-of-the-art baselines. Human raters preferred VISTA in 66.4% of comparisons per the arXiv listing, 68% per Google Research. A secondary summary says it also improved Veo 2.
  - Sources — [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Long_VISTA_A_Test-Time_Self-Improving_Video_Generation_Agent_CVPR_2026_paper.html); [Google Research](https://research.google/pubs/vista-towards-test-time-self-improving-video-generation-agent/); [arXiv](https://arxiv.org/abs/2510.15831)
- **MovieAgent** (NUS Show Lab, Mar 2025) is training-free and takes a script plus a character bank. It does hierarchical chain-of-thought planning across director, screenwriter, storyboard-artist and location-manager agents, covering scenes, camera settings and cinematography (primary repo). Per secondary reviews, human evaluators rated it highest on script faithfulness and character consistency — [showlab/MovieAgent](https://github.com/showlab/MovieAgent); [arXiv HTML](https://arxiv.org/html/2503.07314v1)
- **FilmAgent** (SIGGRAPH Asia 2024) is a multi-agent film-production system (director, screenwriter, actors, cinematographer) set in 3D virtual spaces, *not* generative photoreal video. Its agents refine work iteratively: in Critique-Correct-Verify, an action agent drafts and a critique agent reviews and suggests fixes. Human evaluation over 15 ideas and 4 aspects gave an average of 3.98/5, beating all baselines. The GPT-4o-based multi-agent system also outperformed single-agent o1. Manual set construction is its main limitation — [HF papers](https://huggingface.co/papers/2501.12909.md); [arXiv HTML](https://arxiv.org/html/2501.12909v1); [repo](https://github.com/HITsz-TMG/FilmAgent)
- **Anim-Director** (ACM 2024, DOI 10.1145/3680528.3687688; arXiv Aug 2024) has GPT-4(V) act as director over Midjourney and Pika. Its self-reflection step picks the best of about 4 candidate images per scene and re-generates until validation passes; it also chooses among video candidates. Secondary sources say it was evaluated on 100 TinyStories narratives, with video quality 0.74, subject consistency 0.86 and background consistency 0.93 — [secondary review](https://themoonlight.io/review/anim-director-a-large-multimodal-model-powered-agent-for-controllable-animation-video-generation); [secondary (KO)](https://liner.com/ko/review/animdirector-large-multimodal-model-powered-agent-for-controllable-animation-video)
- **VideoGen-of-Thought** is a training-free pipeline (script → keyframes → shot videos → cross-shot smoothing) with identity-preserving embeddings. A later version reports +20.4% within-shot face consistency and +17.4% style consistency over baselines, and a user study preferred its outputs (secondary) — [HF papers](https://huggingface.co/papers/2412.02259); [v2 listing (secondary)](https://fugumt.com/fugumt/paper_check/2503.15138v2_enmode)
- **VideoAgent** (Oct 2024; robotics) iteratively refines generated video *plans* with pretrained-VLM feedback ("self-conditioning consistency") and real execution feedback. It reduced hallucination and raised manipulation success in MetaWorld and iTHOR. This is evidence that VLM feedback can drive refinement, though in a different domain — [arXiv HTML](https://arxiv.org/html/2410.10076v2)
- **2026 agentic video successors** (search snippets only):
  - **Co-Director** (arXiv 2604.24842) validates its judge against humans with Krippendorff's α and Cohen's κ, treating human–human agreement as the ceiling — [arXiv](https://arxiv.org/pdf/2604.24842)
  - **STAGE** (2512.12372) is storyboard-anchored multi-shot generation — [arXiv](https://arxiv.org/pdf/2512.12372)
  - **SPIRAL** (2603.08403) uses reflective planning agents trained with SFT and then DPO — [arXiv](https://arxiv.org/pdf/2603.08403)
  - **DramaDirector** (2606.24107) trains a storyboard planner with GRPO — [arXiv](https://arxiv.org/pdf/2606.24107)
- **ExpeL** (AAAI 2024) learns without weight updates. It recalls successful trajectories by task similarity and extracts natural-language insights from successes and failures, refining them by adding, editing and voting on importance. On HotpotQA it scored 40% versus Reflexion's 39%. It also beat Reflexion on ALFWorld; the summary gives "54% vs. 59%" with ambiguous ordering, so check the paper's table. Reflexion was still ahead on WebShop. Insights transferred from HotpotQA to FEVER at 70%, and they stay human-inspectable and editable — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29936); [project](https://andrewzh112.github.io/expel)
- **ReasoningBank** (Google/UIUC; ICLR 2026) distills general reasoning strategies from both successful and failed self-judged trajectories, without ground-truth labels. Memory-aware test-time scaling (MaTTS) runs many trajectories per task and uses them as contrastive signal to build better memories. A secondary summary cites up to +34.2% relative success and 16.0% fewer steps on web-browsing and software-engineering benchmarks. The code supports Gemini 2.5 and Claude 3.7 Sonnet on Vertex AI — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/980ea04d23d1f6908964eba2a74afe45-Abstract-Conference.html); [emergentmind (secondary)](https://www.emergentmind.com/papers/2509.25140); [repo](https://github.com/google-research/reasoning-bank)
- **ACE** and **Training-Free GRPO** are context-space learners (details in Q1). ACE learned from execution feedback without labels and matched the top production agent on AppWorld using a smaller open model — [ICLR 2026](https://iclr.cc/virtual/2026/10018667); [ace-agent/ace](https://github.com/ace-agent/ace)
- **Classic reflective loops** [recalled, not re-fetched]:
  - Reflexion (NeurIPS 2023) reached 91% pass@1 on HumanEval versus 80% for GPT-4 by storing verbal self-reflections in episodic memory — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)
  - Huang et al. (ICLR 2024) found that without external feedback, LLMs struggle to self-correct reasoning and sometimes get worse — [arXiv 2310.01798](https://arxiv.org/abs/2310.01798)
- **Exemplar retrieval.** Retrieving semantically similar in-context examples beats random selection (KATE) [recalled, not re-fetched] — [arXiv 2101.06804](https://arxiv.org/abs/2101.06804). Many-shot ICL with hundreds of examples gives further gains over few-shot [recalled, not re-fetched] — [arXiv 2404.11018](https://arxiv.org/abs/2404.11018). PIAST (ACL 2026) concludes that optimizing examples is the dominant lever for data-efficient prompt engineering (search snippet; Anthology ID unconfirmed, either long.515 or long.537) — [ACL 2026 long.515](https://aclanthology.org/2026.acl-long.515/); [ACL 2026 long.537](https://aclanthology.org/2026.acl-long.537/)

### Inferences
- **Run two coupled loops.**
  - *Inner loop* (test time, for hero shots or high-value scenes): VISTA-style N candidates → order-balanced pairwise tournament → per-dimension critique → prompt rewrite. Cap it at about 2–3 rounds because every round costs generations.
  - *Outer loop* (across projects): ACE/ReasoningBank-style playbooks and memories, updated by bullet-level deltas with helpful/harmful counters.
  - Feed the inner loop's tournament outcomes into the outer loop as *contrastive* evidence: same scene, different prompts, and which won and why. MaTTS shows this compounds, and it turns test-time spend into learning signal.
- **Ground every critique.** Ma et al. and Huang et al. imply reflections must cite observable evidence: specific frames, per-dimension checklist failures (PromptEnhancer-style key points), user edits and ratings. "General impressions" are not enough. Retire or rewrite a strategy only when the statistical layer (Q3) agrees.
- **Upgrade the exemplar memory.**
  1. Retrieve by scene-embedding similarity plus diversity (MMR), not top score alone.
  2. Tag every exemplar with target model version, mode (keyframe/video) and the judge's rationale. Expire or down-weight exemplars from superseded model versions.
  3. Store contrastive pairs (winning and losing prompt for the same scene).
  4. Treat exemplar sets as arms: TRIPLE's few-shot extension, PIAST-style utility estimates.
- **Transfer planning findings, not visual ones.** The human-quality evidence for agentic video pipelines comes mostly from the authors' own small studies (FilmAgent: 15 ideas). FilmAgent works in 3D virtual sets and Anim-Director targets animation, so their planning and critique patterns transfer, but their visual findings do not carry over to photoreal live action.

### Gaps
- No independent replications of agentic video-pipeline gains were found, and rater counts and inter-rater agreement were not retrieved.
- VISTA's per-prompt generation budget (rounds × candidates) and cost were not retrieved.
- Mora, DreamFactory, StoryAgent, VideoDirectorGPT and LLM-grounded Video Diffusion were not retrieved this session.
- No study was found that measures whether exemplar memories or playbooks improve *human-rated* video quality across sessions; the evidence for those methods is from text and agent tasks.

## 5. Learning the director/planner itself from feedback (preference data, DPO/RLHF/RLAIF) and what is practical with hosted frontier LLMs

### Takeaway
With Claude as a hosted director, learning at the weight level is effectively unavailable. Late-2026 secondary reports find no fine-tuning route for Claude Sonnet or Opus, and OpenAI's self-serve fine-tuning is winding down. The practical lever is therefore learning in context:
- Infer latent preferences from users' plan edits (CIPHER).
- Maintain a director playbook updated by deltas (ACE).
- Run GEPA or Training-Free-GRPO-style reflective optimization on the planning instructions, using accepted-vs-revised plans and downstream video outcomes as feedback.

If weight learning is wanted, the evidence points to training a *small open-weight* rewriter or planner with DPO or GRPO, as VPO, Prompt-A-Video, PromptEnhancer, SPIRAL and DramaDirector do. That is possible via Vertex AI preference tuning (Gemini 2.5 Flash/Flash-Lite) or Tinker (LoRA on open models).

### Cited Findings
- **PRELUDE / CIPHER** (Microsoft, NeurIPS 2024) learn from user edits to the agent's output without fine-tuning:
  - An LLM infers a natural-language description of the user's latent preference from their edits.
  - Preferences inferred in the most similar past contexts are retrieved and aggregated, then used to condition new generations.
  - With a GPT-4-simulated user, CIPHER achieved the lowest edit-distance cost on summarization and email writing; a secondary write-up cites 31% and 73% reductions versus no learning. The learned preferences are interpretable and similar to the ground-truth preferences.
  - Sources — [HF papers](https://huggingface.co/papers/2404.15269); [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/96078)
- **RLAIF** (Google, ICML 2024). Against an SFT baseline, RLAIF won 71% (summarization) and 63% (helpful dialogue) of human comparisons, versus 73% and 64% for RLHF; the differences were not significant. On harmlessness RLAIF scored 88% against 76% for RLHF and 64% for SFT. An AI labeler the same size as the policy still helped, and "direct" RLAIF, which skips the reward model, matched or beat canonical RLAIF — [arXiv HTML](https://arxiv.org/html/2309.00267v3)
- **OpenAI fine-tuning.**
  - Per OpenAI's update note dated May 8, 2026 (as quoted in coverage), the self-serve platform is "no longer accessible to new users". Existing users can create jobs "for the coming months", and fine-tuned models remain available until their base models are deprecated.
  - Third-party summaries of the deprecations page give 2027-01-06 as the date existing customers can no longer create jobs (including reinforcement fine-tuning) and 2027-01-14 as the shutdown for GPT-4o, 4o-mini, 4.1, 4.1-mini and their fine-tunes.
  - The DPO guide states that DPO supports text only.
  - Sources — [Tessl report (secondary)](https://tessl.io/blog/openai-shutting-fine-tuning-signals-for-enterprise-ai); [developersdigest (secondary)](https://www.developersdigest.tech/models/fine-tuning/openai); [OpenAI DPO guide](https://developers.openai.com/api/docs/guides/direct-preference-optimization/index.html)
- **Claude fine-tuning.**
  - Historically the only route was Claude 3 Haiku on Amazon Bedrock, generally available in US-West (Oregon) — [AWS blog](https://aws.amazon.com/blogs/aws/fine-tuning-for-anthropics-claude-3-haiku-model-in-amazon-bedrock-is-now-generally-available)
  - A late-2026 third-party article says new jobs stopped in March 2026 when Claude 3 Haiku went to Legacy status, that it reached end-of-life on 2026-09-10, and that "there is no documented route as of 30 October 2026" for Sonnet or Opus. Earlier 2026 blogs still describe Bedrock as the route. These sources are secondary and conflict — [Ertas (secondary)](https://www.ertas.ai/blog/can-you-fine-tune-claude); [CallSphere (secondary)](https://callsphere.ai/blog/vw8g-anthropic-claude-fine-tuning-patterns-bedrock-2026)
- **Google Vertex AI** supports DPO preference tuning for Gemini 2.5 Flash and Flash-Lite (per a Google developer-forum post). Google recommends SFT on the preferred responses first, then preference tuning from that checkpoint. Limits are text-only modality, a 1 GB training file, 131,072 input+output tokens per example, and adapter sizes 1/2/4/8/16. The API takes `preferenceOptimizationSpec` or `method="PREFERENCE_TUNING"` — [Vertex docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-preference-tuning); [forum post](https://discuss.google.dev/t/preference-tuning-now-available-for-gemini-2-5-flash-models-on-vertex-ai/287524)
- **Tinker** (Thinking Machines, Oct 2025) is a managed LoRA post-training API for open-weight models. Launch coverage names the Qwen and Llama series, including MoE models such as Qwen3-235B-A22B; a third-party note also lists DeepSeek, Kimi and Nemotron. It supports SFT, RL and preference-optimization loops through low-level primitives (`forward_backward`, `optim_step`, `sample`), and trained weights are downloadable — [InfoQ](https://www.infoq.com/news/2025/10/thinking-machines-tinker); [Tinker docs](https://tinker-docs.thinkingmachines.ai); [The Batch](https://www.deeplearning.ai/the-batch/thinking-machines-new-tinker-api-makes-it-easier-to-fine-tune-models-on-many-gpus)
- **Planner and rewriter preference training in video** (search snippets, 2026):
  - SPIRAL trains a planning agent with SFT and then DPO, contrasting valid plans against "carefully constructed flawed alternatives" (hallucinations, invalid object interactions, temporal inconsistencies) — [arXiv 2603.08403](https://arxiv.org/pdf/2603.08403)
  - DramaDirector trains a storyboard planner with GRPO, rewarded by an LLM judge that scores whether a plan is ready for video generation — [arXiv 2606.24107](https://arxiv.org/pdf/2606.24107)
  - VPO builds video-level preference pairs by comparing reward scores of generated videos — [secondary review](https://themoonlight.io/review/vpo-aligning-text-to-video-generation-models-with-prompt-optimization)
- **Combining prompt and weight optimization.** DSPy ships `BootstrapFinetune` and `BetterTogether` optimizers alongside MIPROv2, GEPA and SIMBA (primary repo; details not read) — [DSPy optimizers](https://github.com/stanfordnlp/dspy/tree/main/docs/docs/api/optimizers). The GEPA README lists a third-party report of "+142% student performance" from adding GEPA on top of an RL-tuned teacher (self-reported) — [gepa-ai/gepa](https://github.com/gepa-ai/gepa)

### Inferences
- **Build preference data from plan edits.**
  - Log every director output (scene list, shot list, per-frame storyboard layout) with the user's final accepted version and a structured diff.
  - The accepted-vs-original pair is preference data (DPO-format), and the edit distance is a cost signal (CIPHER).
  - With Claude, use the data in context: CIPHER-style inferred preference statements retrieved by story or scene similarity, plus ACE-style director-playbook bullets with helpful/harmful counters. Do not fine-tune.
- **Assign credit between planning and prompt writing.** A low-rated video can be the planner's fault (a bad shot choice) or the prompt writer's fault (a bad prompt for that shot). Track them as separate learnable components with separate feedback: plan edits and acceptance for the director, and per-shot generation outcomes for the prompt writer. GEPA's module-targeted reflection does this kind of per-module credit assignment.
- **Synthetic negatives without training.** SPIRAL's "constructed flawed alternatives" idea works with a hosted model. Claude can generate plausible-but-flawed plan variants (continuity breaks, impossible blocking, off-style 3D or animated looks). These can serve as contrastive few-shot examples in the director's context, and as test items for calibrating the judge.
- **Weight-level fallback.** If learning at the weight level becomes necessary, the feasible configuration in November 2026 is a hybrid. Claude stays director. A distilled open-weight per-target prompt rewriter is trained with SFT and then DPO or GRPO on logged preferences, via Tinker or a self-hosted stack, or Gemini 2.5 Flash on Vertex if the team accepts a non-Claude rewriter. This follows the VPO and Prompt-A-Video recipes. RLAIF results suggest AI preference labels can substitute for human labels for *text-level* plan quality. They are much weaker for video-level judgments (Q6).

### Gaps
- No retrieved study applies DPO or RLHF to an LLM *story or storyboard planner* using human video ratings at app scale. The amount of data needed (number of preference pairs) is unknown.
- Anthropic's own documentation on fine-tuning status was not retrieved; the Claude fine-tuning status is from conflicting secondary sources.
- The OpenAI dates come from third-party summaries of OpenAI notices.
- Whether Vertex preference tuning has been extended to newer Gemini versions in late 2026 was not confirmed.

## 6. Evaluation design: measuring improvement reliably when evaluations are expensive and noisy

### Takeaway
Each evaluation is a costly, high-variance video generation, and VLM judges agree with humans only weakly on generated video: about 53–60% pairwise agreement, κ ≈ 0.18, in 2025–2026 benchmarks. Measurement should therefore rest on:
- Paired, order-balanced comparisons on the same storyboard (same seed where the API allows).
- Always-valid sequential tests for stopping decisions.
- Propensities logged at decision time, for off-policy evaluation.
- A frozen-baseline holdout.
- A periodically refreshed human gold set used to calibrate, and prompt-optimize, the judge.

Judge–human agreement is the dominant driver of the sample size needed.

### Cited Findings
- **Reliability of VLM/MLLM judges on generated video:**
  - UVE-Bench (2025; 15 aspects, pairwise human preference annotations) concludes MLLMs "have not yet achieved reliable capabilities for assessing pairwise video quality" — [arXiv](https://arxiv.org/pdf/2503.09949v1); [secondary review](https://liner.com/review/uve-are-mllms-unified-evaluators-for-aigenerated-videos)
  - JudgeAnything reports 66.55% human agreement on multimodal *understanding* pair comparisons but only 53.37% on multimodal *generation* tasks under checklist evaluation — [secondary review](https://www.themoonlight.io/review/judge-anything-mllm-as-a-judge-across-any-modality)
  - MEDit-Bench (2026) documents position bias, where judges favour the video shown in position A. Its judge reached 59.9% pairwise agreement with humans (chance is 50%) and Cohen's κ = 0.18 — [arXiv 2607.25300](https://arxiv.org/pdf/2607.25300)
  - AVENUE (2026) mitigates order bias by running comparisons in both A-B and B-A order, and reports pairwise, leave-one-out and Kendall τb agreement — [arXiv 2609.04253](https://arxiv.org/pdf/2609.04253)
  - The original MLLM-as-a-Judge benchmark (2024) names egocentric, position and length biases — [arXiv 2402.04788](https://arxiv.org/pdf/2402.04788v3)
- **Optimizing the judge itself.** A Nubank case study listed by the GEPA maintainers used GEPA to optimize LLM-as-a-judge prompts. Eval accuracy rose from 68.88% to 88.89%, and Cohen's κ between the GPT-4.1 and GPT-4.1-mini judges went from 0.00 to 0.745 (self-reported; arXiv 2606.08867) — [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- **Checklist rewards.** PromptEnhancer's AlignEvaluator scores 24 explicit key points across 6 categories instead of one holistic score — [arXiv](https://arxiv.org/abs/2509.04545). VISTA uses pairwise tournaments for selection plus a multi-critic panel — [CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Long_VISTA_A_Test-Time_Self-Improving_Video_Generation_Agent_CVPR_2026_paper.html)
- **Sequential testing.** Johari, Pekelis and Walsh's "always valid" p-values and confidence intervals stay valid under continuous monitoring. Repeated fixed-horizon tests inflate error, and Kohavi et al. call it "one of the most common mistakes in practical A/B testing". Optimizely uses mSPRT for this, at some cost in power compared with fixed-horizon tests (secondary) — [arXiv 1512.04922](https://arxiv.org/pdf/1512.04922v1); [Morning Paper summary](https://blog.acolyer.org/2017/09/28/peeking-at-ab-tests-continuous-monitoring-without-pain/)
- **Off-policy evaluation of logged bandit data:**
  - Doubly-robust estimation gives lower variance than inverse-propensity scoring and lower bias than direct modelling — [Dudík et al.](https://www.arxiv.org/pdf/1103.4601); [Statistical Science version](https://alphaxiv.org/abs/1503.02834)
  - The Open Bandit Pipeline implements Replay, IPS, self-normalized IPS, DR and DRos estimators — [OBP docs](https://zr-obp.readthedocs.io/)
  - When the logging policy is fixed, propensity and DR estimators use data more efficiently than replay — [contextual R package docs](https://rdrr.io/github/Nth-iteration-labs/contextual/man/OfflineReplayEvaluatorBandit.html)
  - Practitioner guidance: greedy policies push propensities toward 0 or 1, which makes IPS and DR unstable; log propensities *at decision time*, since reconstructing them later adds bias (secondary) — [MetricGate, Feb 2026](https://metricgate.com/blogs/online-evaluation-vs-offline-evaluation/)
- **Off-policy learning when actions are generated text.** "Prompt Optimization with Logged Bandit Data" (Kiyohara, Saito, Cao, Joachims; ICLR 2024 workshop, arXiv 2504.02646) uses a kernel-based off-policy gradient over generated-sentence embedding similarity, which substantially reduces variance with small bias. A RecSys '25 follow-up (DSO) estimates gradients in sentence space rather than prompt-action space — [arXiv](https://arxiv.org/abs/2504.02646v1); [ICLR workshop](https://iclr.cc/virtual/2024/22442); [RecSys '25 PDF](https://www.cs.cornell.edu/people/tj/publications/kiyohara_etal_25a.pdf)

### Inferences
- **Sample sizes for paired win-rate tests** (my computation: two-sided α = 0.05, 80% power, normal approximation). Detecting a true win rate of 55%, 60%, 65% or 70% against 50% needs about **783, 194, 85 or 47** paired comparisons. Most strategy edits will be small effects (≤ 60%), so per-context A/B tests are infeasible without pooling (Q3) or multi-fidelity proxies.
- **Judge noise dilutes the signal sharply.**
  - If the judge agrees with humans with probability a, the observed win rate is 0.5 + (2a−1)(p−0.5), where p is the true human win rate.
  - At a = 0.6, close to MEDit-Bench's 59.9%, a true 60% human win shows up as 52% and needs about **4,900** comparisons to detect.
  - At a = 0.8 it needs about 543, and at a = 0.9 about 305 (my computation).
  - Raising judge agreement is therefore the highest-leverage investment in sample efficiency. The tools for it: per-dimension checklists, both-order evaluation, judge-prompt optimization against human labels, and the app's existing calibration of judge weights from star ratings.
- **Recommended measurement protocol for this layer:**
  1. **Paired design.** Run both arms on the same storyboard frame and shot plan, with a shared seed when the provider exposes one (common random numbers). Show human raters the pair in randomized order, and have the VLM judge both orders.
  2. **Judge as calibrated proxy.** For each objective (adherence, photographic realism vs CGI/animation look, motion, continuity, aesthetics), track judge–human Cohen's κ / Krippendorff's α on a rolling human-rated sample. Down-weight objectives with low κ.
  3. **Frozen baseline holdout** (e.g. 5–10% of traffic). It measures cumulative improvement and catches vendor drift.
  4. **Always-valid sequential tests** (mSPRT or Bayesian with pre-registered thresholds) for promoting or retiring strategies and playbook versions.
  5. **Log TS propensities at decision time.** Estimate the probability that each arm is the posterior argmax by Monte Carlo over posterior draws, and store it with the candidate set, prompt embeddings and model version. This enables IPS/DR replay of alternative policies, and kernel/embedding-based estimators for free-form prompts in the style of Kiyohara et al.
  6. **Monthly human gold set** of about 30–50 representative scenes across targets × styles, regenerated with the current versus frozen policy and rated blind and pairwise. With about 4 raters that gives about 200 comparisons, enough for roughly 60/40 effects.
  7. **Multi-objective guardrails.** Accept a change only if it is Pareto-non-inferior across objectives, since OPT2I and HunyuanVideo's Master mode show aesthetic optimization costing adherence.

### Gaps
- No source was retrieved on which hosted APIs expose deterministic seeds (Veo 3.1, Kling 3.0, Seedance 2.0, Sora 2, Wan 2.2 hosted). Seeds are needed for common-random-number designs.
- No study was found quantifying seed-to-seed variance of these models' outputs. That variance determines how many takes per arm are needed.
- Prediction-powered inference (combining many judge labels with a few human labels for unbiased estimates) was not retrieved this session. It is a candidate technique for the judge-plus-human design.
- No retrieved study validates offline replay or OPE specifically on logged generative-media prompt bandits.
