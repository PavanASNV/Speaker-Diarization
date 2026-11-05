# Collecting and Handling Human Feedback for Generative Image/Video Products (state as of November 2026)

*Sourcing note for the report writer: in this environment, page fetches to arxiv.org, fal.ai, huggingface.co and *.github.io project pages all failed (DNS blocked by the session network policy). Every finding below therefore comes from search-engine extracts of primary pages (arXiv/CVF/NeurIPS/ICLR/OpenReview/vendor docs) or from secondary summaries; where a number came only from a secondary summary it is flagged "(secondary)". Dates are given as publication dates of the source. "Our app" = Prompt-to-Scene Creator.*

## 1. Which feedback formats work best for generative media (pairwise vs stars/Likert vs best-of-N vs per-dimension vs region/frame-level vs natural-language critiques vs edits)?

### Takeaway
Pairwise comparison of two outputs from the same prompt, with an explicit tie option, is the best-supported primary format. It is more reliable and discriminative than absolute ratings, gives better reward models than pointwise regression for video (VideoAlign, 2025), and beat both 4-image and no-tie designs on engagement and inter-rater agreement in Pick-a-Pic (2023). Richer formats are worth adding on top of the pairwise core, not instead of it. These are per-dimension A/B/tie labels, binary checklists, region heatmaps or 1-second segment labels, written rationales, and edit traces. Each one fixes a specific weakness of a single overall preference: it shows *why* a take won, *where* it failed, and which dimension drove the choice. Even the best predictors agree with an individual user's pairwise choice only about 65–75% of the time.

### Cited Findings
**Pairwise with ties (primary format)**
- Pick-a-Pic (Kirstain et al., NeurIPS 2023) pilot-tested three interfaces: (1) 4 images, no ties; (2) 2 images, no ties; (3) 2 images with ties. The 2-image, tie-enabled option "exceeds the other two in terms of user engagement and inter-rater agreement". The exact effect sizes were not available in the extracts. — [Pick-a-Pic paper (ar5iv)](https://ar5iv.arxiv.org/html/2305.01569)
- The Pick-a-Pic product loop works like this. A user writes a prompt and gets two images. They pick one or declare a tie, the rejected image is replaced by a new one, and the loop repeats until the prompt changes. This produced >500k examples in v1. PickScore (a CLIP model trained on the data) predicts these real-user choices with 70.2% accuracy vs 68.0% for expert humans. — [NeurIPS 2023 poster](https://neurips.cc/virtual/2023/poster/72229); [Stability AI research page](https://stability.ai/research/pick-a-pic)
- Each Pick-a-Pic example is labeled with the preferred image, or a tie "when no image is significantly preferred". v2 has ">one million examples". One downstream training paper reports 959,040 preference pairs / 1,029,802 unique text–image pairs; which paper this was is not confirmed. — [Pick-a-Pic arXiv PDF](https://arxiv.org/pdf/2305.01569)
- A Jan 2026 analysis reports a 14.2% near-tie prevalence on Pick-a-Pic. — [Lee, "How Many Human Judgments Are Enough?" (arXiv 2601.09084)](https://arxiv.org/html/2601.09084v1)
- Midjourney V7 onboarding (Apr 2025) uses the same pattern: press 1 (left), 2 (right) or 3 ("unsure"). Users rate about 200 images (~5 minutes) to build a personalization profile. Skipped comparisons must still be completed later. (secondary) — [DataCamp V7 tutorial](https://www.datacamp.com/tutorial/midjourney-v7)
- VideoAlign (Kuaishou/KwaiVGI, Jan 2025; NeurIPS 2025) found that a Bradley–Terry (pairwise) reward model "consistently outperform[s]" a regression (pointwise score) model across dataset sizes, with the gap narrowing as data grows. — [VideoAlign arXiv HTML](https://arxiv.org/html/2501.13918v2)
- GenArena (Feb 2026): switching a VLM judge from pointwise to pairwise scoring raised Qwen3-VL-8B-Instruct accuracy against human choices on VideoGen-RewardBench from 57.0% to 61.5%. — [GenArena (arXiv 2602.06013)](https://arxiv.org/pdf/2602.06013)
- Subjective quality-assessment literature: Absolute Category Rating (ACR) is popular for simplicity, but Pair Comparison (PC) has "significant advantages... in terms of discriminability". PC also avoids observers' differing interpretations of a rating scale. Full PC is "much more time-consuming". The proposed fix is ACR-initialized PC plus MST-based active batch sampling (preprint, 2020). — [Ling et al. (arXiv 2010.00370)](https://arxiv.org/pdf/2010.00370)
- Ye & Doermann (CVPR 2014) combine MOS-style ratings and PC through active sampling by expected information gain, reducing experiment cost. They also note that raters are confused about which absolute scale value to give. — [Ye & Doermann, CVPR 2014](https://openaccess.thecvf.com/content_cvpr_2014/papers/Ye_Active_Sampling_for_2014_CVPR_paper.pdf)
- A 2023 study is titled "Crowdsourcing subjective annotations using pairwise comparisons reduces bias and error compared to the majority-vote method" (title-level claim only; details not verified). — [arXiv 2305.20042](https://arxiv.org/pdf/2305.20042)

**Absolute ratings (stars/Likert) vs rankings**
- Bansal, Dang & Grover (ICLR 2024; LLM text, not images) found that preferences derived from ratings and from rankings disagree about 60% of the time, for both humans and GPT-3.5 annotators. Annotators rated denser responses higher but preferred accuracy in pairwise judgments. Models aligned on ranking data looked better under rank-based evaluation but not under rating-based evaluation. — [HF Papers 2308.15812](https://huggingface.co/papers/2308.15812); [arXiv HTML v3](https://arxiv.org/html/2308.15812v3)
- Netflix (2017) replaced 5-star ratings with thumbs up/down after 2016 tests "with hundreds of thousands of members" produced 200% more ratings. VP Todd Yellin: "the implicit signal of your behavior is more important". Users gave documentaries 5 stars but watched lightweight films more often. — [VentureBeat](https://venturebeat.com/2017/04/05/netflix-ditches-5-star-rating-system-for-thumbs-up-and-thumbs-down); [Business Insider](https://www.businessinsider.in/netflix-has-replaced-its-5-star-rating-system-with-thumbs-up-thumbs-down-heres-why/articleshow/58030588.cms)
- Where research datasets do use Likert, they use it per dimension with several raters. Two examples:
  - RichHF-18K (CVPR 2024): four 5-point scores (plausibility, text–image alignment, aesthetics, overall) from 3 annotators per image, averaged and rescaled to [0,1].
  - VideoFeedback (EMNLP 2024): 1–4 scores on 5 dimensions.
  — [RichHF arXiv](https://arxiv.org/pdf/2312.10240); [VideoScore (alphaXiv)](https://alphaxiv.org/abs/2406.15252)

**Best-of-N / K-wise ranking**
- InstructGPT (2022, text) had labelers rank K=4–9 responses, yielding C(K,2) comparisons per prompt. These were trained as one batch element so that correlated pairs do not overfit. (secondary) — [InstructGPT (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.02155); [summary](https://papermap.vercel.app/paper/Training_Language_Models_to_Follow_Instructions_with_Human_Feedback.html)
- Theory supports K-wise rankings, with a caveat about the model assumptions:
  - Plackett–Luce K-wise MLE is asymptotically more efficient than splitting rankings into pairs. — [Zhu, Jordan & Jiao (arXiv 2301.11270)](https://web3.arxiv.org/pdf/2301.11270v5)
  - The benefit may be an artifact of PL assumptions that real evaluators violate. — [arXiv 2306.14111](https://arxiv.org/pdf/2306.14111)
  - Oct 2025 work gives the first bounds that actually improve as subset size grows. — [arXiv 2510.18713](https://arxiv.org/pdf/2510.18713v2)
- No empirical generative-media study comparing best-of-4 to pairwise was found beyond Pick-a-Pic's pilot, where 4-image/no-tie lost (see above).

**Per-dimension scores and binary checklists**
- MPS/MHP (Kuaishou et al., CVPR 2024) argues that "preference results vary when humans evaluate images with different aspects". The MHP dataset has 918,315 choices on 607,541 images across 4 dimensions (aesthetics, semantic alignment, detail quality, overall). A preference-condition module lets one CLIP model score each dimension. — [CVPR 2024 open access](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_Multi-Dimensional_Human_Preference_for_Text-to-Image_Generation_CVPR_2024_paper.html)
- VisionReward (THUDM, Dec 2024) breaks preference into yes/no checklist questions:
  - Video: 9 dimensions / 64 questions. Images: 5 dimensions / 18 sub-dimensions / 61 questions.
  - A VLM answers each question, and a linear weighting produces an interpretable score.
  - Data: 81K samples, ~5M binary annotations.
  - Reported results: 17.2% better video preference prediction than VideoScore, and a 31.6% higher pairwise win rate for models optimized with it.
  — [HF Papers 2412.21059](https://huggingface.co/papers/2412.21059); [arXiv v4](https://arxiv.org/html/2412.21059v4)
- FIRM-Video (Aug 22, 2026) uses "check-before-score":
  - Build per-dimension checklists, verify each criterion against temporal visual evidence, and aggregate only verified decisions.
  - FIRM-Video-90K = 88,044 dimension-specific instances from 29,348 videos and 3,012 prompts.
  - FIRM-Video-8B (Qwen3-VL) reports the best MAE on its human benchmark (0.78; secondary).
  — [arXiv 2608.21839](https://arxiv.org/pdf/2608.21839); [OpenTrain summary](https://www.opentrain.ai/papers/firm-video-check-before-you-score-for-reliable-text-to-video-reward-modeling--arxiv-2608.21839/)

**Region- and frame/segment-level annotations**
- RichHF-18K (Google, CVPR 2024):
  - Annotators click points on implausible/artifact regions and on text-misaligned regions. Each point becomes a disk of radius 1/20 of image height, producing heatmaps.
  - They also mark misaligned prompt keywords and give the four scores above.
  - Labels are consolidated by averaging scores and heatmaps and by majority vote on keywords.
  - The predicted feedback was used to select fine-tuning data and to build inpainting masks for problem regions.
  — [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Rich_Human_Feedback_for_Text-to-Image_Generation_CVPR_2024_paper.pdf); [arXiv 2312.10240](https://arxiv.org/abs/2312.10240); [Google Research blog](https://research.google/blog/rich-human-feedback-for-text-to-image-generation/)
- BrokenVideos (Jun 2025): 3,254 AI-generated videos with pixel-level artifact masks.
  - Annotation used a SAM2-based GUI with tracked propagation and frame-by-frame refinement, with multi-label, multi-instance labels: 80% of videos have 1–2 regions, 17% have 3–6, 3% have 7–10.
  - Training detectors and MLLMs on it "significantly improves" localization.
  — [arXiv 2506.20103](https://arxiv.org/abs/2506.20103)
- DenseDPO (Snap, NeurIPS 2025): vanilla video DPO compares clips from independent noise, which "biases the annotators towards low-motion clips as they often contain fewer visual artifacts".
  - The fix: build pairs by denoising corrupted copies of one video, so motion is aligned, then label ~1-second segments.
  - With one-third of the labeled data it improves dynamic degree while matching vanilla DPO on alignment, visual quality and temporal consistency (secondary for benchmark numbers).
  - GPT-predicted segment labels perform close to human labels.
  — [arXiv 2506.03517](https://arxiv.org/pdf/2506.03517); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/117435); [review](https://www.themoonlight.io/review/densedpo-fine-grained-temporal-preference-optimization-for-video-diffusion-models)

**Natural-language critiques / rationales**
- LiFT (Dec 2024): LiFT-HRA has ~10k human annotations, each with per-dimension scores and a written rationale. The authors note that prior work "overlook[s] the reasoning behind evaluations". A reward model trained on it (LiFT-Critic), combined with reward-weighted likelihood fine-tuning, made CogVideoX-2B beat CogVideoX-5B on all 16 metrics. — [HF Papers 2412.04814](https://huggingface.co/papers/2412.04814); [arXiv](https://arxiv.org/abs/2412.04814v1)
- VideoScore2 / VideoFeedback2 (TIGER-Lab, Sep 2025):
  - 27,168 videos from 22 T2V models, with human scores plus reasoning traces on 3 dimensions (visual quality, text alignment, physical/common-sense consistency).
  - Training: SFT, then GRPO.
  - Results: 44.35 in-domain accuracy (+5.94) and 50.37 average out-of-domain (+4.32). License: Apache-2.0.
  — [arXiv 2509.22799](https://arxiv.org/html/2509.22799v1); [dataset card](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2/blob/main/README.md)
- ViPer (ECCV 2024): users comment freely on a set of images. A fine-tuned IDEFICS2-8b turns the comments into liked/disliked attributes that steer generation with no training (see §5). — [ECCV 2024 paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9492_ECCV_2024_paper.php)

**Edits/revisions as implicit preference**
- PRELUDE/CIPHER (Microsoft, NeurIPS 2024):
  - An LLM infers a natural-language description of the user's latent preference from each edit and stores it with its context. For new requests it retrieves and merges preferences from similar contexts, with no fine-tuning.
  - It had the lowest edit cost against baselines. The secondary-reported reductions are 31% (summarization) and 73% (email) vs no learning.
  - The learned preferences are human-readable and editable.
  — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/aligning-llm-agents-by-learning-latent-preference-from-user-edits/); [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/96078)
- Diffusion-DRO (NeurIPS 2025, "Ranking-based Preference Optimization for Diffusion Models from Implicit User Feedback"): preference learning grounded in inverse RL that needs "expert demonstrations only" plus online policy-generated negatives, with no paired comparisons. Secondary-reported user-study win rates: 75% vs SD1.5, 56.67% vs Diffusion-DPO+SFT, 66.67% vs Diffusion-KTO+SFT. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118987); [arXiv 2510.18353](https://arxiv.org/abs/2510.18353v1); [review](https://www.themoonlight.io/review/ranking-based-preference-optimization-for-diffusion-models-from-implicit-user-feedback)

**Cost and agreement reference points**
- Collecting Pick-a-Pic v2 cost about $50K. A similar-scale synthetic set labeled by a pretrained reward model cost about $200 (authors' estimate). Their RankDPO on synthetic ranked labels improved SDXL and SD3-Medium. — [RankDPO, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Karthik_Scalable_Ranked_Preference_Optimization_for_Text-to-Image_Generation_ICCV_2025_paper.html); [arXiv 2410.18013](https://arxiv.org/pdf/2410.18013)
- Reported annotator agreement:
  - InstructGPT: training labelers agreed 72.6 ± 1.5% of the time, held-out labelers 77.3 ± 1.3% (secondary). — [summary](https://papermap.vercel.app/paper/Training_Language_Models_to_Follow_Instructions_with_Human_Feedback.html)
  - ImageReward: annotator–researcher agreement ~71–73% (secondary, unverified). — [Emergent Mind](https://www.emergentmind.com/topics/imagereward)
- VLM-judge agreement with humans on video:
  - 2024: VideoScore reached Spearman 77.1 with humans, while "Gemini and GPT-4o also achieve very low agreement with human raters". — [VideoScore](https://alphaxiv.org/abs/2406.15252)
  - Aug 2026 (VGI-Bench), pairwise accuracy vs humans: Gemini-3-Flash judge 73.2%, GPT-5-mini 64.3%, Claude-Haiku-4.5 47.9%. The protocol and chance level were not confirmed. — [arXiv 2608.19583](https://arxiv.org/pdf/2608.19583)
  - VideoRewardBench (a different benchmark from VideoGen-RewardBench): top of 28 multimodal reward models is Gemini-2.5-Pro at 63.6%. — [alphaXiv leaderboard](https://alphaxiv.org/benchmarks/university-of-science-and-technology-of-china/videorewardbench)

### Inferences
- **Make the primary explicit signal a within-shot A/B with ties.** For a given shot (same storyboard frame, same prompt version, possibly different models or seeds), show two takes side by side. Offer: A better / B better / tie, both usable / both unusable. "Both unusable" is a separate negative label: a "both bad" tie carries different information from a "both good" tie. This is Pick-a-Pic's winning design plus VideoAlign's A/B/tie, and it is also the format the VLM judge is most accurate in (GenArena: pairwise > pointwise).
- **Keep stars, but don't build on them.** Treat 1–5 stars as an optional, low-weight signal, and convert them into *relative* comparisons only within the same user, session and shot. Two reasons: ratings and rankings disagree about 60% of the time (Bansal), and absolute scales are interpreted differently by different people (ACR literature). Netflix's 200% uplift suggests a binary thumb plus optional reason chips will get far more responses than stars.
- **Collect "why" as chips that map onto the judge's 7 dimensions:** prompt alignment, photoreal vs 3D/CG look, composition vs storyboard, character continuity, motion, surreal integration, aesthetic. Each chip is a binary checklist item, as in VisionReward and FIRM-Video. It costs one tap and gives per-dimension pairwise labels, as in MPS and VideoAlign. Do not ask users to fill in 7 Likert scales per take.
- **Add cheap localization:**
  - "Flag a moment" stores a ~1-second segment, following DenseDPO.
  - "Tap the problem area" stores a point with a radius, following RichHF's disk heatmaps.
  - Store both as `(take_id, t_start, t_end, x, y, radius, dimension_tag)`.
  - This makes regional re-generation and inpainting possible (RichHF used masks for inpainting). It also gives the judge evidence to "check before scoring".
- **Treat "Direct this shot" text and edits as the richest signal.**
  - Log the raw text.
  - Have Claude extract structured tuples `(dimension, polarity, target region/time, attribute, desired change)` from it.
  - Keep CIPHER-style natural-language preference statements per project.
  - When a revised take is approved, record a pairwise label `revised > original` tagged with the stated reason.
- **Information per interaction** (simple arithmetic, not from a source):
  - A pairwise choice carries ≤1 bit, or ≤log2(3) ≈ 1.58 bits with a tie option.
  - "Pick the best of 4" carries ≤2 bits.
  - A full ranking of 4 carries ≤4.6 bits, but costs much more effort and is less reliable.
  - Recommendation: when a shot has ≥3 takes, offer "pick best (optionally also worst)" and convert it to pairs by rank-breaking or Plackett–Luce. Don't force full rankings.
- **Calibrate expectations for the judge.** An individual user's pairwise choices are noisy. Experts reach ~68% on Pick-a-Pic, labelers agree ~73–77% in RLHF, and the best VLM judges agree 60–75% with humans on video. So judge–user agreement in the 65–75% range is roughly at the ceiling, not a failure. Track it as a KPI per dimension.

| Format | Reliability / info | User cost | Best use in our app |
|---|---|---|---|
| Pairwise + tie (per shot) | Highest reliability; ≤1.58 bits | Low | Primary preference label; judge calibration |
| Best-of-N pick | ≤log2(N) bits; PL assumptions | Low–medium | When ≥3 takes exist |
| Stars 1–5 | Scale-usage bias; inconsistent with rankings | Low | Optional; within-user relative only |
| Reason chips (7 dims) | Binary checklist; interpretable | Very low | Attach to every thumbs/A-B |
| Region/segment flags | Localizes failures | Medium | Artifacts, continuity breaks, CG-look spots |
| NL critique / "direct this shot" | Richest, needs LLM parsing | Medium–high | Revisions; per-project style memory |
| Edits / revisions | Implicit, high intent | Zero extra | `revised > original` pairs; preference statements |

### Gaps
- Exact Pick-a-Pic pilot numbers (engagement, agreement for each of the 3 interfaces) and VideoAlign's inter-annotator agreement could not be retrieved, because arxiv.org, huggingface.co and github.io fetches were blocked.
- No study was found that directly compares star ratings with pairwise comparison *for generative video specifically*, or that measures the reliability of 1–5 star ratings by end users of creative apps.
- No empirical head-to-head of best-of-4 vs pairwise for video reward-model training was found.

## 2. What implicit signals do products use, how are they turned into preference data, and what is publicly known about Midjourney, Runway, Pika, Kling, Luma, Sora, Google Veo/Flow, ByteDance and Adobe?

### Takeaway
Public detail is thin. Vendors document *that* they collect interaction data and feedback, but almost never *how* re-rolls, upscales or downloads become training labels. What is documented:
- **Midjourney:** explicit pair-ranking ("Rank Pairs"/"rating parties", ~200 ratings to unlock V7 personalization) plus Explore-page likes.
- **OpenAI (Sora app):** likes, comments and remixes feed a steerable recommendation feed.
- **Google Flow:** thumbs feedback that human reviewers see.
- **Pika:** terms cover training on "user interactions".
- **ByteDance (Seedance 1.0) and Kuaishou (Kling's maker):** human pairwise labels feed multi-dimensional video reward models (foundational/motion/aesthetic for Seedance; visual quality/motion quality/text alignment for VideoAlign).
- **Adobe:** says it does not train Firefly on customer content.

Research shows that sparse implicit or engagement signals can drive RL. Meta's Love reactions raised Love rates by up to 28% in live A/B tests. But such signals are easy to reward-hack.

### Cited Findings
**Implicit and sparse signals turned into rewards (research)**
- Meta "Reinforcement Learning from User Feedback" (May 2025):
  - A P[Love] reward model was trained on binary "Love" reactions, which are sparse (~0.1% of messages).
  - Love reactions correlated more strongly with 14-day retention than thumbs-up did.
  - Optimizing P[Love] alongside helpfulness and safety rewards raised Love reactions by up to 28% in live A/B tests.
  - Over-optimization produced reward hacking: repetitive sign-offs like "Bye! Sending Love!".
  — [arXiv HTML 2505.14946](https://arxiv.org/html/2505.14946v1); [alphaXiv overview](https://www.alphaxiv.org/overview/2505.14946)
- Diffusion-DRO (NeurIPS 2025) learns from "implicit user feedback" framed as expert demonstrations (positives) plus online model samples (negatives), with no explicit pairs (details in §1). — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118987)
- Netflix (2017) deliberately de-emphasized explicit ratings because "the implicit signal of your behavior is more important". — [Business Insider](https://www.businessinsider.in/netflix-has-replaced-its-5-star-rating-system-with-thumbs-up-thumbs-down-heres-why/articleshow/58030588.cms)
- Pick-a-Pic shows how to make labeling the product action itself: choosing the better image *is* how the user keeps working (the rejected image is replaced). — [Stability AI research page](https://stability.ai/research/pick-a-pic)

**Midjourney**
- Official docs: personalization (`--p`) is created "by ranking pairs on midjourney.com". You can only use `--p` once you have done enough Rank Pairs. Users can apply other users' personalization codes. — [Midjourney docs: Personalization](https://docs.midjourney.com/docs/personalization)
- Personalization codes update as users rank more pairs *or like images on the Explore page*. Skipping a pair doesn't affect the profile (third-party guides, 2024). — [Dataconomy guide](https://dataconomy.com/2024/06/12/midjourney-model-personalization-guide); [MJ Tips & Tricks](https://mjtipsandtricks.substack.com/p/midjourney-ai-personalization-unlock)
- Official site docs (current), three pages:
  - The Personalize page builds custom aesthetics "by rating images".
  - The Moodboards page curates image sets for a look.
  - The Tasks page offers "activities that help tailor your Personalization profiles and potentially earn some rewards".
  - The Explore "For You" view "is based mostly on your Global V7 Personalization Profile".
  — [Midjourney docs: Website Overview](https://docs.midjourney.com/docs/explore-page)
- V7 required rating about 200 images (~5 min) before first use (Apr 2025; secondary). — [DataCamp](https://www.datacamp.com/tutorial/midjourney-v7)
- "Rating parties" (V6 era): paying members rated image pairs, "two images at a time", so the system could learn what the community wanted from a new model (secondhand quote of staff). — [Geeky Gadgets](https://www.geeky-gadgets.com/midjourney-6-training/); see also [The Decoder on faster personalization and moodboards](https://the-decoder.com/midjourney-adds-faster-model-customization-and-mood-board-support/)
- Midjourney's terms say content is public and remixable by default, under a license granted to Midjourney. Stealth Mode (Pro/Mega) is only a "best efforts" promise not to publish. — [Midjourney Terms of Service](https://docs.midjourney.com/docs/terms-of-service)

**OpenAI (Sora app)**
- The "Sora feed philosophy" describes "steerable ranking", so users can tell the algorithm what they want to see. ChatGPT-history use can be turned off in Sora's Data Controls. — [OpenAI: Sora feed philosophy](https://openai.com/index/sora-feed-philosophy)
- According to secondary summaries of that page, ranking signals are:
  - Sora activity: posts, likes, comments, remixes.
  - Approximate location from IP.
  - Engagement metrics.
  - Author-level data such as follower counts.
  - ChatGPT history, unless turned off.
  The ranking reportedly favors creation and connection over watch time. — [gend.co summary](https://www.gend.co/blog/sora-feed-philosophy)
- Sora has its own training opt-out in Sora Settings, and the privacy-portal opt-out also applies to Sora. In ChatGPT, thumbs feedback can cause "the entire conversation associated with that feedback" to be used for training "even if you've opted out". — [OpenAI: How your data is used](https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/); [OpenAI Help Center](https://help.openai.com/en/articles/5722486-how-do-i-opt-out-of-my-content-being-used-to-improve-model-performance)

**Google (Veo / Flow)**
- Flow privacy notice (effective May 19, 2026):
  - Human reviewers see and annotate a sample of uploads and outputs after the data is disconnected from the Google Account.
  - Reviewed or annotated data is kept separately even if the user deletes their data or turns off "Help improve Flow".
  - Deleted data may persist up to 72 hours.
  — [labs.google/flow/privacy](https://labs.google/flow/privacy)
- Labs-wide notice: "human reviewers read, annotate, and process" a sample of data. — [Google Labs VideoFX privacy](https://labs.google/fx/tools/video-fx/privacy)
- Labs data is stored up to 18 months by default (news report). — [AOL](https://www.aol.com/googles-ai-filmmaker-program-flow-153000467.html)
- Google feedback policy, stated for NotebookLM/Gemini Notebook, not confirmed for Flow: thumbs feedback is reviewed by trained teams after disconnection from the account and retained up to 3 years. Workspace and Education users are excluded from human review. — [Google support: What happens when you send feedback](https://support.google.com/notebooklm/answer/17004255)

**ByteDance (Seedance / Dreamina / CapCut)**
- Seedance 1.0 tech report (Jun 2025): the RLHF stage uses three reward models:
  - A Foundational RM (a VLM) for image–text alignment and structural stability.
  - A Motion RM to "mitigate video artifacts while enhancing motion amplitude and vividness".
  - An Aesthetic RM working on keyframes in image space, "inspired by Seedream".
  Human preference labels were collected on video pairs, including "synthetic videos generated by different stages of our model". Human preference data is also used in adversarial post-training during distillation. — [Seedance 1.0 (arXiv 2506.09113)](https://arxiv.org/pdf/2506.09113); [ByteDance Seed blog](https://seed.bytedance.com/en/blog/tech-report-of-seedance-1-0-is-now-publicly-available)
- CapCut's June 12, 2025 terms grant a perpetual, irrevocable, sublicensable, worldwide license over user content, which drew a backlash. Dreamina's terms were last updated Jan 22, 2026. — [Techloy](https://www.techloy.com/capcuts-latest-terms-of-service-raises-big-questions-about-content-ownership/); [Digital Camera World](https://www.digitalcameraworld.com/tech/social-media/capcuts-new-terms-of-service-are-angering-the-internet-and-so-confusing-that-video-editors-are-asking-chatgpt-to-decipher-them); [Dreamina ToS](https://dreamina.capcut.com/clause/dreamina-terms-of-service)

**Kuaishou (Kling)**
- Kuaishou research shows multi-dimensional human-preference pipelines for video. VideoAlign has 182k human triplets with visual-quality, motion-quality and text-alignment labels (§3). MPS has 4-dimension image preferences. — [VideoAlign](https://arxiv.org/html/2501.13918v2); [MPS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_Multi-Dimensional_Human_Preference_for_Text-to-Image_Generation_CVPR_2024_paper.html)
- Kling's terms, per third-party trackers (low confidence):
  - Sec. 4.7.3(f) reserves Kling's right to use Input for its own model training.
  - Sec. 3.1(w) bars users from using Output or platform data to train third-party AI models.
  - Training or authorization can reportedly be revoked by emailing support@kling.ai.
  — [ConductAtlas: Kling ToS](https://conductatlas.com/platform/kling-ai/kling-ai-terms-of-service/); [VerifyWise](https://verifywise.ai/ai-trust-index/kling-ai)

**Runway, Pika, Luma**
- Runway (third-party analyses): consumer plans license inputs and outputs for training, with no opt-out outside Enterprise. Enterprise Services Terms are separate. The Usage Policy restricts using outputs to train competing models. — [terms.law](https://terms.law/ai-output-rights/runway/); [ConductAtlas: Runway training license](https://conductatlas.com/platform/runway/runway-terms-of-service/ai-training-license-over-inputs-and-outputs/); [Runway Enterprise Terms](https://runway.com/enterprise-terms)
- Pika's terms (quoted by a tracker): "Inputs, Outputs, and user interactions with the Service may be used by Pika to train, enhance, evolve and improve its machine learning models". This explicitly includes interaction (implicit-feedback) data. — [ConductAtlas: Pika](https://conductatlas.com/platform/pika/pika-terms-of-service/pika-may-use-content-to-train-ai-models/)
- Luma's business/API terms: Luma "will not use Input... via your Business or API use... or output... to train" its models. The consumer privacy policy lists model training as a purpose for uploads. — [Luma Enterprise/API terms](https://lumalabs.ai/api/terms); [Luma privacy policy](https://lumalabs.ai/learning-hub/privacy-policy)

**Adobe Firefly**
- Adobe "does not and has never trained Adobe Firefly on customer content". Firefly is trained on licensed content (Adobe Stock) and public-domain works. Adobe "provides feedback mechanisms" for reporting biased outputs. This followed the June 2024 terms-of-use backlash. — [Adobe AI ethics](https://www.adobe.com/sg/ai/overview/ethics/link-free.html); [Adobe Firefly business](https://business.adobe.com/dk/products/firefly-business/firefly-ai-approach.html); [Axios](https://www.axios.com/2024/06/18/adobe-ai-customer-data-training)

**Public arenas as feedback infrastructure**
- Artificial Analysis Video Arena ranks models by Elo from blind pairwise votes on two videos generated from the same prompt. Ratings with audio use only votes from "validated private evaluators". Ratings without audio also include public votes cast before Jan 1, 2026. This is a move away from open crowd votes toward vetted raters. — [AA image-to-video leaderboard](https://artificialanalysis.ai/video/leaderboard/image-to-video); [AA text-to-video leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video)

### Inferences
- **Proposed signal taxonomy for our app.** These weights are hypotheses to validate against explicit A/B labels.
  - *Strong positive:* take approved or placed in the timeline; final export or download containing the take.
  - *Medium positive:* "extend / variations from this take", "use as reference or keyframe", "make more like this".
  - *Weak positive:* replays, long dwell.
  - *Weak negative:* re-roll with no take selected. This applies to every take *shown*, not to all generated ones.
  - *Negative:* delete; revert to an earlier take.
  - *Targeted negative:* "direct this shot" critique. It applies to the dimensions named, not to everything about the take.
- **Only form implicit pairs within a matched context:** same project, shot, storyboard frame and prompt version. A pair looks like "approved take ≻ each other take the user actually viewed". Log exposure, viewing time and screen position for every take. A take the user never saw is not a rejection.
- **Use approved takes as positive "demonstrations" and rejected or re-rolled takes as negatives.** This is the Diffusion-DRO / RankDPO-style setup. It suits prompt-policy learning on the Claude side, since we do not fine-tune the hosted video models. Use it for model routing, prompt-template preferences and the judge's calibration.
- **Don't optimize a single engagement proxy.** An "approval rate" reward will be hacked, as RLUF's "Sending Love!" shows. Examples: prompt styles that produce low-motion "safe" clips, or over-polished CG-looking renders that win at a glance. Keep the 7 judge dimensions as separate objectives and constraints.
- **Industry practice matches our setup.** Kuaishou (VideoAlign: visual quality / motion quality / text alignment) and ByteDance (Seedance: foundational / motion / aesthetic reward models) both use separate dimension-specific rewards, including a dedicated motion reward. That supports a 7-dimension judge over a single "overall" score.

### Gaps
- No vendor (Midjourney, Runway, Pika, Kling, Luma, OpenAI, Google, ByteDance, Adobe) publicly documents how re-rolls, upscales, downloads, variations or deletions become training labels or how they are weighted. Treating Midjourney's upscale or variation choices as preference signals is common lore but was not confirmed by any official source found.
- No RLHF details were found for Veo 3.1, Kling 3.0, Sora 2 or Wan 2.2 (Wan was not specifically searched). The latest detailed public vendor account found was Seedance 1.0 (Jun 2025).
- No source was found on whether Runway, Kling, Luma or Google Flow show thumbs or rating widgets in their generation UIs, or on response rates for such widgets.
- The Sora feed philosophy page's publication date could not be confirmed (secondary summaries disagree).

## 3. How are large human-preference datasets for images/video built (guidelines, multi-dimensional rubrics, ties, annotator QC, disagreement handling)?

### Takeaway
The current recipe has four parts:
1. Real or LLM-expanded prompts, including hard categories.
2. Outputs from many generators: 9–22 models, sometimes with real images as anchors.
3. Pairwise A/B/tie or per-dimension labels, often several dimensions per pair.
4. Several annotators per item (3 for RichHF, 9–19 for HPDv3), plus inspection or reliability weighting.

Ties are increasingly modeled rather than discarded (VideoAlign's Bradley–Terry-with-ties, BTT). Annotator disagreement is handled by consolidation rules (mean or majority vote), uncertainty-aware losses (HPSv3), or per-rater weights (Rapidata's userScore). Video datasets have moved from single scores to rationales (LiFT, VideoScore2) and checklists (VisionReward, FIRM-Video).

### Cited Findings
- **Pick-a-Pic** (2023): real users of a free web app. Each example is (prompt, two images, preferred-or-tie label). v1 has >500k examples; v2 has >1M. — [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/72229); [arXiv PDF](https://arxiv.org/pdf/2305.01569)
- **ImageReward** (NeurIPS 2023): a 3-stage pipeline of prompt annotation, text–image rating, and image ranking. After 2 months it had 136,892 compared pairs over 8,878 prompts. "Quality inspectors double-check each annotation" and invalid ones are reassigned for relabeling. — [NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/33646ef0ed554145eab65f6250fab0c9-Paper-Conference.pdf)
- **HPD v2** (2023): 798,090 human preference choices on 433,760 image pairs (v1 of the paper said 430,060), from 9 generators plus COCO Captions. HPS v2 fine-tunes CLIP on it. — [arXiv 2306.09341](https://arxiv.org/abs/2306.09341v2)
- **HPDv3 / HPSv3** (ICCV 2025): 1.08M text–image pairs and 1.17M pairwise comparisons, covering autoregressive and diffusion outputs *plus real images of varying quality*. Each sample was compared by roughly 9–19 annotators. The model (Qwen2-VL based) uses an "uncertainty-aware ranking loss". The dataset license is reportedly CC BY-NC-SA (secondary). — [arXiv HTML 2508.03789](https://arxiv.org/html/2508.03789v1); [Emergent Mind](https://www.emergentmind.com/topics/human-preference-score-v3-hpsv3)
- **MHP / MPS** (CVPR 2024): 918,315 choices on 607,541 images, labeled separately for aesthetics, semantic alignment, detail quality and overall. — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_Multi-Dimensional_Human_Preference_for_Text-to-Image_Generation_CVPR_2024_paper.html)
- **RichHF-18K** (CVPR 2024): 18K Pick-a-Pic images (16K train / 1K val / 1K test), 3 annotators per image. Consolidation: average the four scores, majority-vote the keywords, average the heatmaps. A later paper says 27 annotators took part in total (unverified). — [RichHF arXiv](https://arxiv.org/pdf/2312.10240); [ImageDoctor (arXiv 2510.01010)](https://arxiv.org/pdf/2510.01010)
- **VisionReward** (Dec 2024): 81K samples with ~5M binary checklist annotations. Video: 9 dimensions / 64 questions. Image: 5 dimensions / 18 sub-dimensions / 61 questions. — [HF Papers](https://huggingface.co/papers/2412.21059)
- **VideoFeedback / VideoScore** (EMNLP 2024): 37.6K videos from 11 T2V models (plus some real videos for augmentation) with VidProM prompts. Each video is scored 1–4 on visual quality, temporal consistency, dynamic degree, text-to-video alignment and factual consistency. — [alphaXiv 2406.15252](https://alphaxiv.org/abs/2406.15252)
- **VideoAlign / VideoReward** (Jan 2025; NeurIPS 2025):
  - 12 T2V models generated 108k videos from 16k prompts, yielding 182k annotated triplets (prompt, video A, video B).
  - Each triplet has A-wins / B-wins / tie labels for Visual Quality, Motion Quality and Text Alignment.
  - The reward model uses Bradley–Terry *with ties*. Secondary summaries say standard BT discards tie labels, while BTT keeps them for robustness and accuracy.
  - Accuracy is reported both with and without ties. Code and data are reportedly at KwaiVGI/VideoAlign on GitHub.
  — [VideoAlign arXiv HTML](https://arxiv.org/html/2501.13918v2); [ar5iv](https://ar5iv.labs.arxiv.org/html/2501.13918)
- **LiFT-HRA** (Dec 2024): prompts expanded by an LLM from random category words. About 10k annotations, each with per-dimension scores and a rationale. — [HF Papers 2412.04814](https://huggingface.co/papers/2412.04814)
- **VideoFeedback2** (Sep 2025): 27,168 videos with three-dimensional scores and reasoning traces.
  - Prompts come from VidProM and Koala-36M, plus hand-written prompts for multi-action, OCR-text and camera-motion cases.
  - Each prompt was rendered by 10 of 22 models, chosen at random.
  — [arXiv 2509.22799](https://arxiv.org/html/2509.22799v1); [dataset card](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2/blob/main/README.md)
- **FIRM-Video** (Aug 2026): a checklist-verified, dimension-specific training set (88,044 instances / 29,348 videos / 3,012 prompts). There is also a separate human-annotated benchmark (750 pointwise annotations over 250 videos, secondary). — [arXiv 2608.21839](https://arxiv.org/pdf/2608.21839)
- **BrokenVideos** (Jun 2025): human artifact masks made with SAM2 propagation and frame-by-frame refinement over 3,254 videos. The training set adds 748 clean "normal" videos; the validation set is 300 videos, half broken and half clean. — [arXiv 2506.20103](https://arxiv.org/abs/2506.20103)
- **Seedance 1.0** (Jun 2025): human pairwise labels on video pairs, including outputs from different training stages of their own model. — [arXiv 2506.09113](https://arxiv.org/pdf/2506.09113)
- **Rapidata-style crowdsourcing** (2025–2026):
  - Example scale: a Veo 3.1 text-to-video preference set has ~74k responses from ~23k annotators.
  - Formats: pairwise questions (e.g., "Which video do you prefer aesthetically?") and Likert ratings for coherence/plausibility, prompt alignment and style.
  - Rater weighting: Likert scores are aggregated "based on the evaluators' responses... as well as their userScore".
  - Per-response demographics (age, country, gender, language, occupation) are recorded.
  - Datasets are collected in about 1 hour through Rapidata's API.
  — [Rapidata Veo 3.1 dataset card (mirror)](https://gitcode.com/hf_mirrors/Rapidata/text-2-video-human-preferences-veo3.1/blob/main/README.md); [Rapidata HunyuanImage-2.1 preference card](https://huggingface.co/datasets/Rapidata/HunyuanImage-2.1_t2i_human_preference/blob/c8a2675e752d9e4c0036f4be4e4d933e07f4c1c6/README.md); [Rapidata Sora style-Likert card (mirror)](https://hf-proxy-2.923828430.workers.dev/datasets/Rapidata/sora-video-generation-style-likert-scoring/blob/886ca598df7d9fa76442d48f21c9b30701333793/README.md)
- **Annotator screening** (RLHF precedent, 2022): InstructGPT labelers were screened on how well their rankings agreed with researcher labels (secondary). — [summary](https://papermap.vercel.app/paper/Training_Language_Models_to_Follow_Instructions_with_Human_Feedback.html)
- **Synthetic and AI labels as a supplement**: reward-model-labeled pairs (Syn-Pic/RankDPO, ~$200 vs ~$50K for human labels), and GPT-labeled segment preferences that come close to human labels (DenseDPO). — [RankDPO](https://arxiv.org/pdf/2410.18013); [DenseDPO](https://arxiv.org/pdf/2506.03517)

### Inferences
- **Build an internal gold evaluation set**, separate from in-product feedback.
  - 300–1,000 shot-level pairs drawn across the five hosted models and across our target looks (photoreal people, photographic-surreal integration, multi-shot character continuity).
  - Label each with ≥3 trained raters, with per-dimension A/B/tie for our 7 dimensions.
  - Consolidate by majority vote or mean, and keep the full vote vector so the evaluation can be uncertainty-aware (HPSv3).
- **Prompt mix.** Combine real user stories/prompts (like VidProM) with deliberately hard categories: multi-action, on-screen text, camera motion (VideoFeedback2), plus app-specific ones. App-specific stress cases include surreal object integration, crowd scenes, hands and faces in close-up, and continuity across adjacent shots.
- **Pair construction.** Use cross-model pairs (VideoAlign used 12 models, VideoFeedback2 used 22) to learn routing preferences. Use same-model, same-prompt, different-seed pairs for fine-grained quality. For motion fairness, pair takes with similar motion magnitude, or use segment-level labels, to avoid DenseDPO's low-motion bias.
- **Add real live-action footage as anchors** for the photorealism and "anti-3D-render" dimension. HPDv3 deliberately included real images of varying quality, so a "real vs generated" contrast helps calibrate a photoreal scale.
- **Write the annotation guide as a binary checklist per dimension**, VisionReward/FIRM-Video style. Examples: "skin reads as plastic/CG?", "lighting has render-like falloff?", "subject identity matches reference?", "camera move matches storyboard?". Ask raters to point to the time and region that justifies each "no".
- **Track rater quality:**
  - Hidden gold pairs with obvious answers.
  - Repeated pairs with left/right swapped.
  - Per-rater agreement with consensus, used as a userScore-style weight.

### Gaps
- Written annotation guidelines and the tie rates actually observed in VideoAlign, HPDv3 and VideoFeedback2 were not retrievable (fetch blocked).
- Rapidata does not document how its userScore is computed, how it vets annotators, or how its labels compare with expert labels.
- Licenses for Pick-a-Pic, MHP and VisionReward data were not confirmed; HPDv3's CC BY-NC-SA comes from a secondary source.

## 4. How to handle noisy, sparse, biased or adversarial feedback (reliability weighting, BT/Thurstone/Elo, ties, position bias, fatigue, cold start, active learning)?

### Takeaway
- **Aggregation:** Bradley–Terry (or BT-with-ties) maximum-likelihood estimation with bootstrap confidence intervals is the standard, rather than online Elo, which depends on match order. Per-annotator reliability (Crowd-BT) and per-annotator position-bias terms are well-established extensions.
- **Biases specific to video:** besides left/right position, annotators favor low-motion clips (DenseDPO) and, at a glance, denser or more polished outputs (Bansal, for text).
- **Sparsity:** organic feedback is extremely sparse (~0.1% of messages get a Love reaction).
- **Statistical power:** small preference margins need hundreds of judgments (≥500 for a margin ≤0.10 at 90% power).
- **Active learning:** query selection by information gain or optimal experimental design (Hybrid-MST, ED-PBRL 2025, adaptive elicitation 2026) is the main lever for asking fewer, better questions.
- **Reward hacking** is the dominant failure mode when the cleaned signal is used for optimization. Multi-dimensional rewards, regularization and held-out human checks are the mitigations.

### Cited Findings
**Aggregation models**
- Chatbot Arena moved from online Elo to Bradley–Terry MLE because BT "gives significantly more stable ratings". Online Elo shifted substantially when the same matches were replayed in reverse order. BT's bootstrap confidence intervals capture variance better than the earlier approach, which reused the same samples for its intervals. — [LMSYS blog (Dec 2023)](https://lmsys.org/blog/2023-12-07-leaderboard); [Chatbot Arena paper](https://arxiv.org/pdf/2403.04132)
- Ties: VideoAlign's Bradley–Terry-with-ties model keeps tie labels instead of discarding them, for better robustness and accuracy. — [VideoAlign](https://arxiv.org/html/2501.13918v2)
- Lee (Jan 2026) analyzed with ties discarded (conditioning on decisive judgments) and called that a "best-case analysis". — [arXiv 2601.09084](https://arxiv.org/html/2601.09084v1)
- HPSv3 trains with an uncertainty-aware ranking loss over 9–19 annotator votes per sample. — [HPSv3](https://arxiv.org/html/2508.03789v1)
- Crowd-BT (Chen, Bennett, Collins-Thompson & Horvitz, WSDM 2013):
  - Extends BT to estimate each annotator's reliability jointly with the global ranking, separating reliable labelers from spammers.
  - Actively selects both the pair and the annotator, using an online Bayesian update to balance exploration and exploitation.
  - Drawback: modeling annotators this way is computationally expensive. Hybrid-MST offers a cheaper batch-mode active sampler.
  - Crowd-BT is still used as a 2026 baseline, with "virtual annotator" regularization.
  — [Microsoft Research: Crowd-BT](https://www.microsoft.com/en-us/research/?p=165700); [Horvitz-hosted PDF](https://www.erichorvitz.com/crowd_pairwise.pdf); [Hybrid-MST (arXiv 1810.08851)](https://arxiv.org/pdf/1810.08851); [Aggregate-then-Calibrate (arXiv 2608.02455)](https://arxiv.org/pdf/2608.02455)
- Rapidata weights Likert aggregation by each evaluator's userScore. Artificial Analysis restricts audio-video Elo to "validated private evaluators". — [Rapidata card](https://gitcode.com/hf_mirrors/Rapidata/text-2-video-human-preferences-veo3.1/blob/main/README.md); [Artificial Analysis](https://artificialanalysis.ai/video/leaderboard/image-to-video)

**Position and other systematic biases**
- Crowdsourced ranking models treat left/right preference as a per-annotator effect. One paper flags "careless" annotators who click one side regardless of content, including bots posing as humans. Another estimates a position-bias parameter per worker to identify biased workers. — [arXiv 1605.05860](https://www.arxiv.org/pdf/1605.05860); [arXiv 1804.11177](https://arxiv.org/pdf/1804.11177); [Aggregation with bias reduction (factorBT)](https://ar5iv.labs.arxiv.org/html/1906.03711)
- Image-evaluation protocols counter position bias in two ways. One randomizes left/right order, offers a tie option, and reports quadratic-weighted Cohen's kappa. GenAI-Bench shuffles candidate order on every comparison. — [arXiv 2601.04946](https://arxiv.org/pdf/2601.04946); [EditReward (arXiv 2509.26346)](https://arxiv.org/pdf/2509.26346)
- LLM judges show measurable position bias. Methodology for position consistency and preference fairness is in an ICLR 2025 study. — [arXiv 2406.07791](https://arxiv.org/html/2406.07791v4)
- Video-specific: vanilla video preference collection biases annotators toward low-motion clips. — [DenseDPO](https://arxiv.org/pdf/2506.03517)
- Text analogue: annotators rate denser responses higher in absolute ratings. — [Bansal et al.](https://huggingface.co/papers/2308.15812)

**Sparsity, statistical power and fatigue**
- Love reactions occur on ~0.1% of messages. — [RLUF](https://arxiv.org/html/2505.14946v1)
- Lee (2026), statistical power:
  - 17.6% of model pairs with ≥200 judgments have preference margins |δ| ≤ 0.10, which needs ≥500 judgments for 90% power.
  - Required sample size scales as δ⁻².
  - Adaptive prompt allocation does not improve detectability when the protocol is fixed.
  - Recommendation: run a pilot of ~30–100 judgments to estimate the effect size before a full study.
  - Inconclusive human evaluations often reflect underpowered studies, not model equivalence.
  — [arXiv 2601.09084](https://arxiv.org/html/2601.09084v1); [review](https://www.themoonlight.io/review/how-many-human-judgments-are-enough-feasibility-limits-of-human-preference-evaluation)
- Fatigue evidence is mixed:
  - Hata et al. (CSCW 2017), 9M Mechanical Turk annotations: worker quality was "extremely stable" over the long term, and workers self-selected out of tasks they couldn't do well. — [arXiv 1609.04855](https://ar5iv.labs.arxiv.org/html/1609.04855)
  - A 2021 study: quality deteriorated as online workers worked longer, but timely feedback on quality scores maintained it. — [arXiv 2101.04615](https://arxiv.org/abs/2101.04615v1)
  - A Bing study (CHIIR 2022): fatigue, time-of-day and anchoring effects in relevance labels. — [Microsoft CHIIR'22](https://microsoft.com/en-us/research/uploads/prod/2022/01/chiir22-26.pdf)

**Active learning (which pairs or takes to ask about)**
- Approaches to choosing which pairs to label:
  - Classical: expected-information-gain query selection combining MOS-style ratings and pairwise comparison (CVPR 2014). — [Ye & Doermann](https://openaccess.thecvf.com/content_cvpr_2014/papers/Ye_Active_Sampling_for_2014_CVPR_paper.pdf)
  - A taxonomy of pair selection: random, sorting-based, and active (BT posteriors with KL-divergence utility). — [Mohammadi & Ascenso (2023)](https://ar5iv.labs.arxiv.org/html/2311.06093)
- Newer work aimed at generative models:
  - ED-PBRL (Dec 2025) uses optimal experimental design to choose preference queries for personalizing text-to-image diffusion, needing fewer queries than random selection (secondary). — [arXiv 2512.19057](https://arxiv.org/pdf/2512.19057)
  - Adaptive Prompt Elicitation (Feb 2026) applies information-theoretic query selection in an interpretable visual feature space. — [arXiv 2602.04713](https://arxiv.org/pdf/2602.04713)

**Reward hacking when the signal is optimized (2025–2026 video RL)**
- Fixed, single-dimensional rewards in video GRPO are "prone to reward hacking". Under sustained optimization the score loses fidelity as a quality proxy (Goodhart's law). — [Rethinking Reward Signals in Video GRPO (arXiv 2511.19356)](https://arxiv.org/pdf/2511.19356)
- A 2026 survey of hacking types includes object deformation, such as distorting text to raise OCR scores. — [Advances in GRPO for Generation Models: A Survey (arXiv 2603.06623)](https://arxiv.org/pdf/2603.06623)
- GRPO-Guard traces FlowGRPO's implicit over-optimization to gradient imbalance across noise levels and mitigates it. — [GRPO-Guard (arXiv 2510.22319)](https://ar5iv.arxiv.org/html/2510.22319)
- Engagement-reward hacking in production: RLUF's repetitive "Sending Love!" sign-offs, and a trade-off against helpfulness. — [RLUF](https://arxiv.org/html/2505.14946v1)

### Inferences
- **Aggregation stack:**
  - Per-dimension BT-with-ties (Rao–Kupper/Davidson-style) over our pairs.
  - A hierarchical model with: a global take/model/prompt-template utility, per-user random effects, a per-user left/right bias term, and per-user reliability (Crowd-BT-style).
  - Bootstrap confidence intervals for any decision: model routing per shot type, prompt-template changes, judge-version upgrades.
  - Never make decisions from a running Elo.
- **Order and exposure hygiene:**
  - Randomize which take appears left or right, and log it.
  - Occasionally re-show a pair swapped to measure each user's consistency.
  - Detect users whose picks track screen position, and down-weight them.
  - Apply the same swap test to the VLM judge: score both orders and treat disagreement as uncertainty.
- **Motion-bias guard:** compare takes with similar motion magnitude, or label motion separately. Otherwise both users and the judge will reward static clips, which is exactly the failure DenseDPO documents.
- **Sparsity and cold start:**
  - Start from the judge plus global priors.
  - Add organic implicit signals.
  - Ask explicit A/B questions only when they are informative: judge scores for the two takes are close or uncertain, the judge and the user's implicit behavior disagree, or a routing or template decision is under-powered.
  - Cap explicit asks per session (e.g., at most 1 ask per N generations, never during playback) and make them skippable.
- **Power planning:** for global decisions (e.g., "Veo 3.1 vs Kling 3.0 for handheld street shots"), pool across users. Expect ≥500 decisive judgments when the preference margin is small (|δ| ≤ 0.10 as defined by Lee 2026); run 30–100 judgment pilots first.
- **Adversarial and low-quality feedback:**
  - Hidden "sanity" pairs with obvious answers (e.g., a corrupted take vs a clean one).
  - Rate limits.
  - Per-account weight caps, so one prolific user can't dominate the global model.
  - Kept separate from per-user personalization, where a user's own taste should dominate.
- **Optimization safety:** use the cleaned feedback to tune the prompt policy, the routing and judge calibration, not one scalar. Keep the photoreal/anti-CG dimension as a hard constraint, and hold out human A/B checks to catch Goodhart drift. Two drift patterns to watch: over-sharpened "AI look" winning quick glances, and motion collapse.

### Gaps
- No quantified human left/right position-bias magnitude was found for image or video A/B interfaces.
- No study was found on within-session fatigue for *end users* of creative tools, as opposed to paid crowd workers. The appropriate frequency of feedback prompts in creative apps is undocumented.
- No published comparison of Thurstone vs BT for generative-media preferences was found. BT/BTT dominates current practice.

## 5. Personalization: per-user/per-project preference models vs a global model; how products handle taste diversity

### Takeaway
Products and research converge on a shared global quality model plus lightweight personal layers:
- **Midjourney:** personalization profiles built from pairwise ratings and likes, multiple profiles, shareable codes, moodboards.
- **Sora:** steerable feed ranking.
- **Research, three routes:** few-shot user embeddings (PPD: 4 pairwise examples → 76% win rate; PAL: latent-mixture rewards that generalize to unseen users), natural-language preference extraction (ViPer, CIPHER), and active query design (ED-PBRL).

For an LLM-planned pipeline, the interpretable natural-language route (per-project "style memory" fed into Claude's prompts) is the most direct fit, because it needs no model training and the user can edit it.

### Cited Findings
- Midjourney:
  - Personalization comes from Rank Pairs and Explore likes. Users can apply someone else's personalization code but cannot update it. — [Midjourney docs](https://docs.midjourney.com/docs/personalization); [Dataconomy](https://dataconomy.com/2024/06/12/midjourney-model-personalization-guide)
  - The V7 "For You" feed is based mostly on the user's global V7 personalization profile, alongside Moodboards and Tasks for tailoring profiles. — [Midjourney docs: Website Overview](https://docs.midjourney.com/docs/explore-page)
  - Faster personalization and moodboard support were added later. — [The Decoder](https://the-decoder.com/midjourney-adds-faster-model-customization-and-mood-board-support/)
- Sora: "steerable ranking" lets users tell the feed what they're in the mood for, and parents can turn off feed personalization for teens. — [OpenAI](https://openai.com/index/sora-feed-philosophy); [gend.co](https://www.gend.co/blog/sora-feed-philosophy)
- ViPer (EPFL, ECCV 2024):
  - Users comment on a small set of images; a VLM extracts liked and disliked attributes; generation is steered without any training.
  - A parameter β sets how strongly to personalize.
  - User studies (secondary): personalized images were picked over non-personalized 86.1% of the time, and over images personalized for *other* users 65.4% of the time.
  — [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9492_ECCV_2024_paper.php); [project page](https://viper.epfl.ch); [Emergent Mind](https://www.emergentmind.com/papers/2407.17365)
- PPD, Personalized Preference Fine-tuning (Stanford, CVPR 2025):
  - Diffusion-DPO optimizes a single population-level reward, which the authors argue is insufficient for personal alignment, and one model per user doesn't scale.
  - Instead, a VLM (LLaVA-OneVision) embeds a few of a user's pairwise preferences, and Stable Cascade is conditioned on that embedding through cross-attention.
  - With as few as 4 examples from a new user it reaches a 76% average win rate over Stable Cascade (judged by a VLM). It was trained on Pick-a-Pic.
  - The authors note there are no automatic metrics for individual alignment, and that extension to video is open.
  — [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Dang_Personalized_Preference_Fine-tuning_of_Diffusion_Models_CVPR_2025_paper.pdf); [arXiv 2501.06655](https://arxiv.org/abs/2501.06655v1)
- PAL (ICLR 2025):
  - Uses a latent ideal-point/mixture model that shares structure across users while localizing preferences for new users from few samples.
  - On Pick-a-Pic v2: 2.5% more accurate than the best prior method with 156× fewer learned parameters.
  - On Reddit TL;DR: 36% more accurate for *unseen* users.
  — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2858f8c8683aaa8c12d487354cf328dc-Abstract-Conference.html)
- MPS conditions a single reward model on which preference dimension to score. This lets one model weight dimensions differently, a building block for per-user dimension weights. — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Learning_Multi-Dimensional_Human_Preference_for_Text-to-Image_Generation_CVPR_2024_paper.html)
- CIPHER learns per-context natural-language preferences from edits, retrieves them by context similarity, and lets users read and edit them. — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/aligning-llm-agents-by-learning-latent-preference-from-user-edits/)
- ED-PBRL (Dec 2025) targets user-specific style personalization of text-to-image diffusion with fewer, optimally designed preference queries. — [arXiv 2512.19057](https://arxiv.org/pdf/2512.19057)
- Fixed, single-dimensional rewards in video RL are "prone to reward hacking", and under sustained optimization the score loses fidelity as a quality proxy (Goodhart's law). — [Rethinking Reward Signals in Video GRPO (arXiv 2511.19356)](https://arxiv.org/pdf/2511.19356)

### Inferences
- **Two-tier model:**
  - *Global, non-negotiable product standards*, learned from everyone: photoreal / anti-3D look, artifact-free motion, continuity, prompt alignment.
  - *Personal and per-project taste*: color grade, lens and framing habits, pacing, motion energy, degree of surrealism, film-stock look.
  - Global dimensions act as constraints or gates. Taste dimensions are re-weighted per user and project, with MPS-style dimension conditioning.
- **Personalization layer 1 (ship first): a per-project "Style Memory".**
  - Claude maintains short natural-language preference statements extracted from critiques, edits and A/B reasons, as in CIPHER and ViPer.
  - Each statement carries its source events and a confidence.
  - The statements are injected into planning and shot prompts.
  - The user can view, edit or delete them. That gives transparency and a way to fix a wrong inference.
- **Layer 2 (later): learned user/project embeddings** once there is enough data. PPD shows gains from ~4 pairwise examples; PAL shows few-shot generalization to new users. Use them to re-rank takes and pick routing, not to fine-tune hosted video models.
- **Onboarding:** an optional short "taste calibration" of ~10–20 A/B pairs of photoreal stills/clips per project genre. Midjourney's ~200-pair gate works for an engaged audience, but few-shot results (PPD) suggest far fewer pairs are useful when the picks are well chosen (ED-PBRL/active selection).
- **A "lean into my style" control**, like ViPer's β, plus multiple profiles per user (Midjourney-style) mapped to projects. This handles taste diversity across a user's projects, e.g., a noir short vs a sunlit surreal ad.
- **Separation of data flows:** a user's own feedback should dominate their personalization. For the global model, cap each account's influence and pool across users, as in §4.

### Gaps
- No personalization study for *video* generation was found; PPD explicitly lists it as open. No product documents per-project, as opposed to per-user, preference models.
- No public data on how many pair ratings Midjourney currently requires, or whether the V8-era flow differs from V7's ~200 (the latter is secondary, Apr 2025).

## 6. Data governance: consent, privacy, licensing of user prompts/outputs for training, retention, opt-out

### Takeaway
The market splits three ways:
- **No training on customer content:** Adobe Firefly.
- **Train by default, with exemptions for business/API/enterprise use:** Luma, Runway, OpenAI (business/API excluded by default), Anthropic's commercial API (no training on customer content).
- **Broad perpetual licenses over consumer content:** Midjourney, Pika (explicitly including "user interactions"), Kling, CapCut/Dreamina.

Two practices matter most for a feedback layer:
- Feedback submission can override training opt-outs (OpenAI), and human-reviewed feedback outlives deletion (Google keeps reviewed data separately, up to 3 years for NotebookLM feedback). The app's consent design must say whether a thumbs-up counts as consent to training.
- Several hosted-model terms forbid using their outputs to train third-party or competing models (Kling, Luma, Runway). That may constrain training our own judge or reward model on generated takes.

In the EU, EDPB Opinion 28/2024 permits legitimate interest for model training only after a documented 3-step test. AI Act Art. 50 marking and deepfake-disclosure duties apply from Aug 2, 2026.

### Cited Findings
- OpenAI:
  - Sora has its own training toggle in Sora Settings, the privacy-portal opt-out also covers Sora, and the ChatGPT toggle does not carry over to Sora.
  - Thumbs feedback may lead to "the entire conversation" being used for training even when the user has opted out (ChatGPT).
  - Business tiers and the API are not trained on by default.
  — [OpenAI: How your data is used](https://openai.com/policies/how-your-data-is-used-to-improve-model-performance/); [OpenAI Help Center](https://help.openai.com/en/articles/5722486-how-do-i-opt-out-of-my-content-being-used-to-improve-model-performance)
- Google Flow (May 19, 2026 notice):
  - Data is de-linked from the account before human review.
  - Reviewed or annotated data is retained separately even after deletion or after turning off "Help improve Flow".
  - Deleted data may persist up to 72 hours.
  — [labs.google/flow/privacy](https://labs.google/flow/privacy)
- Google feedback retention, other products: NotebookLM/Gemini Notebook feedback is retained up to 3 years, with Workspace/Education excluded from human review. — [Google support](https://support.google.com/notebooklm/answer/17004255)
- Google Labs general data is stored up to 18 months (news report). — [AOL](https://www.aol.com/googles-ai-filmmaker-program-flow-153000467.html)
- Adobe: does not train Firefly on customer content; uses licensed and public-domain training data; codified this pledge in June 2024 after the terms-of-use backlash. — [Adobe AI ethics](https://www.adobe.com/sg/ai/overview/ethics/link-free.html); [Axios](https://www.axios.com/2024/06/18/adobe-ai-customer-data-training)
- Midjourney: public and remixable by default under a license to Midjourney; Stealth Mode is a "best efforts" non-publication promise for Pro/Mega. — [Midjourney ToS](https://docs.midjourney.com/docs/terms-of-service)
- Runway (third-party analyses):
  - The training license is irrevocable, perpetual, worldwide, royalty-free and sublicensable for non-Enterprise plans.
  - Enterprise terms are separate.
  - Outputs may not be used to train competing models without written consent.
  — [ConductAtlas: Runway](https://conductatlas.com/platform/runway/runway-terms-of-service/ai-training-license-over-inputs-and-outputs/); [ConductAtlas: Runway usage policy](https://conductatlas.com/platform/runway/runway-usage-policy/prohibition-on-using-outputs-to-train-competing-ai-models/); [Runway Enterprise Terms](https://runway.com/enterprise-terms)
- Pika (via tracker): inputs, outputs *and user interactions* may be used for training. A separate tracker entry covers a training license over voice and likeness inputs. — [ConductAtlas: Pika training](https://conductatlas.com/platform/pika/pika-terms-of-service/pika-may-use-content-to-train-ai-models/); [ConductAtlas: Pika voice/likeness](https://conductatlas.com/platform/pika/pika-terms-of-service/ai-training-data-license-on-user-inputs-including-voice-and-likeness/)
- Luma:
  - Business/API inputs and outputs are not used for training. — [Luma API terms](https://lumalabs.ai/api/terms)
  - Consumer content may be. — [Luma privacy policy](https://lumalabs.ai/learning-hub/privacy-policy)
  - Trackers report two further points: account deletion does not revoke the training license, and outputs may not be used to train competing models. — [ConductAtlas: Luma deletion](https://conductatlas.com/platform/luma-ai/luma-ai-terms-of-service/account-deletion-does-not-revoke-ai-training-license/); [ConductAtlas: Luma competing models](https://conductatlas.com/platform/luma-ai/luma-ai-terms-of-service/prohibition-on-using-output-to-train-competing-ai-models/)
- Kling (trackers, low confidence):
  - Kling may train on Input.
  - Users may not use Output or platform data to train third-party AI models (Sec. 3.1(w)).
  - Opt-out or revocation is by email.
  — [ConductAtlas: Kling](https://conductatlas.com/platform/kling-ai/kling-ai-terms-of-service/); [VerifyWise](https://verifywise.ai/ai-trust-index/kling-ai)
- CapCut (June 2025): a perpetual, irrevocable, sublicensable license over user content. Dreamina's terms were updated Jan 22, 2026. — [Techloy](https://www.techloy.com/capcuts-latest-terms-of-service-raises-big-questions-about-content-ownership/); [Dreamina ToS](https://dreamina.capcut.com/clause/dreamina-terms-of-service)
- Anthropic Commercial Terms: "Anthropic may not train models on Customer Content from Services". API retention figures conflict across secondary sources (30 days vs 7 days from Sep 2025). Zero data retention is available by arrangement. — [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms); [secondary: developersdigest](https://www.developersdigest.tech/models/data-retention/anthropic)
- EDPB Opinion 28/2024 (adopted Dec 17, 2024):
  - Whether an AI model is anonymous is assessed case by case.
  - Legitimate interest (Art. 6(1)(f)) can in principle support model development and deployment, subject to the three-step test (interest, necessity, balancing) and mitigating measures.
  — [CMS](https://cms.law/en/int/legal-updates/edpb-opinion-28-2024-key-takeaways-on-processing-personal-data-in-the-context-of-ai-models); [Hunton](https://huntonak.com/privacy-and-information-security-law/edpb-publishes-opinion-on-processing-of-personal-data-in-the-context-of-ai-models)
- EU AI Act Art. 50 (applies from Aug 2, 2026):
  - Providers of generative systems must mark outputs in machine-readable form. Per Commission guidance as described by commentators, systems already on the market get until Dec 2, 2026.
  - Deployers must disclose deepfakes, with lighter duties for artistic, creative, satirical or fictional works.
  - Fines reach up to €15m or 3% of turnover.
  - A pending "AI Omnibus" may alter the watermarking deadline (unverified).
  — [didit](https://didit.me/blog/eu-ai-act-deepfake-labelling-2026/); [SRD Rechtsanwälte](https://www.srd-rechtsanwaelte.de/en/blog/deepfake-labelling-under-the-ai-act-what-the-new-eu-guidance-clarifies); [CSA research note (Jul 2026)](https://labs.cloudsecurityalliance.org/wp-content/uploads/2026/07/CSA_research_note_eu_ai_act_article_50_transparency_20260729-csa-styled-1.pdf)
- Dataset licensing: VideoFeedback2 is Apache-2.0; HPDv3 is reportedly CC BY-NC-SA (non-commercial; secondary). — [VideoFeedback2 card](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback2/blob/main/README.md); [Emergent Mind: HPSv3](https://www.emergentmind.com/topics/human-preference-score-v3-hpsv3)
- Crowdsourced preference datasets can carry per-rater demographics (Rapidata records age, country, gender, language and occupation per response). — [Rapidata card](https://gitcode.com/hf_mirrors/Rapidata/text-2-video-human-preferences-veo3.1/blob/main/README.md)

### Inferences
- **Consent design.** Use two separate, plainly worded toggles:
  - (a) "Use my feedback to personalize *my* results". Needed for the product to work; per-project scope.
  - (b) "Allow my prompts, outputs and feedback to improve the app's shared models".
  Feedback widgets must state clearly whether clicking them counts as consent under (b). Copying OpenAI's "thumbs overrides opt-out" pattern silently would be a trust risk. Make the default jurisdiction-aware: consent-based in the EU unless a documented EDPB 3-step legitimate-interest assessment exists.
- **Storage architecture:**
  - An append-only event log of feedback events, keyed by pseudonymous user and project IDs, with a full context snapshot: prompt version, storyboard frame, model and version, seed, judge version and scores, UI position, exposure time.
  - Derived preference tables are rebuilt from the log, so a deletion or opt-out propagates by recomputation.
  - Content assets are kept separately with their own retention.
  - Human-reviewed samples are stored de-identified (the Google pattern), with disclosed retention.
  - Retention windows are explicit. For reference: Google keeps deleted data up to 72 hours, Labs data up to 18 months and feedback up to 3 years. Set our own windows shorter where possible.
- **Hosted-model terms check (legal review needed before any training).** Kling (Output/platform data may not train third-party models), Luma and Runway (no training of competing models) suggest that fine-tuning our own judge, reward model or a video model on their outputs could breach consumer terms. API or enterprise terms may differ and must be checked. Storing outputs for evaluation, and training prompt policies or routers from *preference labels*, are lower risk but should still be confirmed. Do not fine-tune on research datasets with non-commercial licenses (HPDv3 reportedly CC BY-NC-SA).
- **The Claude planner:** under Anthropic's commercial terms, user stories and critiques sent through the API are not used to train Claude. Confirm the retention window (sources conflict: 30 vs 7 days) and whether zero data retention is needed for enterprise customers.
- **Outputs and likeness.** Photoreal output is the product goal, so plan for AI Act Art. 50 machine-readable marking (e.g., C2PA-style provenance) and deepfake-disclosure UX, noting the reduced duties for fictional or artistic works. Require consent records for any real-person likeness or character references that enter feedback data (Pika's voice/likeness controversy is a cautionary example).
- **Minimize demographics.** Don't collect them in in-product feedback. If external crowd raters are used, set the vendor's demographic fields to the minimum needed.

### Gaps
- Official Kling, Runway and Pika terms could not be read directly; only third-party trackers, some marked low-confidence. API-specific terms for Seedance (BytePlus/ModelArk), Veo (Vertex AI/Gemini API), Kling API, Sora 2 API and Wan were not researched. They determine whether outputs from our hosted-model calls may be used to train our own judge or reward models.
- Whether Midjourney Stealth Mode excludes content from training could not be verified. Secondary sources claim it only hides images from the gallery.
- The final Commission guidance on Art. 50 and the status of the AI Omnibus could not be confirmed from primary EU sources.
- Anthropic API retention figures conflict across secondary sources and need checking against Anthropic's own data-retention documentation.
