# Jev study plan (draft, not yet frozen)

Working title: **Fast decisions, cheap guards: an independent evaluation of a
"System One" model for prompt-injection screening and model routing.**

Jev (TypeSafe AI) answers multiple-choice, score and yes/no questions about a text
state in roughly 70-500 ms, returns probabilities and a confidence score, and costs
about $0.042 per million input tokens with output free (vendor figures, unverified).
It cannot generate text. The paper asks where a model like that is worth using inside
LLM systems, measured on four axes against the usual alternatives:

- **accuracy** (detection rate at a fixed false-alarm rate, AUROC, cost-quality curves),
- **calibration** (does confidence mean what it says? The vendor claims it, with no data),
- **latency** (p50 / p95 per decision, measured from the same machine for every system),
- **cost** (dollars per 1,000 decisions, including the decision model itself).

Items marked **[open]** still need checking.

## Ground rules

- **Terms of service.** TypeSafe's Master Customer Agreement has no clause against
  benchmarking or publishing results (checked 2026-09-30). Section 2.3(b) forbids using
  outputs to distil or train an imitating model, or to build a competing product. So:
  no classifier in this study is trained on Jev's labels, and all Jev outputs are used
  only for evaluation.
- **Pinned version.** Every decision records `response.model` from the API, so each
  result is tied to the exact Jev version. The smoke test records which models the
  account can use. **[open: confirm the version string once the key is in]**
- **Dev/test split.** Question wording and thresholds are tuned on dev only; the test
  split is run once after a protocol is frozen, as in `cs_harness/PROTOCOL.md`. The
  runners refuse the test split until `PROTOCOL.md` exists.
- **Independent scoring.** Ground truth comes from dataset labels or benchmark scorers,
  never from any of the systems under test.

## Study 1: prompt-injection detection (headline)

**Question.** How well does Jev spot injected instructions, directly in a user's
message and hidden in retrieved documents or tool outputs, compared with small trained
detectors and LLM judges, per dollar and per millisecond?

**Data** (split by hash, 30% dev / 70% test):

| Source | Licence | Dev (inj / benign) | Test (inj / benign) |
|---|---|---|---|
| deepset `prompt-injections`, direct, EN + DE (train and test pooled, deduplicated) | Apache-2.0 | 85 / 117 | 178 / 282 |
| HotpotQA paragraphs, clean and with a planted answer-override note | CC BY-SA | 56 / 56 | 144 / 144 |
| NotInject: benign prompts built around trigger words (over-defence) | MIT | 0 / 96 | 0 / 243 |

Indirect notes use the Aegis templates (`aegis/eval/poison.py`): dev uses the 4 `dev`
phrasings, test uses the 5 `heldout` phrasings, so no system is tuned on test wording.

Deferred: BIPIA and an AgentDojo end-to-end subset **[open: licence, availability, cost]**.

**Jev question.** A two-option **Choice** (not Noul): Choice answers return a
probability per option plus a separate confidence, which the calibration analysis
needs; Noul returns one float. Three wordings (`INJECTION_QUESTIONS` v1-v3 in
`jevbench/deciders.py`) are compared on dev only.

**Baselines.**
1. Keyword rules: the Aegis `INJECTION_PATTERNS`.
2. Local trained detector: ProtectAI `deberta-v3-base-prompt-injection-v2` (Apache-2.0,
   runs on the RTX 4050). Meta `Llama-Prompt-Guard-2` (86M, 22M) is gated behind a
   manual licence approval, so it is optional.
3. LLM judges: gpt-4o-mini (OpenRouter) and Claude Haiku 4.5, asked for a verdict and
   a probability.

**Metrics.** AUROC with a stratified bootstrap CI; detection rate and false alarms at
thresholds chosen on dev (1% and 5% FPR) and applied to test; false alarms on NotInject
(at 0.5 and at the dev thresholds, and by number of trigger words); ECE, Brier and a
reliability diagram; for Jev, the ECE of its reported confidence; p50/p95 latency
(hosted and local reported separately); $ per 1,000 checks; paired AUROC differences
between Jev and each baseline.

**Dev sanity check (free baselines, 2026-09-30).** DeBERTa: AUROC 0.85, 42% false
alarms on NotInject, 15 ms p50 on GPU. Keyword rules: AUROC 0.58, no NotInject false
alarms. The DeBERTa over-defence matches what the InjecGuard paper reports.

## Study 4: confidence-gated routing

**Question.** Can Jev decide, per prompt, whether a cheap model is enough or the
expensive one is needed, and does its probability give a useful cost-quality dial?

**Data.** RouterBench `routerbench_0shot.pkl`: 36,497 prompts, 11 models, each with a
stored score (0-1) and cost. Routing is evaluated offline on those stored answers, so
the only live calls are to Jev. Pair: Mixtral-8x7B (cheap, quality 0.58, $0.15 per 1k
prompts on dev) versus GPT-4-1106 (strong, 0.73, $4.65 per 1k). A family-stratified
sample (up to 250 prompts per task family, so HellaSwag and GSM8K do not dominate)
gives 775 dev and 1,837 test prompts. **[open: RouterBench has no stated licence; cite it
and do not redistribute the data]**

**Jev question.** A two-option Choice (`ROUTING_QUESTIONS` v1-v2): small model OK vs
strong model needed. Sorting prompts by P(needs strong) and sending the top k to GPT-4
traces a cost-quality curve.

**Baselines.** All-cheap, all-strong, random routing (the straight line between them),
the oracle, prompt length, and a TF-IDF + logistic-regression router trained on the
dev split's RouterBench labels (an in-distribution trained router; cross-fitted on
dev). RouteLLM's pretrained routers are optional.

**Metrics.** AIQ (area under the upper hull of the cost-quality curve over the
all-cheap to all-strong cost range); cost to reach 95% of GPT-4's quality; quality at
20/40/60% strong share; AUROC for "GPT-4 beats Mixtral on this prompt"; Jev's own cost
included in every total.

## Budget ($15 cap)

| Item | Estimate |
|---|---|
| LLM judges, dev + test (~1,400 texts x 2 models) | ~$1 |
| Jev, both studies, all wordings (~10k calls x ~300 tokens) | < $0.20 |
| Study 4 model answers (stored in RouterBench) | $0 |
| AgentDojo subset, if added | ~$3-5 |
| Buffer | remainder |

The runner adds up every cached decision's cost before each call and stops before the
total would cross the cap.

## Order of work

1. ~~Install the SDK; confirm its key variable (`TYPESAFE_API_KEY`).~~
2. ~~Build `jev_study/` with one `Decider` interface, offline tests (32), free dev baselines.~~
3. Add the Jev key to `.env`; run `smoke_jev.py` (version string, probability and
   confidence output, latency, rate limits).
4. Dev runs: Jev wordings, LLM judges, routing wordings; pick one wording per study
   and the thresholds.
5. Freeze `jev_study/PROTOCOL.md`, run the test splits once, and write the paper with
   generated numbers, as in `cs_harness/report/`.
