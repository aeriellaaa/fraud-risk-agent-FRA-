# Fraud/Risk Flag Agent — Razorpay AI Buildathon 2026, Track 02: AI Risk Manager

A three-agent fraud detection system for merchant transactions. Instead of a single model
producing a score that gets acted on directly, this system adds an independent **Reviewer
Agent** that checks whether a score's confidence is actually earned by its evidence before
any auto-decision is made — and reports a measured, India-specific cost model behind the
decision threshold, not just an accuracy number.

---

## The problem with "train a classifier, report accuracy, done"

Most fraud-detection prototypes stop at a single model and a headline accuracy figure. Two
things get skipped almost universally, and both are explicitly part of Track 02's grading bar:

1. **The decision threshold is treated as an afterthought**, left at the model's default
   (usually 0.5) instead of chosen deliberately against the real cost of getting it wrong.
2. **A model's score is trusted at face value.** Nothing checks whether the evidence behind
   a given score is actually strong enough to justify auto-approving or auto-rejecting on it.

This project treats both as first-class problems, not afterthoughts.

---

## Architecture

```
Transaction data (Razorpay test-mode payment, via MCP client)
      │
      ▼
Feature pipeline (24 features: velocity, geolocation, VPN/proxy signals,
                   merchant risk, prior disputes, CVV retries, etc.)
      │
      ▼
Agent 1 — Pattern/Evasion Agent          (defensive drift detection only —
      │                                    never frames or simulates fraud techniques)
      ▼
Agent 2 — Detection & Scoring Agent      (Random Forest, 300 estimators,
      │                                    class_weight="balanced" + SHAP evidence)
      ▼
Agent 3 — Reviewer Agent                 (independent evidence-strength check —
      │                                    NOT a re-score, a check on the score)
      ▼
Decision Router                          (auto-approve / escalate to human / auto-reject,
      │                                    thresholds chosen by INR cost optimization)
      ▼
Audit Log                                (append-only: score, evidence, verdict,
                                           decision, timestamp, actor)
```

### Why three agents, not one model + SHAP?

SHAP explains *what* drove a score. It does not tell you whether the score itself is
trustworthy given how the evidence combines. A high score resting on one weak signal, or on
several redundant signals from the same category, or on evidence that contradicts other facts
about the transaction, can all produce a SHAP explanation that *looks* reasonable while the
underlying confidence is not earned. Agent 3 exists specifically to catch that — it is
deliberately not a re-scoring function, it consumes `{score, evidence}` and returns
`{verdict, confidence_adjustment, reason}`.

**Reviewer Agent — four independent checks (`app/agents/reviewer_agent.py`):**
1. Evidence count vs. score magnitude — elevated score, one weak signal → downgrade
2. Evidence diversity — redundant same-category signals count for less than independent ones
3. Contradiction check — e.g. a long-standing, dispute-free card cuts against a flagged score
4. Threshold margin — scores near the auto-reject line escalate to a human instead of
   auto-deciding

This makes the Reviewer Agent a **model-agnostic trust layer**: Agent 2 could be swapped for
any upstream fraud scorer — a rule engine, this Random Forest model, or in principle a
production-scale system — and Agent 3's job stays the same, because it only needs a
`{score, evidence}` shape as input.

---

## Model selection: why Random Forest, and why the threshold matters more than the model

Full methodology in [`docs/model_selection.md`](docs/model_selection.md). Summary:

**Round 1 — five models compared at the default 0.5 threshold**, on a realistically imbalanced
dataset (1.7% fraud rate — not artificially balanced):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Decision Tree | 0.063 | 0.515 | 0.113 | 0.599 |
| Random Forest | 0.0 | 0.0 | 0.0 | **0.927** |
| Extra Trees | 0.0 | 0.0 | 0.0 | 0.902 |
| XGBoost | 0.400 | 0.118 | 0.182 | 0.852 |
| LightGBM | 0.284 | 0.279 | 0.282 | 0.870 |

Random Forest had the **best ranking ability of any model (AUC 0.927)** but predicted **zero**
fraud cases at 0.5 — not a model failure, a threshold failure. At a 1.7% base rate, predicted
probabilities rarely cross 0.5 even when the model ranks correctly. This single result is the
reason the Decision Router treats **score** and **decision threshold** as two separate
concerns, not one.

**Round 2 — India-specific cost model, not accuracy, chose the threshold.**

Costs are grounded in two independently sourced figures, not assumptions:

- **False negative (missed fraud): ₹73,168.** Derived from this dataset's average fraud
  transaction amount (~$181.25, converted at ≈₹87/USD ≈ ₹15,769), multiplied by the **4.64×
  fraud-cost multiplier for financial institutions in India**, reported in the LexisNexis *True
  Cost of Fraud Study — Asia Pacific*.
- **False positive (unnecessary manual review): ₹85.** Derived from Indian fraud-analyst hourly
  compensation (~₹377/hour, SalaryExpert), assuming ~13–15 minutes per manual review.
- **Ratio: ≈860:1** — missing one fraud case costs roughly 860× more than one unnecessary
  manual review.

Sweeping thresholds against this cost function on the held-out test set (fine-grained sweep,
0.0001–0.05):

| Threshold | TP | FP | FN | TN | Total cost |
|---|---|---|---|---|---|
| 0.5 (default) | 0 | 0 | 68 | 3932 | ₹4,975,424 |
| **0.0071–0.0096 (cost-optimal plateau)** | **68** | **1,425** | **0** | **2,507** | **₹121,125** |

At the chosen operating threshold, the model catches **100% of fraud in the held-out set** —
zero missed cases — at the cost of 1,425 false alarms out of 4,000 test transactions. Precision
looks low in isolation (≈4.6%), but that is the *correct* trade given the cost asymmetry: a
false alarm costs ₹85 to review; a missed fraud case costs ₹73,168. Optimizing for F1 instead
of cost would have picked a very different, worse threshold.

**This divergence between the F1-optimal and cost-optimal operating points is the core
technical finding of this project** — not "which model wins," but "the decision threshold is a
business decision, and should be chosen against a real, sourced cost model, not left at a
default or optimized for a metric that doesn't reflect the actual cost of being wrong."

---

## Screenshots

**Precision, recall, and AUC across all five models tested (Round 1):**

![Precision vs recall by model](docs/screenshots/precision_recall_by_model.png)

Random Forest and Extra Trees have the best ranking ability (AUC 0.90+) but zero precision/
recall at the default 0.5 threshold — the chart makes the "threshold problem, not a model
problem" finding visually obvious.

**Total cost, default vs. cost-optimal threshold, using the final India-specific cost model:**

![Cost chart, INR, final](docs/screenshots/cost_chart_inr_final.png)

*(An earlier pass of this same analysis, run with placeholder USD costs [$25 FP / $2,000 FN]
before the LexisNexis and Indian salary figures were sourced, is kept at
`docs/screenshots/cost_chart_round2_superseded_usd.png` for transparency about the development
process — it is not the final number and should not be read as current.)*

---

## Why the false-positive rate is manageable, not just high

1,425 false positives is a real number, and the honest answer to "isn't that too much
friction" is: **that is exactly what Agent 3 exists to triage.** A flagged transaction doesn't
mean an automatic block — it means routing into a queue where the Reviewer Agent's evidence
check determines whether it goes to a human analyst or gets confirmed. The Decision Router's
`escalate` action *is* the human-in-the-loop step, not a side effect of the threshold.

---

## Human-in-the-loop and audit trail

Every transaction resolves to one of three actions:

| Action | Trigger | What happens |
|---|---|---|
| `auto_approve` | Low score, no Reviewer objection | No human involvement |
| `escalate` | Borderline score, or Reviewer flags weak/contradictory evidence | Routed to a human analyst queue; decision logged with analyst identity once resolved |
| `auto_reject` | High score, Reviewer confirms evidence is strong | Transaction blocked automatically |

Every step — score, SHAP evidence, Reviewer verdict, final decision, timestamp, and actor — is
written to an append-only audit log (`app/audit_log.py`), independently verified to have no
delete/update path.

---

## Repository structure

```
app/
├── main.py                          FastAPI app — all endpoints
├── models.py                        Pydantic schemas (Transaction, ScoringResult,
│                                     ReviewVerdict, EvidenceItem, Decision)
├── ml_model.py                      Loads trained Random Forest + SHAP TreeExplainer;
│                                     turns a Transaction into a ScoringResult
├── decision_router.py               Threshold-based routing (INR cost-optimal thresholds)
├── audit_log.py                     Append-only audit log (SQLite)
├── evaluation.py                    Precision/recall + INR cost-analysis endpoint logic
├── mcp_client.py                    Razorpay integration client, with a synthetic-data
│                                     fallback for demos without live credentials
├── agents/
│   ├── pattern_agent.py             Agent 1 — defensive drift detection (statistical,
│   │                                 z-score based; not wired to this dataset — see
│   │                                 Limitations)
│   ├── reviewer_agent.py            Agent 3 — Evidence-Strength Check (see above)
│   └── deprecated_phase1/
│       └── scoring_agent_rule_based.py   Superseded Phase 1 rule engine, kept for
│                                          reference only — not imported anywhere
├── ml_artifacts/
│   ├── model.pkl                    Trained RandomForestClassifier
│   └── encoders.pkl                 Fitted LabelEncoders for categorical features
data/
├── credit_card_fraud_2026.csv       Synthetic, 20,000 rows, 1.7% fraud rate, 24 features
├── fraud_test_sample.json           3 known-fraud held-out cases, used for pipeline
│                                     verification (see Verification below)
└── synthetic_transactions.json      Small hand-crafted set for smoke testing
docs/
├── model_selection.md               Full three-round model/threshold selection methodology
├── endpoints.md                     Full API endpoint specification
└── roadmap.md                       Build phases and competitive-landscape notes
tests/
└── test_pipeline_smoke.py           End-to-end pipeline test (ingest → score →
                                      review → decide → audit)
```

---

## API endpoints

Full spec in [`docs/endpoints.md`](docs/endpoints.md). Core loop:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/transactions/{id}/score` | Run Agent 2 (Random Forest + SHAP), return score + evidence |
| `POST` | `/transactions/{id}/review` | Run Agent 3, return verdict + adjusted confidence |
| `POST` | `/decisions/{id}` | Run the full pipeline, route the decision, write the audit entry |
| `GET` | `/audit-log` | Retrieve audit trail, filterable by transaction |
| `GET` | `/metrics/precision-recall` | Measured precision/recall on the held-out set |
| `GET` | `/health`, `/status` | Liveness + current threshold/model configuration |

---

## Verification

This isn't a claimed result — it was independently reproduced, not just reported:

- The trained model (`app/ml_artifacts/model.pkl`) was reloaded fresh and confirmed to be a
  real, fitted `RandomForestClassifier(class_weight="balanced", n_estimators=300)`.
- The cost-optimal threshold sweep (0.0001–0.05 grid) was rerun independently against the held-
  out test set using the sourced ₹73,168 / ₹85 cost figures, and reproduced the exact same
  result: threshold plateau at 0.0071–0.0096, 68/68 fraud cases caught, 1,425 false positives,
  total cost ₹121,125.
- All 3 known-fraud cases in `data/fraud_test_sample.json` were independently run through the
  full pipeline (`score → review → route`) and correctly resulted in `auto_reject`, with legible
  evidence (CVV retries + billing mismatch; foreign transaction + elevated merchant risk; IP
  mismatch + foreign transaction).
- No stale references to the deprecated Phase 1 rule-based scorer remain anywhere outside
  `app/agents/deprecated_phase1/` — confirmed by repo-wide search.

---

## Running it locally

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API.

**Tests:**
```bash
python -m tests.test_pipeline_smoke
```

---

## What's fully built vs. designed for

Being explicit about this distinction rather than blurring it:

**Built and verified:**
- Full three-agent pipeline (Pattern Agent implemented, Scoring + Reviewer Agents wired and tested)
- Trained Random Forest model with SHAP-based evidence generation
- Cost-optimal threshold, sourced from two independent Indian industry figures
- Append-only audit trail
- FastAPI backend with the full endpoint set
- End-to-end smoke tests, independently reproduced

**Designed, not yet built:**
- Live integration against a running `razorpay-mcp-server` instance — the client code exists
  (`app/mcp_client.py`) with a synthetic-data fallback, but has not been tested against a live
  server; this is the one deployment risk explicitly flagged for demo day.
- Frontend/dashboard UI — the API is fully functional and demoable via `/docs`, but there is no
  dedicated UI yet.
- Kubernetes/production infra, RBAC, and broader MCP-ecosystem exposure — reasonable next steps
  for a production version, out of scope for a buildathon timeline.

---

## Honest limitations

1. **Low absolute precision (≈4.6%) at the chosen threshold.** This is the correct trade given
   the ₹73,168 : ₹85 cost asymmetry, not an error — but it means a real deployment leans hard on
   the Reviewer Agent and human analysts to triage the resulting volume, not treat every flag as
   equally urgent.
2. **Model probabilities are compressed and not well-calibrated.** Maximum probability on the
   held-out set was 0.28, with fraud/non-fraud ranges overlapping substantially. The model
   discriminates well by *rank* (AUC 0.927) but its raw probability outputs should be read as a
   relative risk signal, not a calibrated percentage — router thresholds are set as relative
   bands for this reason, not interpretable probabilities.
3. **Synthetic dataset.** `credit_card_fraud_2026.csv` is clearly synthetic (see filename).
   These numbers are methodology-validated on this dataset, not universal constants that will
   transfer unchanged to live Razorpay transaction data.
4. **No merchant/customer grouping key in this dataset.** Agent 1's drift-detection logic is
   fully implemented (`app/agents/pattern_agent.py`) but cannot be demonstrated end-to-end
   against this dataset, which has no entity ID to group transactions by over time. It needs a
   real grouping key (customer_id, card_id, or merchant_id) to run live.
5. **Cost figures are sourced but not Razorpay-specific.** The ₹73,168 and ₹85 figures come from
   independent, named industry studies (LexisNexis *True Cost of Fraud — Asia Pacific*;
   Indian fraud-analyst salary benchmarks), not from Razorpay's own transaction or operations
   data, which we don't have access to. They are the strongest publicly available proxy, stated
   as such rather than presented as Razorpay-internal figures.

---

## Defensive-use statement

Agent 1 performs **statistical drift detection only** — it flags when a merchant's transaction
pattern deviates from its own historical baseline, for review. It does not model, simulate, or
output fraud techniques, evasion strategies, or attacker behavior in any form. This is stated
explicitly here and in the module docstring (`app/agents/pattern_agent.py`), consistent with
Track 02's defense-only requirement.

---

## Disclaimer

This is a buildathon submission built primarily on a synthetic dataset. It is not a production
fraud-detection system, has not been evaluated against live payment data, and the cost figures,
while independently sourced, are industry proxies rather than Razorpay-specific numbers. Nothing
in this repository should be treated as a certified or regulator-approved risk model.
