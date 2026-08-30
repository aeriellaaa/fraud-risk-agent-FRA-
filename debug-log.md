# Debug & Progress Log — Fraud/Risk Flag Agent

Living document. Updated as we go. Purpose: capture what broke, how long it cost us, what
we learned, and the moments that actually taught us something about the system itself --
not just "here's what we built."

---

## Bugs found, in order

### 1. `storage.py` referenced `txn.card_id` after schema migration
**Symptom:** 500 Internal Server Error on every `/transactions/ingest` call, first real request
against the live server.
**Root cause:** When we rewrote `models.py` to match the real dataset schema (no card-linking
field), `storage.py`'s `save_transaction()` still had a leftover line building a `card_history`
dict keyed by `txn.card_id` -- a field that no longer existed on `TransactionIn`.
**Time to resolve:** ~15-20 minutes.
**Lesson:** When a schema changes, grep the whole codebase for the old field name before
assuming a clean cutover.

### 2. Phantom file writes -- files reported as created, but weren't actually on disk
**Symptom:** Recurring, multiple times -- test files, training scripts, and agent rewrites all
silently failed to save at least once, discovered only when a later command referenced them
and failed.
**Root cause:** Long multi-line PowerShell heredoc pastes, especially when immediately preceded
or followed by other commands in the same paste, appear to have silently truncated or not
executed at all, with no visible error at the time.
**Time to resolve:** ~10-15 minutes each time it happened.
**Lesson (the real one):** Never trust "I did it" -- verify with a read-back command
(`Select-String`, `Test-Path`, `Get-Content`) immediately after every file write, before
building anything on top of it. This became a hard rule partway through and caught several
later bugs before they became demo-day surprises.

### 3. Environment: `pandas`/`numpy` pinned versions had no Python 3.13 wheel
**Symptom:** `pip install -r requirements.txt` failed trying to compile `pandas` from source,
erroring on a missing Visual Studio component.
**Time to resolve:** ~5 minutes -- relaxed pins from `==` to `>=`.
**Lesson:** Pin versions loosely for fast-moving ML libraries unless there's a specific reason
to hard-pin, especially on an unusual Python version.

### 4. Environment: `shap` needed a C++ compiler
**Symptom:** Same category as #3 -- `shap==0.46.0` had no prebuilt Windows/3.13 wheel, tried to
build its C extension, failed needing Visual C++ Build Tools.
**Time to resolve:** ~5 minutes -- bumped to `shap>=0.46.0`, resolved to a version with a wheel.
**Lesson:** Same as #3. Also: always have a fallback plan (we had one -- feature-importance
evidence instead of SHAP) before attempting an install that could eat significant time.

### 5. Terminal confusion: cmd.exe vs PowerShell vs `curl` alias collision
**Symptom:** `#` comments failing, `mkdir -p` failing, heredocs opening an editor instead of
executing, and later `curl -H "..."` throwing an `Invoke-WebRequest` parameter error.
**Root cause:** cmd and PowerShell have incompatible syntax, and PowerShell secretly aliases
`curl` to `Invoke-WebRequest`, which doesn't accept real curl flags.
**Time to resolve:** ~20-30 minutes cumulative.
**Lesson:** Confirm which shell you're in early, never assume a tool name means the same thing
across shells.

### 6. Agent 3 evidence-strength calibration bug
**Symptom:** Not a crash -- a logic bug. When Agent 2 was rewritten to use SHAP, evidence
strength was normalized relative to the top signal within each transaction, making minor SHAP
noise look like strong contradicting evidence.
**How we caught it:** Reading actual evidence values in a live test response, not from an
automated test.
**Time to resolve:** ~20 minutes.
**Lesson:** This bug would NOT have been caught by a green test suite, because the tests were
written before this rewrite. Reading actual output values is what caught this, not pass/fail.

### 7. Reviewer Agent: verdict/reason text mismatch
**Symptom:** For a transaction with zero supporting evidence, the verdict correctly read
`insufficient_evidence`, but the reason text said "confidence downgraded" -- language belonging
to a different verdict.
**Root cause:** A contradiction-check reason string was appended before the final verdict was
decided, so it beat the "no supporting evidence" explanation into the reasons list.
**Time to resolve:** ~10 minutes.
**Lesson:** String/message consistency bugs don't show up as crashes or wrong numbers, only as
something that reads wrong to a human. Worth a final "read it like a judge would" pass.

### 8. `-replace` matched nothing because the comment text was guessed, not verified
**Symptom:** A verification check for an exact updated line came back empty after what looked
like a successful write with no errors.
**Root cause:** The `-replace` pattern targeted exact remembered line text that didn't actually
match what was on disk. Since it found no match, it silently did nothing.
**Time to resolve:** ~5 minutes once caught.
**Lesson:** Exact-string find-and-replace fails silently when the string doesn't match. Broad,
value-only patterns are safer than guessing full remembered lines.

### 9. Rewritten reviewer test had incorrect expected math
**Symptom:** After rewriting all 4 test files for the ML scorer, one test failed -- expected
`CONFIDENCE_DOWNGRADED`, got `CONFIDENCE_UPHELD`.
**Root cause:** Not a code bug -- a test-authoring bug. The fabricated evidence didn't actually
cross any of Agent 3's real thresholds.
**Time to resolve:** ~10 minutes.
**Lesson:** When a newly-written test fails against newly-written code, don't assume the code
is wrong by default -- trace the actual threshold math for both.

### 10. `/metrics` rewrite silently never landed -- twice compounded
**Symptom:** A targeted text update matched nothing; closer inspection showed the file was
still the entire original Phase 1 placeholder router, not the real-metrics rewrite from
several messages earlier.
**Time to resolve:** ~10 minutes once caught.
**Lesson:** The strongest version of this session's core lesson -- verification needs to happen
close to the time of the write. A silently-failed file can sit undetected for many turns.

### 11. `.env` file had a UTF-8 BOM, `python-dotenv` couldn't read it
**Symptom:** "Key not found in .env" even though the file visibly had the right two lines.
**Root cause:** PowerShell's `Set-Content -Encoding utf8` writes a byte-order mark by default,
invisibly prefixing the first key so it didn't match exactly.
**Time to resolve:** ~15 minutes.
**Fix:** Rewrote `.env` using `[System.IO.File]::WriteAllText(...)` with explicit no-BOM
encoding.
**Lesson:** A file "looking right" when displayed is not the same as being byte-for-byte
correct.

### 12. `pattern_agent.py` two-mode rewrite never saved
**Symptom:** `TypeError: detect_drift() got an unexpected keyword argument 'card_history'` on
the first real call to the entity-drift demo endpoint.
**Root cause:** Same as #10 -- a rewrite from several messages earlier never actually saved,
and went unverified through an entire detour (Razorpay API setup, real data generation) before
finally being exercised.
**Time to resolve:** ~5 minutes once caught.
**Lesson:** The clearest illustration all session of why immediate verification matters -- real
work was built correctly on top of one silently-broken foundation file.

---

## The moment that actually taught us something

Running the same ambiguous fraud transaction through both the Phase 1 rule-based scorer and the
Phase 2 trained model: rule-based scored it 0.365, weighted heavily on `merchant_risk_score`.
The trained Random Forest scored it 0.10, and `merchant_risk_score` didn't even appear in its
top-6 SHAP contributors for that case. Neither system is "wrong" -- it's a concrete
demonstration that intuition-based feature weighting and data-driven weighting can genuinely
disagree, and you don't find out which signals actually matter until you train on real data.

Confirmed again independently later: hand-crafted "obviously suspicious" transactions don't
reliably score high on this model. Fixed properly by scoring all 339 real fraud cases and
finding the genuinely highest-scoring ones for demo use (median score across all real fraud:
0.6467) -- rather than guessing which transaction would demo well.

---

## What would have gone faster with hindsight

1. Verify every file write immediately, from the very first one.
2. Confirm the shell (cmd vs PowerShell) before giving multi-line commands.
3. Read actual output values, not just status codes, before declaring a step done.
4. Check `git status`/`git branch` proactively at natural checkpoints, not just when something
   breaks.

---

## Status

README fully rewritten for Phase 2. Real trained model, real SHAP evidence, real cited Indian
cost figures, real Razorpay test-mode data (8 customers, 70 orders), Agent 1's entity-drift
mode demonstrated against real customer linkage, 18/18 tests passing, curated real demo
transactions identified. Frontend not yet started.
