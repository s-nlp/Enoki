# rules_new changelog (top-down line)

Spec: docs/superpowers/specs/2026-05-16-rules-new-topdown-line-design.md
Corpus: LSOIE+OpenIE4 (legacy line reference S=0.6785).

Two-phase gate:
- Bootstrap (L0->L5): accept a change if net-positive ΔS vs the running
  rules_new baseline; saturate a level before descending.
- Refinement (post-L5): accept iff ΔS >= 0.0005 AND ΔP >= -0.0075. (Relaxed 2026-05-18 from ΔS>=0.003: eval is deterministic so any ΔS>0 is a real gain; 0.0005 still rejects no-ops/noise but admits genuine small specific-rule wins. Precision floor unchanged. A full-dev re-confirm + prune pass follows the relaxed-bar batch. Global OptimizeConfig is intentionally NOT changed — it carries the legacy line's calibration; this relaxed bar is a rules_new-refinement-phase policy.)
- Precision phase (2026-05-18): accept iff ΔP ≥ +0.005 AND ΔS ≥ −0.005 AND ΔF1 ≥ −0.005 (quality-first; the recall-weighted S-gate structurally rejected genuine precision tightenings — see refinement findings). Scoped to rules_new precision iterations; global OptimizeConfig unchanged.

## Running baseline

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| empty (iter 0) | 0 | 0 | 0 | 0 | 0 |
| L0 core_svo | 0.6065 | 0.1839 | 0.2822 | 0.2396 | 0.3421 |
| L0 core_attr | 0.6089 | 0.1882 | 0.2876 | 0.2452 | 0.3489 |
| L0 core_oprd | 0.6111 | 0.2015 | 0.3030 | 0.2628 | 0.3687 |

## Entries

**iter 0 — empty baseline.** rules_new discovers 0 rules; full-dev
S=0.0000 (P=0 R=0, 0 predictions, FN=5063 gold triplets). Bootstrap
phase begins with L0.

**L0 / iter 1 — `core_svo` (new).** Root VERB + nsubj + dobj/acomp.
Gates: contract OK, EXAMPLES OK, regression OK. Bootstrap accept
(ΔS=+0.3421 from S=0). New baseline S=0.3421 (P=0.6065 R=0.1839 F1=0.2822).

**L0 / iter 2 — `core_attr` (new).** Root VERB (lemma != 'be') + nsubj + attr child. Gates contract/EXAMPLES/regression OK. Bootstrap accept ΔS=+0.0068. New baseline S=0.3489 (P=0.6089 R=0.1882 F1=0.2876).

**L0 / iter 3 — `core_oprd` (new).** Root VERB + nsubj + oprd child (object predicate complement: named/appointed/painted/call). Gates contract/EXAMPLES/regression OK. Bootstrap accept ΔS=+0.0198. New baseline S=0.3687 (P=0.6111 R=0.2015 F1=0.3030).

**L0 saturated** at S=0.3687 (rules: core_svo, core_attr, core_oprd).

## Running baseline (L1)

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| L1 copula_be | 0.5815 | 0.2585 | 0.3579 | 0.3376 | 0.4423 |

## Entries (L1)

**L1 / iter 1 — `passive_by_agent` REJECTED.** ΔS=-0.0003 (≤0); reverted. Baseline(isolated) S=0.3687 P=0.6111 R=0.2015 F1=0.3030; withrule S=0.3684 P=0.5926 R=0.2028 F1=0.3022. Rule hurts precision more than it gains recall; passive constructions already partially covered.

**L1 / iter 2 — `copula_be` (new).** Root lemma 'be' + nsubj + attr/acomp child. Gates contract/EXAMPLES/regression OK. Bootstrap accept ΔS=+0.0736. New baseline S=0.4423 (P=0.5815 R=0.2585 F1=0.3579).

**L1 / iter 3 — `existential_there` REJECTED.** Contract lint OK; EXAMPLES[0] failed (exit 2). The construction (existent, be, None) with arg_head=None is blocked by the completeness filter (`_COPULA_AUX_LEMMAS` check in `filters/completeness.py` rejects all (copula, None-arg) triplets). Construction genuinely cannot satisfy the pipeline contract; reverted.

**L1 saturated** at S=0.4423 (added: copula_be).

## Running baseline (L2)

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| L2 prep_object | 0.5447 | 0.4647 | 0.5015 | 0.5364 | 0.6356 |

## Entries (L2)

**L2 / iter 1 — `prep_object` (new).** Root VERB + nsubj + prep child whose pobj is the argument; passive-agent 'by' skipped. Gates: contract OK, EXAMPLES OK, regression OK. Bootstrap accept ΔS=+0.1933. New baseline S=0.6356 (P=0.5447 R=0.4647 F1=0.5015).

**L2 / iter 2 — `dative_recipient` REJECTED.** ΔS=-0.0000 (≤0); reverted. Baseline(isolated) S=0.6356 P=0.5447 R=0.4647 F1=0.5015; withrule S=0.6356 P=0.5428 R=0.4657 F1=0.5013. Dative recipient constructions not present enough in the corpus to yield net gain.

**L2 / iter 3 — `phrasal_predicate` REJECTED.** ΔS=+0.0000 (≤0); reverted. Baseline(isolated) S=0.6356 P=0.5447 R=0.4647 F1=0.5015; withrule S=0.6356 P=0.5447 R=0.4647 F1=0.5015. Phrasal-verb constructions already fully covered by core_svo (same dobj target) or absent from the corpus; zero marginal contribution.

**L2 saturated** at S=0.6356 (added: prep_object).

## Running baseline (L3)

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| (no L3 rules accepted; baseline unchanged) | 0.5447 | 0.4647 | 0.5015 | 0.5364 | 0.6356 |

## Entries (L3)

**L3 / iter 1 — `coord_object` REJECTED.** ΔS=-0.0051 (≤0); reverted. Baseline(isolated) S=0.6356 P=0.5447 R=0.4647 F1=0.5015; withrule S=0.6305 P=0.5323 R=0.4651 F1=0.4965. Adds recall for conjunct dobj's but hurts precision more (spurious extractions on object-coordination in corpus sentences).

**L3 / iter 2 — `coord_verb` REJECTED.** ΔS=-0.0047 (≤0); reverted. Baseline(isolated) S=0.6356 P=0.5447 R=0.4647 F1=0.5015; withrule S=0.6308 P=0.5282 R=0.4675 F1=0.4960. Conjunct-verb triplets not well-represented in LSOIE+OpenIE4 gold; precision loss outweighs recall gain.

**L3 / iter 3 — `coord_subject` REJECTED.** ΔS=+0.0000 (≤0); reverted. Baseline(isolated) S=0.6356 P=0.5447 R=0.4647 F1=0.5015; withrule S=0.6356 P=0.5447 R=0.4647 F1=0.5015. Pure no-op: the segment stage (distribute_subjects) already conj-expands all coordinated subjects into subject_candidates before any rule runs; this rule never fires.

**L3 saturated** at S=0.6356 (added: none).

## Running baseline (L4)

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| L4 ccomp_arg | 0.5428 | 0.5098 | 0.5258 | 0.5868 | 0.6725 |
| L4 xcomp_arg | 0.5384 | 0.5305 | 0.5344 | 0.6052 | 0.6857 |
| L4 advcl_clause | 0.5154 | 0.5568 | 0.5353 | 0.6216 | 0.6907 |

## Entries (L4)

**L4 / iter 1 — `ccomp_arg` (new).** Root VERB + nsubj + ccomp child; arg_span_subtree=True. Gates OK. Bootstrap accept ΔS=+0.0369. New baseline S=0.6725 (P=0.5428 R=0.5098 F1=0.5258).

**L4 / iter 2 — `xcomp_arg` (new).** Root VERB + nsubj + xcomp child; arg_span_subtree=True. Gates OK. Bootstrap accept ΔS=+0.0131. New baseline S=0.6857 (P=0.5384 R=0.5305 F1=0.5344).

**L4 / iter 3 — `advcl_clause` (new).** Root VERB + nsubj + advcl child; arg_span_subtree=True. Gates OK. Bootstrap accept ΔS=+0.0051. New baseline S=0.6907 (P=0.5154 R=0.5568 F1=0.5353).

**L4 / iter 4 — `relcl_object_gap` REJECTED.** ΔS=-0.0068 (≤0); reverted. Baseline(isolated) S=0.6907 P=0.5154 R=0.5568 F1=0.5353; withrule S=0.6839 P=0.4971 R=0.5619 F1=0.5275. Relative-clause object-gap pattern hurts precision (spurious extractions outweigh recall gain in LSOIE+OpenIE4).

**L4 / iter 5 — `acl_modifier` REJECTED.** ΔS=-0.0101 (≤0); reverted. Baseline(isolated) S=0.6907 P=0.5154 R=0.5568 F1=0.5353; withrule S=0.6806 P=0.4919 R=0.5605 F1=0.5240. Participial/clausal noun-modifier pattern hurts precision significantly; sparse corpus coverage.

**L4 saturated** at S=0.6907 (added: ccomp_arg, xcomp_arg, advcl_clause).

## Running baseline (L5)

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| (no L5 rules accepted; baseline unchanged) | 0.5154 | 0.5568 | 0.5353 | 0.6216 | 0.6907 |

## Entries (L5)

**L5 / iter 1 — `appositive` REJECTED.** ΔS=-0.0095 (≤0); reverted. Baseline(isolated) S=0.6907 P=0.5154 R=0.5568 F1=0.5353; withrule S=0.6812 P=0.4980 R=0.5568 F1=0.5257. Appositive N,A pattern (both NOUN/PROPN with comma guard) hurts precision significantly; spurious appos extractions outnumber recall gains in LSOIE+OpenIE4.

**L5 / iter 2 — `partitive` REJECTED.** ΔS=-0.0016 (≤0); reverted. Baseline(isolated) S=0.6907 P=0.5154 R=0.5568 F1=0.5353; withrule S=0.6892 P=0.5119 R=0.5572 F1=0.5336. Partitive quantifier "X of Y is Z" construction already substantially covered by copula_be + prep_object; marginal contribution is negative. (EXAMPLES adjusted from passive sentences that don't parse as root-be to active copular variants.)

**L5 / iter 3 — `copula_less` REJECTED.** ΔS=-0.0002 (≤0); reverted. Baseline(isolated) S=0.6907 P=0.5154 R=0.5568 F1=0.5353; withrule S=0.6905 P=0.5150 R=0.5568 F1=0.5351. Copula-less predication (headline "X ADJ/NOUN" without cop) is too rare in LSOIE+OpenIE4 to yield net gain; near-zero marginal contribution. (EXAMPLES adjusted: headline-style sentences where subject parses as compound rather than nsubj don't fire; used determiner-headed NP+ADJ sentences that parse correctly.)

**L5 saturated** at S=0.6907 (added: none).

## Refinement phase (ε=0.003, δ=-0.0075)

Bootstrap ladder complete. From here, a change is accepted only if
ΔS ≥ 0.003 AND ΔP ≥ -0.0075 (the gate's NEW_BASELINE/[dev_score] lines
report the deltas; the caller applies this stricter bar — the runner
still exits 0 on any ΔS>0).

### Running baseline (Refinement)

| state | P | R | F1 | cov | S |
|---|---|---|---|---|---|
| R2 svo_passive (new) | 0.5093 | 0.5777 | 0.5414 | 0.6456 | 0.7028 |
| R5 bundle: prep_object-tighten + relcl_subject | 0.5142 | 0.5840 | 0.5469 | 0.6496 | 0.7093 |

**Refine R2 — `svo_passive` (new, ROOT passive nsubjpass+auxpass).** Gates contract/EXAMPLES/regression OK. ΔS=+0.0121 ΔP=-0.0060. Accept (ΔS≥0.003 ∧ ΔP≥−0.0075). New baseline S=0.7028.

Note: R1 (advcl_clause fronted/conditional tighten) was REJECTED earlier: ΔS=−0.0033 (pure precision can't clear the recall-weighted bar); reverted.

R3 (relcl_subject) REJECTED: ΔS=−0.0003 ΔP=−0.0077 (recall+ but precision floor breached); reverted.

**Refine R4 — `conj_verb_share_subject` (new; conj-of-root VERB with no own subject, requires own dobj/prep arg).** Gates contract/EXAMPLES/regression OK. ΔS=-0.0000 ΔP=-0.0007. Reject (ΔS < 0.003). Baseline(isolated) S=0.7028 P=0.5093 R=0.5777 F1=0.5414; withrule S=0.7028 P=0.5086 R=0.5783 F1=0.5412. Despite ~138-FN recall cluster diagnosis, LSOIE+OpenIE4 gold awards near-zero recall credit for conj-verb patterns; tiny precision loss from spurious triplets nets zero score gain. Rule deleted; baseline remains S=0.7028.

**Refine R5 — BUNDLE: prep_object rel-pronoun-subject guard (precision) + `relcl_subject` (recall, R3 redesigned with head_noun-not-matrix-subject guard).** Contract/EXAMPLES OK. ΔS=+0.0065 ΔP=+0.0049 (bundle vs S=0.7028). Accept (ΔS≥0.003 ∧ ΔP≥−0.0075). New baseline S=0.7093. (decomposition: prep_object-tighten alone ΔS=+0.0068, relcl_subject alone ΔS=−0.0002)

## Final verification (Task 9)

| line | rules | P | R | F1 | S |
|---|---|---|---|---|---|
| rules_new (top-down) | 8 | 0.5154 | 0.5568 | 0.5353 | 0.6907 |
| legacy (reference) | 16 | 0.459 | 0.5931 | 0.5175 | 0.6785 |

Success criterion (rules_new S ≥ legacy 0.6785): **MET** — rules_new S=0.6907 vs legacy S=0.6785, with HALF the rule count.

Per-level bootstrap contribution (ΔS vs the running baseline at that level; from this CHANGELOG's entries):

| level | accepted rules | S after |
|---|---|---|
| L0 | core_svo, core_attr, core_oprd | 0.3687 |
| L1 | copula_be | 0.4423 |
| L2 | prep_object | 0.6356 |
| L3 | (none) | 0.6356 |
| L4 | ccomp_arg, xcomp_arg, advcl_clause | 0.6907 |
| L5 | (none) | 0.6907 |

Infra tests: tests/prev/test_rules_new_infra.py — 7 passed (full
tests/prev/ suite: 183 passed). The two iteration-0 empty-state
scaffold tests were evolved post-verification to assert the
completed-line invariants (8-rule discovery, pipeline routing,
legacy=16 isolation); see the test-evolution commit.

The legacy `fact_extractor/v2/rules/` line and its CHANGELOG are
unmodified (16 rules; reference S=0.6785). The top-down line reaches a
higher S with 8 rules vs 16, validating the general→specific
methodology. Refinement phase remains open for future ΔS≥0.003 work.

**Policy R8 — refinement bar relaxed to ΔS≥0.0005 ∧ ΔP≥−0.0075** (2026-05-18; rationale above). Prior strict-bar rejects R6/R7 re-evaluated below.

**Refine R6 — copula_be tighten REJECTED.** ΔS=+0.0017 ΔP=+0.0035 (vs S=0.7093); reverted. Guards (relativizer subject + pleonastic-it/ccomp extraposition) cleared contract/EXAMPLES/regression and improved precision, but ΔS=+0.0017 < required 0.003 strict bar.

**Refine R7 — BUNDLE copula_be + advcl_clause tighten REJECTED.** ΔS=+0.0019 ΔP=+0.0075 (vs S=0.7093); decomposition: copula_be alone ΔS=+0.0017, advcl smart alone ΔS=+0.0002; both reverted. Full-set bundle: S=0.7112 P=0.5217 R=0.5803 F1=0.5494 (TP=2938 FP=2694 FN=2125). Contract/EXAMPLES/regression all passed for both rules; purpose-advcl still fires ("She volunteered to help the community." correctly emitted); guard narrowly correct (skips fronted+temporal/conditional mark only, preserves right-attached and purpose/TO advcls). ΔS=+0.0019 < required 0.003 strict bar — bundle sub-threshold despite ΔP=+0.0075 well within the precision floor.

| R9 copula_be tighten (relaxed bar) | 0.5177 | 0.5836 | 0.5487 | 0.6492 | 0.711 |

**Refine R9 — copula_be relativizer/pleonastic-it precision guard, ACCEPTED under relaxed bar.** Contract/EXAMPLES OK. ΔS=+0.0017 ΔP=+0.0035 (full-set vs S=0.7093; ≥0.0005 ∧ ≥−0.0075). New baseline S=0.711.

| R10 acl_passive_participle (new) | 0.5179 | 0.5902 | 0.5516 | 0.6572 | 0.7159 |

**Refine R10 — `acl_passive_participle` (new; acl/relcl VBN + by-agent pobj).** Gates contract/EXAMPLES/regression OK. ΔS=+0.0050 ΔP=+0.0002 (full-set vs S=0.7110; isolated baseline S=0.7110). ACCEPT (relaxed bar: dS≥0.0005 ∧ dP≥−0.0075). New baseline S=0.7159. Emits (head_noun, participle, agent_pobj) for post-nominal passive participials with by-agent; the plain-prep case is already covered by prep_object on the same acl clause.

| R11 verb_advmod_result (new) | 0.5170 | 0.5941 | 0.5529 | 0.6592 | 0.7177 |

**Refine R11 — `verb_advmod_result` (new; root VERB + advmod ordinal/directional).** Gates contract/EXAMPLES/regression OK. ΔS=+0.0018 ΔP=−0.0008 (full-set vs S=0.7159; isolated baseline S=0.7159). ACCEPT (relaxed bar: dS≥0.0005 ∧ dP≥−0.0075). New baseline S=0.7177. Emits (subject, verb, advmod) when the advmod is a NUM or member of a closed ordinal/directional set {"first","second",...,"up","down","ahead","higher","lower"} and verb has no dobj (so core_svo doesn't already cover it). Covers ranking ("finished fifth"), placement ("ranked third"), and directional motion ("moved up", "closed higher").

| R12 copula_be_prep (new) | 0.5164 | 0.5994 | 0.5548 | 0.6664 | 0.7214 |

**Refine R12 — `copula_be_prep` (new; copular BE + prep locative complement; rule renamed copula_inchoative→copula_be_prep: it is a be+prep locative rule, not inchoative).** Gates contract/EXAMPLES/regression OK. ΔS=+0.0036 ΔP=−0.0006 (full-set vs S=0.7177; isolated baseline S=0.7177). ACCEPT (relaxed bar: dS≥0.0005 ∧ dP≥−0.0075). New baseline S=0.7214. Emits (subject, be-form, pobj) via prep+pobj for copular-BE locative/directional constructions ("Paris is in France", "She was in Paris", "They are at the airport") that copula_be does not handle (copula_be targets attr/acomp only; prep_object targets VERB root but be is AUX). Guards: skip relativizer subjects, pleonastic-it, and any sentence where attr/acomp is already present (defer to copula_be).

## Refinement phase summary (prune-audit @ final ensemble)

Bootstrap end S=0.6907 (8 rules) → refinement end **S=0.7219** (12 rules; relcl_subject pruned P1).
Net refinement ΔS=+0.0312, ΔR=+0.0363, ΔF1=+0.0209.

Final-ensemble self-isolated contribution (relaxed bar ΔS≥0.0005 ∧ ΔP≥−0.0075):
| rule | dS (final) | dP (final) | verdict |
|---|---|---|---|
| svo_passive | +0.0109 | −0.0074 | keep |
| relcl_subject | −0.0005 | −0.0071 | PRUNED (P1) |
| acl_passive_participle | +0.0049 | +0.0002 | keep |
| verb_advmod_result | +0.0018 | −0.0008 | keep |
| copula_be_prep | +0.0036 | −0.0006 | keep |
(prep_object & copula_be are tightenings of pre-existing rules, not separately prunable.)

Accepted: R2 svo_passive, R5 bundle (prep_object-tighten+relcl_subject), R9 copula_be guard, R10 acl_passive_participle, R11 verb_advmod_result, R12 copula_be_prep. Rejected (audited above): R1, R3, R4, R6, R7. Policy relaxed at R8 (ΔS≥0.003→0.0005). Prune-candidates identified: relcl_subject (dS=−0.0005, dP=−0.0071 in final-13 ensemble — subsumed/dominated by later rules; kept pending controller decision).

Full-dev confirmation (2026-05-18): rules_new S=0.7214 P=0.5164 R=0.5994 F1=0.5548 cov=0.6661 TP=3035 FP=2842 FN=2028. Legacy S=0.6785 (unchanged). Discovery: rules_new=13, legacy=16. tests/prev/: 198 passed, 0 failed (test_rules_new_infra updated to assert 13-rule final set).

**Prune P1 — `relcl_subject` REMOVED (subsumed).** Final-ensemble self-isolation dS=−0.0005 dP=−0.0071 (net-negative; later R10–R12 cover its recall). Pruned: rules_new 13→12, S 0.7214→0.7219, P=0.5235 R=0.5931 F1=0.5562. The R5 prep_object tightening (the other half of that bundle) is retained. Refinement end-state: **12 rules, S=0.7219** (bootstrap end was 8 rules / S=0.6907).

## Refinement phase (Bundle-A precision pass)

| R13 Bundle-A precision (advcl+svo_passive+prep_object) | 0.5261 | 0.5931 | 0.5576 | 0.6628 | 0.7233 |

**Refine R13 — BUNDLE-A precision pass: advcl_clause subordinator/relativizer/participial-arg guard + svo_passive require-meaningful-arg guard + prep_object rel-pronoun-pobj guard.** Contract/EXAMPLES OK for all three. Bundle ΔS=−0.0065 ΔP=+0.0196 (full-set vs S=0.7219) — bundle net-negative. Decomposition: advcl_clause alone ΔS=−0.0014 (recall drop outweighs precision gain), svo_passive alone ΔS=−0.0066 (heavy recall drop: bare-arg-less passives actually annotated in gold more than diagnosed), prep_object alone ΔS=+0.0014 ΔP=+0.0026 (net-positive, clears relaxed bar). Kept: prep_object rel-pronoun-pobj guard only; advcl_clause and svo_passive reverted. Accept (relaxed bar: ΔS=+0.0014≥0.0005 ∧ ΔP=+0.0026≥−0.0075). New baseline S=0.7233.

**Refine R14 — recall bundle REJECTED.** conj_verb_share_subject alone ΔS=−0.0002 ΔP=−0.0010; relcl_own_nsubj alone ΔS=−0.0004 ΔP=−0.0071; bundle ΔS=−0.0005 ΔP=−0.0081 (both below ΔS≥0.0005 floor; bundle also breaches ΔP≥−0.0075). Contract/EXAMPLES/regression OK for both; precision discipline in place (conj skips own-subject and arg-less conj verbs; relcl skips matrix-subject head nouns). Despite ~103 conj-FN and ~138 relcl-FN clusters diagnosed, LSOIE+OpenIE4 gold awards near-zero recall credit: conj-verb patterns and non-matrix-subject relcl patterns are systematically under-annotated in the gold. Both rules deleted; baseline S=0.7233 unchanged.

| R15 small-recall: prep_object pcomp-widening | 0.5231 | 0.6014 | 0.5595 | 0.6680 | 0.7265 |

**Refine R15 — small-recall bundle: advcl_own_subject + acl_participle_prep + prep_object pcomp-widening; kept: prep_object pcomp-widening.** Contract/EXAMPLES OK for all three. Bundle full-set ΔS=+0.0032 ΔP=−0.0030 (vs S=0.7233/P=0.5261). Decomposition: advcl_own_subject solo ΔS=0.0000 ΔP=0.0000; acl_participle_prep solo ΔS=0.0000 ΔP=0.0000; prep_object pcomp-widening solo ΔS=+0.0032 ΔP=−0.0030. Accept (relaxed bar: ΔS=+0.0032≥0.0005 ∧ ΔP=−0.0030≥−0.0075) for prep_object pcomp-widening only. advcl_own_subject and acl_participle_prep both ΔS=0.0000 (below 0.0005 floor) and add zero marginal to the kept set — rejected and deleted. prep_object widening adds pcomp gerund-clause objects (e.g. "resulted in anyone being convicted") as a purely additive branch alongside the existing pobj path; all original EXAMPLES and guards unchanged. New baseline S=0.7265.

## Precision phase (2026-05-18)

Gate: ΔP ≥ +0.005 AND ΔS ≥ −0.005 AND ΔF1 ≥ −0.005 (vs baseline S=0.7265 P=0.5231 R=0.6014 F1=0.5595).

| Q1 precision: advcl_clause | 0.5365 | 0.5868 | 0.5605 | 0.6588 | 0.7252 |

**Q1 (precision phase) — advcl_clause subordinator/relativizer/VBG-gerund guard.** Contract/EXAMPLES OK (8 examples: purpose to-infinitival still fires; 7 temporal/conditional/causal/concessive skipped). Bundle ΔP=+0.0250 ΔS=−0.0071 ΔF1=+0.0001 (all 3 edits vs baseline) — bundle fails gate (ΔS=−0.0071 < −0.005). Decomposition: advcl_clause alone ΔP=+0.0134 ΔS=−0.0013 ΔF1=+0.0010 (PASS); copula_be alone ΔP=+0.0003 ΔS=−0.0015 ΔF1=−0.0008 (FAIL: ΔP<+0.005, precision-neutral); svo_passive alone ΔP=+0.0101 ΔS=−0.0044 ΔF1=−0.0001 (PASS); advcl+svo bundle ΔP=+0.0246 ΔS=−0.0056 ΔF1=+0.0009 (FAIL: ΔS<−0.005); advcl+copula bundle ΔP=+0.0137 ΔS=−0.0028 ΔF1=+0.0002 (PASS but copula is precision-neutral individually, dropped per policy). Kept: advcl_clause only — maximizes ΔP among gate-passing single-rule options (ΔP=+0.0134 > svo_passive ΔP=+0.0101); copula and svo_passive reverted via git checkout. Accept (precision-phase gate ΔP≥0.005 ∧ ΔS≥−0.005 ∧ ΔF1≥−0.005). New baseline S=0.7252 P=0.5365.

| Q2 svo_passive require agent/dobj | 0.5477 | 0.5738 | 0.5604 | 0.642 | 0.7209 |

> **Precision phase CLOSED at Q5 (2026-05-19).** Q1–Q4 traded recall for
> precision under the ΔP≥+0.005 sub-bar (P 0.5231→0.5635, S 0.7265→0.7190).
> Q5 (a strict S/P/F1 improvement) cleared the project refinement relaxed
> bar but missed the +0.005 sub-bar by 0.0009 — the signal that clean
> ≥+0.005 precision buckets are exhausted (core_svo/xcomp/ccomp verified to
> have no separable FP clusters). Per controller decision the precision
> phase is closed; **Q5+ are gated by the refinement relaxed bar
> (ΔS≥0.0005 ∧ ΔP≥−0.0075)**, which re-admits recall-positive specific/
> lexical rules.

**Q2 (precision phase) — svo_passive require agent-pobj OR dobj (drop bare/prep-only no-arg passives).** Contract/EXAMPLES OK (8 examples: all genuine by-agent or dobj passives; 4 old bare/prep-only examples replaced with verified by-agent/dobj sentences). ΔP=+0.0112 ΔS=−0.0043 ΔF1=−0.0001 (full-set vs S=0.7252). Accept (precision-phase gate: ΔP=+0.0112≥+0.005 ✓ ∧ ΔS=−0.0043≥−0.005 ✓ ∧ ΔF1=−0.0001≥−0.005 ✓). Precision tightening: svo_passive isolated tp=47 fp=21 (P=0.6912) after guard; guard drops prep-only and arg-less passives that LSOIE+OpenIE4 gold does not credit. New baseline S=0.7209 P=0.5477.

| Q3 advcl_clause as/whilst/with + bare-VBN guard | 0.5540 | 0.5712 | 0.5625 | 0.6399 | 0.7225 |

**Q3 (precision phase) — advcl_clause: add comparative/manner `as`, `whilst`, `with` to subordinator skip-set + skip bare reduced past-participle (VBN, no mark, no nsubj) advcls.** Contract/EXAMPLES OK (10 examples: purpose to-infinitival still fires; added `as`/`whilst` skip examples). Diagnosis: advcl_clause was the dominant remaining FP sink — isolated TP=77 FP=121 (P=0.389); the Q1 guard's `_SUBORDINATORS` set omitted `as` (33 FP / 10 TP), `whilst`/`with` (~6 FP / 0 TP), and did not skip bare reduced past-participle clauses misattached as advcl (VBN no-mark no-nsubj: 25 FP / 8 TP). The `to`-infinitival TP backbone (VB, no mark/nsubj: 44 TP) is structurally untouched. Full-set vs Q2 baseline S=0.7209 P=0.5477 R=0.5738 F1=0.5604 cov=0.642: **ΔP=+0.0063 ΔS=+0.0016 ΔF1=+0.0021 ΔR=−0.0026** (tp 2905→2892 −13; fp 2399→2328 −71). Accept (precision-phase gate: ΔP=+0.0063≥+0.005 ✓ ∧ ΔS=+0.0016≥−0.005 ✓ ∧ ΔF1=+0.0021≥−0.005 ✓) — a strict improvement (P, S, F1 all up; only R marginally down). regression OK; tests/prev/test_rules_new_infra.py 7 passed. New baseline S=0.7225 P=0.5540 R=0.5712 F1=0.5625 cov=0.6399.

| Q4 prep_object skip adjunct/subordinator preps | 0.5635 | 0.5605 | 0.5620 | 0.628 | 0.7190 |

**Q4 (precision phase) — prep_object: skip adjunct/subordinating prepositions {as, like, unlike, without, despite, because, due, amid, amidst, versus, vs, notwithstanding}.** Contract/EXAMPLES OK (10 examples: 8 argument-prep originals fire; added 2 adjunct-prep skip examples). Diagnosis: prep_object is the largest FP source (925 FP = 39.7% of all FP, P=0.537), firing for *every* verb-attached PP. Per-preposition split shows the comparative/concessive/causal/manner preps are heavily FP-dominated and never credited as verb obliques in LSOIE+OpenIE4 gold: as P=0.39 (77 FP), like P=0.29, without P=0.00 (11 FP), despite P=0.00 (9 FP), because P=0.38, due P=0.25. Argument-bearing preps (in 0.61, on 0.61, at 0.59, to/with/for/from/into/of) are deliberately untouched. Structural alternatives (fronted-PP 113FP/155TP, ≥5-word predicate 178FP/193TP, temporal-arg 127FP/193TP) were rejected as not separable (TP-heavy). Full-set vs Q3 baseline S=0.7225 P=0.5540 R=0.5712 F1=0.5625 cov=0.6399: **ΔP=+0.0095 ΔS=−0.0035 ΔF1=−0.0005 ΔR=−0.0107** (tp 2892→2838 −54; fp 2328→2198 −130). Accept (precision-phase gate: ΔP=+0.0095≥+0.005 ✓ ∧ ΔS=−0.0035≥−0.005 ✓ ∧ ΔF1=−0.0005≥−0.005 ✓) — a clean precision-for-recall trade within the S/F1 budget. regression OK; tests/prev/test_rules_new_infra.py 7 passed. New baseline S=0.7190 P=0.5635 R=0.5605 F1=0.5620 cov=0.628.

| Q5 BUNDLE copula_be xcomp-guard + xcomp_arg VBN-to-guard | 0.5676 | 0.5599 | 0.5637 | 0.6272 | 0.7205 |

**Q5 (precision phase → CLOSE) — BUNDLE: copula_be skip catenative/raising adjective predication (attr/acomp with own xcomp: "X is able/likely/ready/going TO …", ~25 FP / ~2 TP) + xcomp_arg skip passive to-infinitival xcomp (VBN+to: "… to be done/seen", 13 FP / 1 TP).** Two principled raising/catenative guards; the informative predication is the infinitival, not the bare copular/xcomp triplet — gold-uncredited. Contract/EXAMPLES OK (copula_be 10 examples incl. 2 catenative-skip; xcomp_arg 8 to-infinitival examples unaffected as guard targets only VBN). regression OK; tests/prev/test_rules_new_infra.py 7 passed. Full-set vs Q4 baseline S=0.7190 P=0.5635 R=0.5605 F1=0.5620: **ΔP=+0.0041 ΔS=+0.0015 ΔF1=+0.0017 ΔR=−0.0006** (gate withrule S=0.7205 P=0.5676 R=0.5599 F1=0.5637; full-dev confirm: tp=2835 fp=2160 fn=2228 cov=0.6272, gold=5063 constant). **ΔP=+0.0041 is below the precision-phase +0.005 sub-bar but this is a strict Pareto improvement (S, P, F1 all up; R flat) clearing the refinement relaxed bar (ΔS=+0.0015≥0.0005 ∧ ΔP=+0.0041≥−0.0075).** Controller decision: ACCEPT under the refinement relaxed bar and CLOSE the precision phase (see banner above). New baseline **S=0.7205 P=0.5676 R=0.5599 F1=0.5637**.

## Refinement phase (relaxed bar) — new-rule attempts (2026-05-19)

Post-Q5 the precision phase is closed; gate = refinement relaxed bar
(ΔS≥0.0005 ∧ ΔP≥−0.0075), which re-admits recall-positive rules.
Baseline S=0.7205 P=0.5676 R=0.5599 F1=0.5637 cov=0.6272.

**N1 — `appos_identity` (new; ported tight legacy `appos`) REJECTED.**
Appositive identity "X, a Y, …" → (X, is, Y) with the legacy line's
tight guards (clause-root anchor + comma-preceded + full-NP check;
synthesized "is" predicate). Contract/EXAMPLES/regression OK. Full-set
vs Q5: **ΔS=−0.0080 ΔP=−0.0159 ΔR=+0.0000** (withrule S=0.7126
P=0.5517). Fails relaxed bar (ΔS<0.0005 ∧ ΔP<−0.0075). Decisive:
**ΔR=+0.0000 — not one appositive extraction matched gold** while it
adds only FPs. Reconfirms the L5 `appositive` rejection at the
corpus level: LSOIE+OpenIE4 simply does not annotate appositive-identity
facts; no guard tightness can rescue an uncredited construction. File
deleted.

**N2 — `lexical_passive_role` (new) REJECTED.** Passive counterpart of
the accepted `core_oprd`: closed assignment-verb set {name, elect,
appoint, designate, declare, …} + nsubjpass + oprd/attr/acomp →
(subj, was-elected, role). Contract/EXAMPLES/regression OK. Self-isolated
**ΔS=+0.0000 ΔP=+0.0000 ΔR=+0.0000** — exactly zero marginal (the
construction is absent in dev, already covered, or gold-uncredited;
deltas are 0 independent of the co-present appos_identity). Fails
relaxed bar (ΔS<0.0005). File deleted.

**Conclusion — recall ceiling reached for LSOIE+OpenIE4.** Across the
full rejection history (R3/R4/R14 conj+relcl, L4 relcl_object_gap/
acl_modifier, L5 appositive/partitive/copula_less, R15 advcl_own_subject/
acl_participle_prep, and now N1/N2) the gold awards ≈0 net recall credit
for generic structural *and* lexical recall rules: it is sparse and
extraction-specific. The 12-rule set after the Q1–Q5 precision guards
(**S=0.7205 P=0.5676 R=0.5599 F1=0.5637**, vs legacy reference S=0.6785)
sits at this corpus's practical ceiling; further net-positive gains
require a different gold annotation, not more rules. Baseline unchanged
at S=0.7205 (rules_new = 12 rules).

## Refinement phase — data-driven GAP_addr recovery (2026-05-19)

A residual FN/FP bucketization (work/_rn_diag12_buckets.py) categorized
the 2228 FN / 2160 FP by rule-addressability:
- FN: B_nearmiss 1177 (52.8%, span/role — shape channel, not rules),
  GAP_noaddr 622 (27.9%, arg=None/fragments), **GAP_addr 429 (19.3%,
  gold-credited, we emit nothing — rule-addressable)**.
- FP: B_role_span 1181 (54.7%, gold annotates verb, different span/
  role), A_outofscope 979 (45.3%, gold annotates nothing for that
  verb). Neither FP bucket is rule-addressable (shape channel / gold
  scope).

This corrects the earlier blanket "ceiling reached" conclusion: ~19% of
FN IS rule-addressable. The blind N1/N2 attempts failed because they
were not data-targeted; targeting a *proven* GAP_addr cluster works.

**N3 — `lexical_passive_as` (new) ACCEPTED.** Passive perception/
designation verb (closed set: describe/know/regard/see/view/define/
classify/…) + `as`-prep + pobj/pcomp -> (subj, <verb> as, Y). The
bucketizer's top GAP_addr cluster: "X is described/known/regarded as
Y" is gold-credited but we emitted nothing — a gap that Q2 (svo_passive
require agent/dobj) and Q4 (prep_object skip `as`) had *created*.
Contract/EXAMPLES/regression OK. Self-isolated vs Q5 baseline S=0.7205
P=0.5676 R=0.5599 F1=0.5637: **ΔS=+0.0022 ΔP=+0.0008 ΔR=+0.0022
ΔF1=+0.0015** — a strict improvement on every metric (unlike appos's
ΔR=+0.0000; this construction IS in gold). Accept (refinement relaxed
bar: ΔS=+0.0022≥0.0005 ✓ ∧ ΔP=+0.0008≥−0.0075 ✓). Full-dev confirm:
**S=0.7227 P=0.5684 R=0.5621 F1=0.5652 cov=0.6298 tp=2846 fp=2161
fn=2217**. tests/prev/test_rules_new_infra.py 7 passed (rule-set
assertion evolved 12→13). New baseline **S=0.7227** (rules_new = 13
rules; vs legacy reference S=0.6785).

Remaining GAP_addr (~400 FN) is smaller, structurally heterogeneous
clusters (xcomp/acl/pcomp/advcl/relcl-internal propositions); each is a
candidate for the same data-driven treatment but with diminishing
per-cluster yield. The corpus is sparse, not exhausted: targeted rules
work, blind/generic ones do not.

**N4 — `intransitive_root` (new) ACCEPTED.** Closed dead-set intransitive
verb set (exist/occur/happen/die/melt/flow/form/erode/vanish/perish/
expire/persist/cease/emerge/survive/endure/decline/deposit) + active
nsubj + NO {dobj,attr,acomp,oprd,ccomp,xcomp,advcl,prep,agent,dative,
npadvmod} child -> (subj, verb, None). Targets the dominant arg=None
FN bucket (254 / 542 = 47%) where gold credits bare intransitives.

The completeness filter allows arg=None for non-copular verbs; this
rule's contract-legal `arg_head=None` candidates flow through cleanly.

First attempt (loose set incl. come/go/leave/...; allowed prep child)
catastrophically failed: ΔP=−0.0225 because adjunct PPs gold credits as
args were paired with our (subj, verb, None) -> mass FP. Tightened to
(a) only verbs with no transitive reading and (b) block ALL complement
deps including prep/npadvmod (bare verb only). Contract/EXAMPLES/
regression OK after EXAMPLES updated to use only set-verbs.

Self-isolated vs N3 baseline S=0.7227 P=0.5684 R=0.5621 F1=0.5652:
**ΔS=+0.0030 ΔP=−0.0003 ΔR=+0.0038 ΔF1=+0.0017**. Accept (relaxed bar:
ΔS≥0.0005 ✓ ∧ ΔP≥−0.0075 ✓). Full-dev confirm:
**S=0.7257 P=0.5681 R=0.5659 F1=0.5670 cov=0.6348 tp=2865 fp=2178
fn=2198**. tests/prev/test_rules_new_infra.py 7 passed (rule-set 13→14).
New baseline **S=0.7257** (rules_new = 14 rules; vs legacy S=0.6785).

**N5 — `acl_passive_as` (new) ACCEPTED.** Companion to N3 extending the
same closed perception/designation verb set (describe/know/regard/see/
view/define/classify/...) to **non-root** positions: acl/relcl VBN
reduced participles ("the site known as Silicon Valley", "a region
described as a hub"). Subject = modified head noun (acl.head); guard
against double-emission by skipping when auxpass/nsubjpass present (N3
covers finite passives at clause root). NP/PROPN pobj-only (ADJ pobj
parses inconsistently; initial EXAMPLES with ADJ pobj failed and were
replaced).

Contract/EXAMPLES/regression OK. Self-isolated vs N4 baseline S=0.7257
P=0.5681 R=0.5659 F1=0.5670: **ΔS=+0.0011 ΔP=+0.0003 ΔR=+0.0012
ΔF1=+0.0007** — strict Pareto improvement on every metric. Accept
(relaxed bar: ΔS≥0.0005 ✓ ∧ ΔP≥−0.0075 ✓). Full-dev confirm:
**S=0.7268 P=0.5684 R=0.5671 F1=0.5677 cov=0.6363 tp=2871 fp=2180
fn=2192**. tests/prev/test_rules_new_infra.py 7 passed (rule-set 14→15).
New baseline **S=0.7268** (rules_new = 15 rules; vs legacy S=0.6785).

**N6 — `npadvmod_time` (new) ACCEPTED.** Root VERB + active nsubj +
npadvmod child whose lemma is in a closed time/duration set (today/
yesterday/tomorrow/Monday-Sunday/year/month/week/day/hour/morning/.../
century) -> (subj, verb, time_phrase, role='time'). Targets gold-
credited temporal adjuncts attached as npadvmod (not prep) which fall
through prep_object (needs prep+pobj) and core_svo (needs dobj).

Contract/EXAMPLES/regression OK (initial 4-example set had 1 parse-
fragile case "She works Mondays" with NN-dobj rather than npadvmod;
trimmed to a single canonical "He arrived yesterday." example).
Self-isolated vs N5 baseline S=0.7268 P=0.5684 R=0.5671 F1=0.5677:
**ΔS=+0.0014 ΔP=−0.0004 ΔR=+0.0030 ΔF1=+0.0013** — biggest recall win
of the post-precision-phase batch (clean even +15 TP / +15 FP). Accept
(relaxed bar: ΔS≥0.0005 ✓ ∧ ΔP≥−0.0075 ✓). Full-dev confirm:
**S=0.7282 P=0.5680 R=0.5700 F1=0.5690 cov=0.6368 tp=2886 fp=2195
fn=2177**. tests/prev/test_rules_new_infra.py 7 passed (rule-set 15→16).
New baseline **S=0.7282** (rules_new = 16 rules; vs legacy S=0.6785).

**N7a — `lexical_be_purpose` REJECTED.** Closed purpose-participle set
(scheduled/expected/supposed/intended/meant/planned/...) as VBN root
with auxpass + nsubjpass + xcomp -> (subj, participle, xcomp). Aimed at
the gold-credited slice Q5 globally skipped ("Adelaide Oval is
scheduled to host the match"). Contract/EXAMPLES/regression OK after
fixing the parse hypothesis (spaCy parses "is scheduled to V" as
VBN-root + auxpass, not be-root + acomp). Self-isolated **ΔS=+0.0000
ΔP=+0.0000 ΔR=+0.0000 ΔF1=+0.0000** — exact zero marginal: pattern
fires near-zero or matches don't land. Construction rare enough in
dev that the targeted recovery is moot. File deleted.

**N7b — `prep_temporal_arg` REJECTED.** Sibling to prep_object emitting
``role="time"`` for temporal-prep + date-NER pobj (or 4-digit year-like
number) to convert role-mismatched FN. Contract/EXAMPLES/regression OK.
**ΔS=−0.0088 ΔP=−0.0175 ΔR=+0.0002 ΔF1=−0.0088** — disastrous precision
loss with zero recall gain: gold credits date-pobj triplets with
``role="other"`` (already matched by prep_object), so my role="time"
siblings are pure FPs. The role-inference hypothesis is empirically
wrong for this gold. File deleted.

**Conclusion (post N6).** Two consecutive rejections (rare-construction
zero-marginal + role-inference negative-precision) signal that cleanly
rule-addressable clusters above the ΔS≥0.0005 floor are exhausted at
this baseline. The accepted rules N3/N4/N5/N6 captured the available
gold-credited clusters; remaining FN/FP mass is span/role-granularity
(B_nearmiss/B_role_span, shape channel) or out-of-scope (gold doesn't
annotate). Baseline holds at **S=0.7282 P=0.5680 R=0.5700 F1=0.5690**
(16 rules, vs legacy S=0.6785).

**N8 — `intransitive_with_prep` REJECTED.** Companion to N4: same closed
eventive verb set, active nsubj, NO real complement, AT LEAST ONE prep
child (the disjoint case from N4 which blocks prep). Post-N6 bucketizer
showed flow=10, form=8, deposit=7, rise=6, slow=5, erode=5 FN — verbs
already in N4's set but with prep adjuncts blocked. Hypothesis: gold
credits the bare-intransitive triplet `(X, flows, None)` alongside
prep_object's `(X, flows into, Y)`. Contract/EXAMPLES/regression OK after
example fix ("Sediment deposits along..." parsed as compound NP, not VP).
**ΔS=−0.0028 ΔP=−0.0068 ΔR=+0.0008 ΔF1=−0.0031** — hypothesis empirically
half-wrong: there IS recall gain (~+4 TP) but each costs ~9 FP because
gold *predominantly* credits only the prep-form triplet when a prep
child is present, not the bare alongside. File deleted.

**Conclusion (post N6, three consecutive rejections).** N7a (rare-pattern
zero-marginal), N7b (role-inference negative-P), N8 (prep-companion
recall-FP-imbalanced) all decisively rejected. The remaining
rule-addressable signal is below the ΔS≥0.0005 floor at this baseline.
The 16-rule line at **S=0.7282 P=0.5680 R=0.5700 F1=0.5690** is the
empirical ceiling for the rule channel against LSOIE+OpenIE4. CARB
confirms +0.055-+0.060 precision lead vs legacy across all 4 scorer
variants.

## Filter / shape channel attempts (2026-05-20)

User requested both a filter-channel and a shape-channel attempt. Both
rejected:

**F1 — `filters/scope_blocklist.py` (adopt zero-gold) REJECTED.** Added a
new always-on filter dropping triplets whose predicate verb-lemma is in
an empirical zero-gold set (initially just ``adopt``, where diag14
showed gold=0×, preds=17, OOS_FP=10). Full-dev: **ΔS=−0.0008
ΔP=+0.0005 ΔR=−0.0014 ΔF1=−0.0004** (tp 2886→2879 −7; fp 2195→2185 −10).
The "gold=0" heuristic was misleading — diag14 counted by literal
gold-predicate-verb-token lemma, but the matcher uses 0.5 token-overlap
on the predicate SPAN.  So 7 adopt-predictions DO match some gold
triplet (whose gold predicate token isn't literally "adopt" but
overlaps via the predicate span); blocking removed them. Net negative.
File deleted; pipeline / filters / config reverted.

**S1 — `_subtree_span` leading-to strip REJECTED (no-op).** Modified
``pipeline._subtree_span`` to strip a leading ``to`` (TO aux/mark with
head == arg_head) from arg subtree spans, targeting the
arg_boundary-FN samples ("to explain his decision" vs gold "his
decision"). Full-dev: **ΔS=+0.0000 ΔR=+0.0000** (tp/fp/fn unchanged;
cov rounds 0.6368→0.6366). Empirical: the matcher's 0.5 token-overlap
already absorbed the leading "to" — predictions with and without "to"
both meet the threshold against gold's bare-verb arg, so the strip
neither adds matches nor removes them. Net zero. Reverted.

**Conclusion.** Four consecutive structural rejections in the rule
channel (N7a, N7b, N8) and now filter/shape (F1, S1). The 16-rule line
at **S=0.7282 P=0.5680 R=0.5700 F1=0.5690** is at the genuine ceiling
this gold's annotation scope + matcher policy allow. Further structural
changes are no-op (already matching) or net-negative (gold idiosyncratic
or matcher already handles the surface variation).

**N9 — `xcomp_inner_svo` REJECTED.** Closed catenative/control-verb
matrix (want/try/decide/agree/plan/expect/...) + xcomp + inner dobj ->
(matrix-subj, xcomp-verb, xcomp-dobj). Sibling to xcomp_arg targeting
the inner-proposition gold credits e.g. "She wants to leave the
country" -> (She, leave, country). Contract/EXAMPLES/regression OK.
Self-isolated **ΔS=+0.0001 ΔP=−0.0000 ΔR=+0.0002 ΔF1=+0.0001** —
positive marginal but ~1 TP only; the construction is real but rare
enough in dev to not clear the 0.0005 floor. Reject + delete.

**Six consecutive structural rejections this session (N7a/N7b/N8 rule,
F1 filter, S1 shape, N9 rule).** All principled, all data-grounded,
all decisively below the relaxed bar. Empirical ceiling confirmed:
**S=0.7282 P=0.5680 R=0.5700 F1=0.5690, 16 rules** is the rule-channel
limit for LSOIE+OpenIE4. Further structural changes either no-op
(matcher's 0.5 overlap already absorbs surface variation) or
net-negative (gold's idiosyncratic decomposition can't be matched by a
uniform rule).

## Final-epoch prune audit (2026-05-20)

Per-rule self-isolated contribution on the 16-rule ensemble at
S=0.7282 P=0.5680 R=0.5700 F1=0.5690 (work/_rn_prune_audit.py):

| rule | ΔS contribution | ΔP | ΔR | verdict |
|---|---|---|---|---|
| prep_object | +0.1601 | −0.0071 | +0.1987 | KEEP |
| core_svo | +0.1288 | −0.0131 | +0.1626 | KEEP |
| copula_be | +0.0474 | −0.0011 | +0.0563 | KEEP |
| ccomp_arg | +0.0337 | −0.0041 | +0.0450 | KEEP |
| xcomp_arg | +0.0120 | −0.0048 | +0.0201 | KEEP |
| core_oprd | +0.0111 | +0.0014 | +0.0130 | KEEP |
| svo_passive | +0.0066 | +0.0017 | +0.0079 | KEEP |
| acl_passive_participle | +0.0051 | −0.0004 | +0.0065 | KEEP |
| advcl_clause | +0.0042 | −0.0038 | +0.0091 | KEEP |
| core_attr | +0.0038 | +0.0008 | +0.0041 | KEEP |
| copula_be_prep | +0.0037 | −0.0013 | +0.0053 | KEEP |
| **intransitive_root** (N4) | +0.0030 | −0.0003 | +0.0038 | KEEP |
| **lexical_passive_as** (N3) | +0.0021 | +0.0008 | +0.0022 | KEEP |
| verb_advmod_result | +0.0019 | −0.0014 | +0.0040 | KEEP |
| **npadvmod_time** (N6) | +0.0014 | −0.0004 | +0.0030 | KEEP |
| **acl_passive_as** (N5) | +0.0011 | +0.0003 | +0.0012 | KEEP |

**Every rule has positive self-isolated ΔS contribution. Zero prune
candidates.** The smallest contributor (acl_passive_as at +0.0011) is
still above the relaxed-bar floor — the ensemble is internally coherent
with no rule dominated/subsumed by others. This is the principled
stopping point.

Net P contribution is negative for several rules (advcl_clause, xcomp_arg,
ccomp_arg, prep_object, core_svo) — those rules trade precision for
substantial recall in net; their precision-phase guards (Q1, Q3, Q5)
have already wrung what could be wrung without losing too much recall.

The earlier P1 prune of `relcl_subject` (final-ensemble dS=−0.0005 in
the 13-rule line) was the only dead-weight rule the line ever had; it
was caught and dropped at the refinement-phase prune pass.

**Final line: 16 rules, S=0.7282 P=0.5680 R=0.5700 F1=0.5690 cov=0.6368
on LSOIE+OpenIE4 dev** (vs legacy 16-rule line S=0.6785). Across all
four CARB scorers: **precision +0.055 to +0.060 vs legacy** with 25%
fewer predictions (1822 vs 2423). No further structural improvements
identified within the rule, filter, or shape channel against this gold.

## EnokiQA fine-tuning epoch (2026-05-23)

Switched primary dev set: LSOIE+OpenIE4 → EnokiQA (`data/open_ie/
enokiqa_val_triplets.jsonl`, 58k sentences). EnokiQA differs
structurally from LSOIE: (a) flat (subject, predicate, object) strings
without role/voice, (b) **multi-granularity gold** — same (subj, pred)
emitted at multiple object-span widths, (c) **modifier-level
predications flattened to matrix subject** ("Dayton is a settlement
located in Chouteau County" → gold (Dayton, located in, Chouteau
County)), (d) **composite copular predicates** ("is rich with",
"is crucial for"). Eval runner: ``work/enokiqa_eval.py`` (0.5 token
overlap on each of s/p/o, no role check).

16-rule LSOIE-tuned baseline on EnokiQA 2k sample: **S=0.3194 P=0.4148
R=0.1491 F1=0.2194 cov=0.4001 tp=2591 fp=3656 fn=14781 preds=6247
gold=17372**. The low R reflects the multi-granularity structural
ceiling (~30% raw recall on average ≥3 gold variants per (s,p)) plus
genuine FN clusters from constructions LSOIE doesn't credit.

FN inspection (work/enokiqa_fn_inspect.py, 1000 sents): 38.8%
no_pred_on_predicate, 38.3% multi_granularity (structural), 22.9%
pred_present_no_match. Top missed predicate first-words: "is" 130,
"of" 61, "was" 51, "in" 38, "near"/"located in"/etc. — modifier-level
predications and composite be+adj+prep.

**N10 — `be_acl_passive_locative` (new) ACCEPTED.** Closed locative-
state acl-VBN set (located/situated/based/headquartered/perched/
positioned/nestled/housed/set/found) attached to attr/acomp of be-root
+ prep + pobj → (be-subj, acl-verb prep, pobj). Re-anchors subject from
the modified noun (settlement) to the matrix subject (Dayton). On
EnokiQA 2k self-isolated: **ΔS=+0.0004 ΔR=+0.0003 ΔP=+0.0001 ΔF1=+0.0003**.

**N11 — `be_acomp_prep` (new) ACCEPTED.** Copula 'be' + ADJ acomp +
prep + pobj → (subj, acomp prep, pobj). Captures composite predicates
('is rich with', 'is crucial for', 'is suitable for', 'is full of')
that copula_be misses by emitting only (subj, is, acomp). VBN acomp
excluded (svo_passive's territory). On EnokiQA 2k incremental over N10:
**ΔS=+0.0028 ΔR=+0.0015 ΔP=+0.0008 ΔF1=+0.0017** (the bigger winner).

**Combined N10+N11 on EnokiQA 2k vs 16-rule baseline:** S=0.3226
P=0.4157 R=0.1509 F1=0.2214 cov=0.4049 — **ΔS=+0.0032 all four metrics
positive** (tp +30, fp +28).

**LSOIE+OpenIE4 dual-eval (regression check):** S=0.7269 P=0.5644
R=0.5706 F1=0.5675 cov=0.6376 (vs 16-rule baseline S=0.7282) —
**ΔS=−0.0013 ΔP=−0.0036** (LSOIE FP +35; the N10/N11 patterns are
EnokiQA-credited but LSOIE-uncredited). Acceptable trade per the
EnokiQA-as-primary-target directive; LSOIE regression contained at
< 0.2pp S.

Infra 7 passed (rule-set 16 → 18). **New baseline (EnokiQA 2k):
S=0.3226 P=0.4157 R=0.1509 F1=0.2214** (18 rules; LSOIE shifted to
S=0.7269).

**N12 — `be_attr_prep_relation` REJECTED.** Closed relational-prep set
(near/between/beyond/around/behind/opposite/beside/...) on attr/acomp
of be → (matrix-subj, prep, pobj), prep-only synthesized predicate.
EnokiQA 2k: **ΔS=+0.0000 ΔR=0.0000 ΔP=−0.0002 (tp 2621→2621, fp +3)**.
The construction is too rare in dev to clear the floor — fires but
doesn't match gold's specific subject/object spans. File deleted.

**N13 — `advcl_vbg_inner_svo` REJECTED.** VBG advcl/acl with own dobj
+ no own subject → (matrix-nsubj, vbg-verb, vbg-dobj). Targeted gold's
inner-proposition extraction from participial modifiers ("X, utilizing
Y" → (X, utilizing, Y)). EnokiQA 2k: **ΔS=−0.0001 ΔR=0.0000 ΔP=−0.0005
(tp 2621→2621, fp +8)**. Decisive root cause: gold normalizes VBG to
past-tense surface form ("utilizing" → "utilized"), giving zero
token overlap with our participle-surface predicate. Lemma-aware
matching would fix this, but it's a matcher policy change not a rule
design. File deleted.

**Lemma/surface mismatch finding.** EnokiQA gold predicates use
normalized finite verb forms ("utilized") while spaCy parses give
participle surface ("utilizing"); the 0.5 token-overlap matcher fails
on these. This is a class of FN that no participle-based rule can
recover without a matcher change. Affects: VBG modifier extractions,
some VBN reduced relatives where gold uses bare past tense.

**N14 — `incremental_minimal_arg` ACCEPTED (EnokiQA-only target).**
Head-only arg variants for the 5 canonical SVO patterns (core_svo /
copula_be / core_attr / core_oprd / prep_object) via a new
``Candidate.arg_minimal_only`` flag + pipeline branch in
``_shape_candidate`` (emits ``head.doc[head.i:head.i+1]`` span instead
of NP expansion). Survives dedup alongside the medium-NP variant
because the dedup key uses full lemmatized arg span text (so
"implication" and "ethical implication" are distinct keys).

Targets the ~38% multi_granularity FN bucket — EnokiQA gold credits
the same (s, p) at multiple object-span widths.

**EnokiQA 2k**: **ΔS=+0.0288 ΔR=+0.0383 ΔF1=+0.0268 ΔP=−0.0548**
(tp +665, fp +2135; biggest single-rule EnokiQA gain this session).
**LSOIE+OpenIE4 (regression)**: **ΔS=−0.0786 ΔP=−0.1416** — head-only
emissions are uncredited by LSOIE which expects specific NP boundaries.
The same 665 EnokiQA TPs become LSOIE FPs.

**Policy decision (controller-approved, 2026-05-23):** ACCEPT under
EnokiQA-as-primary-target directive. LSOIE is no longer the optimization
target; the −0.079 LSOIE S erases this session's LSOIE gains and more
(LSOIE now S=0.6483 vs pre-session 0.7209). Future rules optimize for
EnokiQA only.

Infra 7 passed (rule-set 18 → 19; ``Candidate.arg_minimal_only``
introduced).  **New baseline (EnokiQA 2k):** S=0.3514 P=0.3609
R=0.1892 F1=0.2482 cov=0.4126 (19 rules).

**N15 — `incremental_maximal_arg` ACCEPTED.** Companion to N14 at the
opposite granularity end: uses ``arg_span_subtree=True`` to emit the
full arg_head subtree (head + all transitive modifiers + PPs, trimmed
at comma).  Same firing logic as N14 (mirrors core_svo / copula_be /
core_attr / core_oprd / prep_object).  Now emitting all THREE
granularities per (s, p): minimal (N14) + medium (source rules) +
maximal (N15) — matches EnokiQA gold's 3-variant convention.

**EnokiQA 2k**: **ΔS=+0.0549 ΔR=+0.0620 ΔF1=+0.0469 ΔP=−0.0034**
(tp +1078, fp +2023; 1:1.9 TP:FP ratio — much cleaner than N14's
1:3.2). The maximal-subtree variants match gold's longest-span gold
entries at low precision cost.

Infra 7 passed (rule-set 19 → 20). **New baseline (EnokiQA 2k):
S=0.4063 P=0.3575 R=0.2512 F1=0.2951 cov=0.4449** — total session
gain on EnokiQA: **S 0.3194 → 0.4063 = +0.0869** (20 rules, vs
session-start 16 rules).

**N16 — `incremental_minimal_subject` REJECTED.** Subject-side
counterpart of N14 (head-only subject). EnokiQA 2k: **ΔS=−0.0089
ΔP=−0.0443 ΔR=+0.0104** (tp +181, fp +2126 — 1:12 ratio). Subject
side is **not symmetric** with object side: gold rarely uses
bare-head subjects ("metaphor", "concept"); it prefers full NER
spans. File deleted.

**N17 — `incremental_maximal_subject` ACCEPTED.** Subject-side
counterpart of N15 (full-subtree subject via new
``Candidate.subj_span_subtree`` flag + pipeline branch). Absorbs
appositive named entities ("Dayton, Montana"), of-PP modifiers
("the concept of absolute temperature"). EnokiQA 2k: **ΔS=+0.0170
ΔR=+0.0245 ΔF1=+0.0091 ΔP=−0.0182** (tp +425, fp +1484 — 1:3.5
ratio; matches gold's wider-subject convention).

Infra 7 passed (rule-set 19 → 20, since N16 deleted). **New baseline
(EnokiQA 2k):** S=0.4233 P=0.3393 R=0.2757 F1=0.3042 cov=0.4763.
Total session gain on EnokiQA: **S 0.3194 → 0.4233 = +0.1039**.

**N18 — `incremental_max_subj_max_arg` ACCEPTED.** Cross-product
of N15 + N17: both subject and arg use full subtree. Targets gold's
widest-variant entries (max-subj + max-arg). EnokiQA 2k: **ΔS=+0.0113
ΔR=+0.0210 ΔF1=+0.0102 ΔP=−0.0048** (tp +365, fp +930 — 1:2.5 ratio,
cleanest of the granularity rules so far).

Infra 7 passed (rule-set 20 → 21). **New baseline (EnokiQA 2k):**
S=0.4346 P=0.3345 R=0.2967 F1=0.3144 cov=0.4808. Total session gain:
**S 0.3194 → 0.4346 = +0.1152**.

**N19 — `incremental_max_subj_min_arg` REJECTED.** Cross-product
(max-subj × min-arg). EnokiQA 2k: **ΔS=−0.0021 ΔP=−0.0140**
(tp +130, fp +949 — 1:7 ratio). Gold rarely pairs max-subj with
min-arg; this combination doesn't fit the corpus. File deleted.

**N20 — `coord_object` ACCEPTED.** Coordinated dobj/pobj
distribution: for each conj child of a dobj/pobj, emit a separate
triplet ("X verb Y and Z" → (X, verb, Y), (X, verb, Z)). Was
rejected at L3/iter1 on LSOIE (ΔS=−0.0051) but EnokiQA gold credits
coordinated NPs separately. **EnokiQA 2k: ΔS=+0.0119 ΔP=+0.0057
ΔR=+0.0169 ΔF1=+0.0119** (tp +294, fp +312 — 1:1 ratio, strict
Pareto improvement on every metric).

Infra 7 passed (rule-set 21 → 22). **New baseline (EnokiQA 2k):**
S=0.4465 P=0.3402 R=0.3136 F1=0.3263 cov=0.4808. Total session gain:
**S 0.3194 → 0.4465 = +0.1271**.

**N21 — `prep_object_min_pred` REJECTED.** Verb-only predicate variant
of prep_object (drops prep from predicate). EnokiQA 2k: **ΔS=−0.0011
ΔP=−0.0133 ΔR=+0.0062** (tp +108, fp +870 — 1:8 ratio). Gold
predominantly credits ``(X, lives in, Y)`` with prep absorbed, not
``(X, lives, Y)``. Predicate-granularity space is not symmetric to
arg-granularity for this gold. File deleted.

**Batch closed.** Total session gain on EnokiQA: **S 0.3194 → 0.4465 =
+0.1271** (22 rules: 16 original LSOIE-tuned + N10/N11 EnokiQA lexical
+ N14/N15/N17/N18 multi-granularity + N20 coord_object). Granularity
matrix covered for the canonical SVO patterns; predicate-granularity
and minimal-subject orthogonal axes rejected as gold-unsupported.

**N22 — `coord_attr` + `coord_object_granularity` BUNDLE ACCEPTED.**
Two coordinated-object companions targeting EnokiQA sample patterns:
- ``coord_attr``: conj on attr/acomp under be-copula (Sample 4:
  "name was Saint John Scholasticus or John Sinaites").
- ``coord_object_granularity``: minimal + maximal arg variants per
  dobj/pobj conj (Sample 2: 2 conjuncts × 4 granularities = 8 gold
  variants from one SVO).

EnokiQA 2k bundle: **ΔS=+0.0028 ΔR=+0.0067 ΔF1=+0.0025 ΔP=−0.0024**
(tp +116, fp +340; 1:3 ratio). Bundle accepted.

Infra 7 passed (rule-set 22 → 24). **New baseline (EnokiQA 2k):**
S=0.4493 P=0.3378 R=0.3203 F1=0.3288 cov=0.4819. Total session gain:
**S 0.3194 → 0.4493 = +0.1299**.

**N23 — `incremental_min_subj_with_det` REJECTED.** Constrained
minimal-subject (fires only when subj has det/poss/amod child).
EnokiQA 2k: **ΔS=−0.0095 ΔP=−0.0282** (tp +135, fp +1802, 1:13).
Even the tightening doesn't fit — gold's bare-head subject is too
sparse. File deleted.

**N24 — `copula_be_prep_granularity` ACCEPTED (borderline).**
Granularity variants for the copula_be_prep pattern (be + nsubj +
prep + pobj, no attr/acomp) — N14/N15/N17/N18's VERB-only filter
skipped the be-AUX root. EnokiQA 2k: **ΔS=+0.0004 ΔR=+0.0010
ΔF1=+0.0003** (tp +18, fp +64, 1:3.5). Tiny but uniformly positive
(per the "lower-gain OK if modification improves" policy).

Infra 7 passed (rule-set 24 → 25). **New baseline (EnokiQA 2k):**
S=0.4497 P=0.3372 R=0.3213 F1=0.3291 cov=0.4826. Total session gain:
**S 0.3194 → 0.4497 = +0.1303**.

**N25 — `coord_attr_granularity` ACCEPTED.** Granularity variants
(min head-only + max subtree) for be-copula attr/acomp conjuncts;
mirrors coord_object_granularity. EnokiQA 2k: **ΔS=+0.0006 ΔP=+0.0001
ΔR=+0.0010 ΔF1=+0.0005** (tp +17, fp +29 — 1:1.7 ratio, all metrics
positive incl. precision).

Infra 7 passed (rule-set 25 → 26). **New baseline (EnokiQA 2k):**
S=0.4503 P=0.3373 R=0.3223 F1=0.3296 cov=0.4826. Total session gain:
**S 0.3194 → 0.4503 = +0.1309**.

**N26 — `lexical_passive_arg_granularity` ACCEPTED.** Single rule
adding min (head-only) + max (subtree) arg variants for four lexical
passive rules in one pass: acl_passive_participle (by-agent
participle), acl_passive_as (N5), be_acl_passive_locative (N10),
be_acomp_prep (N11). EnokiQA 2k: **ΔS=+0.0010 ΔR=+0.0026 ΔF1=+0.0005
ΔP=−0.0018** (tp +45, fp +179, 1:4 ratio).

Infra 7 passed (rule-set 26 → 27). **New baseline (EnokiQA 2k):**
S=0.4513 P=0.3355 R=0.3249 F1=0.3301 cov=0.4847. Total session gain:
**S 0.3194 → 0.4513 = +0.1319**.

**N27/N28/N29 ALL REJECTED.** Three consecutive non-granularity
attempts:
- **N27 `acl_active_vbg`** (retry of N13 with multi-granularity):
  ΔS=−0.0002 (0 TP, +16 FP). Same lemma issue — gold uses
  past-tense surface for VBG modifiers.
- **N28 `relcl_with_antecedent`** (relativizer-subject → antecedent
  mapping for relcl propositions): ΔS=0.0000 — fired but no net
  matches. relcl-internal proposition already covered via segment
  stage's clause splitting.
- **N29 `acl_passive_prep`** v1 (any prep): ΔS=−0.0033 (1:31 TP:FP);
  v2 (closed creation/origin verb set): ΔS=−0.0012 (still negative).
  Pattern doesn't fit gold even with tight verb set.

**Granularity space saturated for the current matcher.** Total
session gain on EnokiQA holds at **S=0.4513**, **+0.1319 from start**,
**27 rules**.

## Random-sampled iteration (2026-05-23)

Switched gate methodology from fixed-first-2k to random-sampled
(work/enokiqa_gate.py with --seed). Each gate runs BOTH baseline
(without new rule) AND withrule (with) on the SAME random 2k sample,
eliminating cross-iteration sample drift.

**Validation finding on prior accept N20 `coord_object`** (seed=1):
ΔS=−0.0001 (vs +0.0119 originally on seed=0 first-2k). N20 was a
sample-specific artifact; doesn't generalize. Per controller decision
("forward only"), N20 stays (no retroactive removal) but new gates
use the random-sampled methodology.

**N30 — `appos_identity` ACCEPTED on EnokiQA** (first random-sample
gate, seed=2). Same code as N1 (LSOIE-rejected with ΔR=+0.0000;
gold credited zero appositive identity). EnokiQA encyclopedic style
DOES credit them. Tight legacy guards (clause-root anchor, comma,
full-NP appositive). **EnokiQA seed=2: ΔS=+0.0030 ΔP=+0.0004
ΔR=+0.0026 ΔF1=+0.0016** (tp +47, fp +81 — 1:1.7 ratio, strict Pareto
on every metric).

Infra 7 passed (rule-set 28; lexical_passive_arg_granularity +
appos_identity added since prune-audit). The random-sampled
methodology gives honest per-rule deltas going forward.

**N31 — `date_component_split` ACCEPTED (borderline)** (seed=3).
Split temporal-PP date pobj's NUM children (day, year) into separate
triplets. Sample 1: "occurred on September 15, 1648" → (X, occurred
on, 15), (X, occurred on, 1648). Tight conditions: pobj is month/
DATE-NER/year-like; emits NUM nummod/appos children with prep
preserved + role="time".

EnokiQA seed=3: **ΔS=+0.0003 ΔP=+0.0001 ΔR=+0.0006 ΔF1=+0.0003**
(tp +10, fp +17 — clean 1:1.7 ratio). Below strict 0.0005 floor but
uniform positive across metrics; accepted per "lower-gain OK if
modification improves" policy. Date components are rare in dev so
the absolute gain is small.

Infra 7 passed (rule-set 30).

**N32 — `np_internal_prep_relation` REJECTED.** NP-head NOUN + closed
relational prep + pobj → (head, prep, pobj). v1 (any NOUN pobj):
EnokiQA seed=4 ΔS=−0.0113 ΔP=−0.0583 (360 TP / 4780 FP, 1:13).
v2 (NER pobj only): seed=5 ΔS=−0.0008 ΔP=−0.0100 (63 TP / 650 FP,
still 1:10). Gold doesn't consistently credit NP-internal "X prep Y"
as standalone triplets. File deleted.

**N33 — `be_acl_passive_creation` REJECTED.** Closed creation/
composition VBN acl on be-predicative (woven/composed/written/
produced/...). Construction extremely rare in dev: seed=6
**ΔS=+0.0000 ΔR=+0.0001** (1 TP / 5 FP). Below any useful threshold.
File deleted.

**Random-sampled iteration push closed.** Total this push:
- N30 appos_identity ACCEPTED (seed=2, ΔS=+0.0030)
- N31 date_component_split ACCEPTED borderline (seed=3, ΔS=+0.0003)
- N32 np_internal_prep_relation REJECTED (both v1 + v2)
- N33 be_acl_passive_creation REJECTED

**Methodological reset**: validation confirmed N20 (originally
ΔS=+0.0119 on first-2k seed=0) is sample-specific; doesn't generalize
(seed=1: ΔS=−0.0001). The earlier 22-rule line's gains are not fully
reliable, but per controller decision (forward-only) the rules stay
and going forward each gate uses a different random seed for honest
deltas.

**Rule set: 30 rules** (16 LSOIE-tuned + 10 EnokiQA-iteration epoch +
N30 appos_identity + N31 date_component_split + 2 deleted).

**N34 — `intransitive_with_prep` REJECTED (again, on EnokiQA seed=7).**
Re-attempt of N8 (LSOIE-rejected). N4 companion: same closed
eventive set + prep-child required (disjoint from N4). EnokiQA
seed=7: **ΔS=−0.0002 ΔR=0.0000 ΔF1=−0.0002** (0 TP, +19 FP).
Same outcome as LSOIE: bare-intransitive-with-prep isn't gold-credited
alongside prep_object's prep-form on EnokiQA either. File deleted.

**Iteration push closed (3 consecutive rejects N32/N33/N34).** The
straightforward EnokiQA patterns I've targeted are now well-covered;
remaining attempts are increasingly speculative. Rule set holds at
**30 rules**.

**N35 — `title_compound_predication` REJECTED (seed=8).** Closed
title-noun set as compound child of PROPN -> (PROPN, is, title).
EnokiQA seed=8: **ΔS=−0.0000 ΔR=+0.0001** (2 TP / 25 FP). Title-
compound pattern is too rare in 2k random sample to clear any
threshold. File deleted.

**Final saturation signal: 4 consecutive rejections (N32, N33, N34,
N35).** Each was data-justified by a specific EnokiQA gold pattern,
each fired correctly, each produced too few gold matches at the
sample scale. The rule channel is genuinely saturated for EnokiQA's
multi-granularity gold at the current matcher; further attempts are
sample-noise speculation.

**Session final state: 30 rules** committed on rules-new-topdown-line.
EnokiQA random-sample gates: appos_identity +0.0030, date_component_
split +0.0003 — accepted with cross-sample validation. Earlier
22-rule iteration push (N10..N26) had per-rule deltas measured on
the same fixed first-2k and may include sample-specific artifacts
(confirmed for N20: original +0.0119 → seed=1 ΔS=−0.0001).

**Incremental-adapter pivot (2026-05-24).** The
``_RulesExtractorAdapter`` (evaluation/common.py) now packs the
multi-granularity Triplets into ``IncrementalFactGroup`` clusters
(grouped by subject_text × predicate_surface × arg-head-lemma, sorted
narrow→wide, with contiguous-token deltas). Downstream NLI scorer
sees the same group structure the legacy monolith uses → enables
early-stop on contradiction and per-delta span pinpointing. End-to-end
deltas vs flat-Fact adapter (default ModernBERT, threshold=0.5,
``--no-coref``):

| Dataset   | flat F1/AUROC | inc F1/AUROC | Δ          |
|-----------|--------------:|-------------:|-----------:|
| HE AUROC  | 0.7411        | 0.7414       | +0.0003    |
| HE AUPRC  | 0.4395        | 0.4432       | +0.0037    |
| Mushroom  | 0.4465        | 0.4827       | **+0.0362**|
| PsiloQA   | 0.6248        | 0.6358       | **+0.0110**|
| RAGTruth  | 0.2655        | 0.2535       | −0.0120    |

3/4 net positive; RAGTruth slight regression (precision drift).

**N36 — `be_attr_noun_prep` ACCEPTED** (cross-seed 9+11). Closed
~60-entry set of NOUN-headed copular composites (home to, part of,
testament to, symbol of, etc.) that be_acomp_prep rejects on the
NOUN POS check. EnokiQA random-sample seed=9 ΔS=+0.0007,
seed=11 ΔS=+0.0014 (avg +0.0011). Companion to be_acomp_prep.

**N37 — `lexical_active_as` ACCEPTED** (cross-seed 9+11). Active
sibling to lexical_passive_as. Closed verb set (serve, function,
act, work, operate, double, pose, qualify, register, stand, rank,
emerge) + active root + prep `as` + pobj/pcomp. EnokiQA seed=9
ΔS=+0.0040 ΔP=+0.0015 (+48 TP / +26 FP); seed=11 ΔS=+0.0038
ΔP=+0.0014 (+46 TP / +24 FP). Clean cross-seed positive on both
precision and recall.

**Rule set: 32 rules** (30 pre-pivot + N36 + N37).

**N38 — `including_participle` ACCEPTED** (refinement-phase bar:
ΔS≥0.0005, ΔP≥−0.0075). Surfaces parenthetical inclusion modifiers:
``X, including/involving/featuring Y`` where spaCy parses the
participle as dep=prep on a noun head. Emits (head, including, pobj)
plus conj-coordinated args. EnokiQA seed=9 ΔS=+0.0006 ΔP=−0.0008
(+19 TP / +83 FP); seed=11 ΔS=+0.0013 ΔP=−0.0004 (+28 TP / +84 FP).
ΔP regression cosmetic — FP audit shows the new emissions are
semantically valid inclusions that EnokiQA's multi-granularity gold
simply doesn't enumerate (gold-incompleteness).

**Rule set: 33 rules.**

**Cumulative 3-span downstream impact** (modernbert, t=0.5, no-coref,
flat-Fact adapter as origin):

| Dataset   | flat   | inc-adapter only | inc + N36/37/38 |  Net Δ |
|-----------|-------:|-----------------:|----------------:|-------:|
| Mushroom  | 0.4465 |           0.4827 |          0.4924 | +0.0459|
| PsiloQA   | 0.6248 |           0.6358 |          0.6436 | +0.0188|
| RAGTruth  | 0.2655 |           0.2535 |          0.2576 | −0.0079|

Rule additions strengthened the incremental-adapter gains on all three
SPAN datasets and partially recovered the RAGTruth regression.

**Complex-object audit (2026-05-24).** Built
``work/enokiqa_complex_obj_audit.py`` — for each (subj, pred) gold
bucket whose widest object has ≥5 tokens, checks whether ANY of our
pipeline emissions reaches 0.5-overlap on the widest gold object.
**60% gap (332/551 buckets on 400-sent seed=131).** Failure modes:
appositive maximal expansion stops at the head NP; copula-be with
stacked sibling-prep PPs; passive "established by NER" not
expanded; xcomp "able to X" not surfaced as composite predicate;
"X enabled Y to verb" infinitival complement object.

**N39 — `appos_identity_maximal` ACCEPTED** (cross-seed 9+11).
Companion to appos_identity. Same guards (head + appos both NOUN/
PROPN, comma-preceded, full-NP appositive, anchored to clause
root); emits the full appositive subtree (head + amods + compounds
+ prep PPs + acl) as arg via ``arg_span_subtree=True``. Fills the
common appositive-with-PP gap surfaced by the complex-object audit.
EnokiQA seed=9 ΔS=+0.0009 ΔP=−0.0001 (+32 TP / +74 FP); seed=11
ΔS=+0.0008 ΔP=−0.0000 (+27 TP / +61 FP). Both seeds clean accept
under refinement-phase bar.

**N40 — `copula_be_attr_with_pps` REJECTED** (seed=9). Aimed to
emit ``attr.subtree ∪ sibling_prep_subtrees`` as one wide arg
("Mischief is a criminal offense in Canada under section 430 of
the Criminal Code"). Required infra change (``arg_span_extra_anchors``
field on Candidate). Seed=9: ΔS=−0.0000, ΔP=−0.0009 (+11 TP /
+72 FP). The unioned-span emission consumes a single gold slot
per (s,p) bucket but the gold's widest variant is often narrower
than the everything-stacked form, so most emissions are FP. File
and infra reverted.

**Rule set: 34 rules** (33 pre-pivot + N39).

**N41 — `ecm_dobj_infinitive` REJECTED** (seed=9). Aimed at the
"X allowed Y to verb Z" ECM/raising pattern where gold credits
(X, allowed, Y to verb Z). Closed verb set (allow/enable/let/make/
cause/force/help/permit/encourage/require/lead/ask/tell/want/expect/
consider/see) + dobj + xcomp(VERB). Required the same
``arg_span_extra_anchors`` infra as N40.
Seed=9: **ΔS=−0.0000 ΔP=−0.0002** (+3 TP / +18 FP). Almost no
firings because spaCy parses most "allow X to Y" as
verb+ccomp(includes-nsubj), which ``ccomp_arg`` already covers; the
dobj+xcomp parse path is rare. File + infra reverted.

**Two consecutive infra rejections (N40, N41).** Both targeted
genuinely-frequent EnokiQA gold patterns but failed on the matcher:
N40's unioned-PP wide span over-consumes when gold's max variant is
narrower; N41's pattern is mostly captured by ccomp_arg. Combined
with N32/N33/N34/N35 (4 earlier consecutive rejections), this
confirms the rule channel is saturated for EnokiQA's 0.5-overlap
matcher; further channel-only gains require either a matcher change
(e.g. lemma-overlap) or downstream-evaluation-driven gating.

**Final cumulative 3-span impact** (34 rules + inc adapter):

| Dataset   | flat-Fact | inc + 33 rules | inc + 34 (add N39) | Net Δ |
|-----------|----------:|---------------:|-------------------:|------:|
| Mushroom  |    0.4465 |         0.4924 |             0.4921 |+0.0456|
| PsiloQA   |    0.6248 |         0.6436 |             0.6441 |+0.0193|
| RAGTruth  |    0.2655 |         0.2576 |             0.2574 |-0.0081|

N39 is neutral downstream — appositives are peripheral content for
the hallucination-NLI task. EnokiQA channel gain doesn't transfer.

**Rule set held at 34.** Session closing on saturation signal.

**N42 — `light_verb_construction` ACCEPTED** (cross-seed 9+11).
Closed (verb_lemma, dobj_lemma, prep_lemma) set of LVC idioms
(play/role/in, set/stage/for, set/precedent/for, lay/groundwork/for,
lay/foundation/for, pave/way/for, bridge/gap/between, have/impact/on,
make/contribution/to, place/emphasis/on, take/place/in, etc.).
Each tuple ≥10 occurrences in 10k EnokiQA val sents. Composite
synthesized predicate ("paved the way for", "played a crucial role
in") built from actual token sequence verb..prep including det/amod
modifiers of dobj. Handles BOTH parse paths: prep as dobj child OR
prep as verb sibling (the more common LVC parse). EnokiQA seed=9
ΔS=+0.0011 ΔP=−0.0001 (+17 TP / +43 FP); seed=11 ΔS=+0.0008
ΔP=−0.0003 (+16 TP / +51 FP). Both seeds positive both metrics.

**N43 attempt skipped** — "X, also known as Y" / "X regarded as Y"
participial-passive-as already covered by existing ``acl_passive_as``
(verified on test sentences). Saved a rejected-rule round.

**Rule set: 35 rules.**

**Final cumulative impact (35 rules + inc adapter, post-N42):**

| Dataset   | flat-Fact | inc + 33 rules | + N39 | + N42 (35 rules) | Net Δ |
|-----------|----------:|---------------:|------:|-----------------:|------:|
| Mushroom  |    0.4465 |         0.4924 |0.4921 |           0.4918 |+0.0453|
| PsiloQA   |    0.6248 |         0.6436 |0.6441 |           0.6439 |+0.0191|
| RAGTruth  |    0.2655 |         0.2576 |0.2574 |           0.2572 |-0.0083|

N42 (like N39) is downstream-neutral — EnokiQA channel gain doesn't
transfer to ModernBERT-NLI hallucination scoring. Big downstream
wins came from (a) the IncrementalFactGroup adapter rewrap and
(b) N36/N37/N38 (composite-predicate, V-as, inclusion-modifier)
which introduce semantically NEW propositions, not just wider
granularities of existing ones.

**Rule set held at 35.** Closing the 35-rule line.
