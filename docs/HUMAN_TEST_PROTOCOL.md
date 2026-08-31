# Proposed shopper study — not yet run

This is future work for the current lexical pipeline, not a completed experiment
or a claimed result. No participants have been contacted by this documentation
update. Obtain informed agreement before recruiting or recording anyone.

## Question and controls

Does guarded paging help shoppers find an acceptable product without losing
relevance or mishandling changed requirements?

Freeze the public `agent.Agent` with `DEFAULT_AGENT_CONFIG` and a matched control
with `dataclasses.replace(DEFAULT_AGENT_CONFIG, guarded_paging=False)`. Keep the
same source, catalog, parser, ranking, question policy and output-width policy.
No neural model is needed. Record configuration/source hashes before any session.

## Proposed method

Recruit five consenting testers. Each writes four tasks before seeing outputs:
an ordinary request, acceptable alternatives, a correction, and a request with
missing catalog information. Record necessary requirements and success criteria
in their own words. Use random tester IDs; do not collect sensitive details,
account credentials or purchase histories.

Alternate blinded A/B presentation order and use isolated sessions. Show actual
product IDs, titles, evidence and questions. Let people accept, reject or revise
freely; do not curate away failures or require them to purchase anything. Keep
sessions to ten turns and record exact messages only with consent.

Record acceptance, turn of first acceptance, repeated products, lost requirements,
contradictions, unsupported claims, errors, and a short satisfaction rating/reason.
Describe differences in follow-up histories as part of the end-to-end comparison.

## Interpretation

Report all twenty paired tasks, failures and order effects. Five people doing
four tasks each are not twenty independent participants. This would be formative
feedback, not a population estimate or a conversion claim. Publish only anonymized
aggregates and separately consented quotations; agree on retention and keep raw
transcripts out of source control. Subsequent code changes need new validation.
