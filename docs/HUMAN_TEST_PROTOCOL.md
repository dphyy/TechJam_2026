# Small shopper test: proposed, not yet run

The current evidence comes from the organizer's simulator and authored synthetic cases. Neither establishes real shopper usefulness. This is a small follow-up study the team can run after obtaining participants' agreement; no participants have been recruited or contacted as part of this work.

## Question

Does preserving explicit alternatives help a shopper recover from a changed requirement without rejecting an acceptable product or losing an unrelated requirement?

## Before testing

1. Recruit five consenting testers. Do not collect names, purchase histories, account credentials or sensitive personal details; use random tester IDs.
2. Each tester writes four shopping tasks before seeing any system output: an ordinary request, an explicit choice between acceptable materials/features, a later correction, and a request involving missing catalog information. Let testers use their own wording. Do not give them the developer demonstration phrases.
3. For each task, record what the tester considers necessary, acceptable alternatives and what would count as success. An unknown catalog fact must not be labeled confirmed. Freeze these criteria before running either system.
4. Freeze the release candidate and parser-only control. Use the same catalog, model and budgets. Label systems A/B without revealing the mechanism; alternate which is shown first. Preserve all tasks, including ones neither system solves.

## Run and record

Run both systems on each authored task using isolated sessions. Present actual product titles, IDs, catalog evidence and follow-up questions; do not replace outputs with curated results. Keep the test to ten turns. Allow the tester to correct the intent in their own words, and record the exact visible messages with consent.

Record task ID, blinded system, shown-first order, accepted-result yes/no, turn of first acceptance, lost unrelated requirement yes/no, contradicted accepted alternative yes/no, unsupported factual claim yes/no, and the tester's short reason. Have a second team member check any catalog-evidence disagreement without seeing which system produced it.

Do not ask testers to purchase anything or equate stated acceptance with a sale. Do not silently exclude a difficult task, an unsupported construction, a fallback or an error.

## Report honestly

Report all five testers and twenty paired tasks, including failures and order effects. Task-level examples are useful, but repeated tasks from one person are not twenty independent people. Treat these results as formative evidence, not a population-level conversion claim. If conversational follow-ups differ, describe the comparison as end-to-end task performance rather than a fixed-query ranking experiment.

Publish only anonymized aggregates and separately consented quotations. Keep raw transcripts local, agree on a retention period with participants, and do not include them in a source release by default. Any later code change must be distinguished from the versions tested.
