# Demo of the current public pipeline

Run `python -m demo.submission --output output/current-demo` from the repo root.
`python -m demo.showcase --output output/current-showcase` is an equivalent CLI.
Use a new destination. Open its `index.html`; keep the paired `evidence.json` and
`transcript.txt`. Nothing is uploaded by these commands.

The default five authored turns request a black leather bag with an adjustable
strap, correct color/material to blue canvas, clear color and exclude leather,
then reject two displayed slates using different phrasings. Show the actual IDs,
active/retired evidence, paging reset/advance receipts, and measured latency.

A suggested three-minute narration:

1. Explain exact-product conversational search and why changed intent and repeated
   results matter. State that this is an authored demonstration, not a scored test.
2. Show the correction and no-preference turns. Confirm category stays bags and
   unrelated constraints survive. Missing catalog evidence is explicitly unknown.
3. Show paging after rejection without inventing new preferences. Explain that
   corrections reset exposure and exhaustion can repeat compatible records.
4. Explain offline lexical retrieval and the adaptive shortlist. Show separate
   verified aggregate evidence if available, then limitations and release status.

To attach metrics, supply both `--evaluation-report PATH/report.json` and
`--evaluation-sha256 SHA256`. Only a valid report for the current source, default
configuration and catalog is accepted. `docs/current-results.json` is a status/
aggregate pointer, not an input to bypass that check. Do not paste historical
neural or pre-fix paging scores onto a current demo.

The old renderer remains at `python -m demo.legacy_showcase`; `demo.replay`,
`demo.release`, `demo.alternatives`, and `demo.role_evidence` are research demos.
They are not demonstrations of the public submission default. Old artifact
folders and terminal recordings retain their original source-specific meaning.
Recording/export, rights review, public video upload, and Devpost submission are
separate actions and have not been completed by this command.
