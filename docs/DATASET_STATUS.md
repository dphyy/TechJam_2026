# Dataset roles and recorded exposure

A dataset's role, target overlap, and evaluation history are separate facts.
`development`, `screening`, `confirmation`, `validation`, and `final` describe its
role in an experiment. A public-disjoint target can still have been evaluated in
another experiment. Indexing the catalog is expected retrieval behavior, not
exposure to evaluation outcomes.

Do not infer an untouched holdout from `sealed`, `reserved`, or `unseen` in a
filename. Preparation-time fields such as `reserved_status`, `consumption`, or
`validation_outcomes_accessed: false` are snapshots. In a lock-verification
receipt, `validation_outcomes_accessed: false` means that verification call did
not evaluate the split; it does not establish that no earlier run did so.

## Audit current recorded exposure

The read-only command matches exact dataset SHA-256 hashes against local run
reports, run manifests, registrations, consumption markers, and ledger events:

```bash
python -m experiments.dataset_status --dataset artifacts/unseen-v1/final-sealed.jsonl
python -m experiments.dataset_status --dataset artifacts/review-prior-weight-tuning-v3/reserved.jsonl
```

It scans `runs/`, `output/`, and `artifacts/` by default, excluding preserved
source trees and model files. Repeat `--evidence-root PATH` to supply a different
set of receipt directories, including copies of teammate receipts. It hashes
dataset bytes but does not parse or print session labels, run an agent, or write
any file.

| Status | Meaning |
|---|---|
| `consumed` | A matching evaluation report, completed run manifest, consumption marker, or ledger event records use. |
| `attempt_recorded` | A matching registration or incomplete run manifest exists; completion is not established. Do not treat the attempt as an untouched holdout. |
| `unknown` | No supported matching receipt was found, or dataset bytes changed during the audit. This does **not** mean unseen. |

The report lists evidence paths and scan warnings. It never certifies a set as
untouched: manual inspection, absent or remote receipts, and target overlap with
different/reformatted dataset bytes require separate review. A consumed event
takes precedence over a stale `sealed` label, regardless of the current agent,
branch, dataset filename, or output directory. The audit reports exposure; it
does not replace or relax existing one-shot evaluation guards.

## Corrections verified on 31 August 2026

- `artifacts/unseen-v1/final-sealed.jsonl` was consumed by the refinement runs on
  29 August. Evidence: `runs/refined-paging-final-sealed/report.json` and
  `runs/refined-paging-final-control/report.json`. The earlier roadmap report
  described its status before those runs, not after them.
- `artifacts/review-prior-weight-tuning-v3/reserved.jsonl` has a hash-bound opening
  marker at `runs/review-prior-weight-tuning-v3/reserved-consumed.json`, dated
  31 August. Its preparation manifest's `reserved_status: sealed` is historical.

These are local artifacts and may not exist in another checkout. Missing local
evidence cannot restore holdout status. These paths and observations are a dated
correction, not a promise that other datasets have remained unused.

## Naming and evidence preservation

Keep existing dataset names, bytes, manifests, freeze records, and checksums
unchanged. Renaming historical files would break commands and provenance without
making the data fresh. Use role-only names such as `final.jsonl` for new pack
formats; keep current exposure in separate hash-bound receipts, not in filenames.
The existing `unseen_prepare` format keeps its legacy filename to reproduce its
registered outputs, and now explicitly labels its metadata as a preparation
snapshot.

Record an opening before a restricted evaluation starts, retaining that record
on failure. A registered frozen pair may run its planned comparison, but seeing
the outcomes consumes the set for subsequent model selection. Generate a new,
independently prepared holdout for subsequent tuning claims; neither a new
branch nor a new output folder makes an old set untouched.
