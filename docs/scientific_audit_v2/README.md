# Scientific audit v2 (Claude C) — integrated docs

These four documents are the deliverables of the independent scientific audit (branch
`experiment/scientific-audit-v2`, tip `63fba7b`). That branch was rooted in the full IsaacLab monorepo
(unrelated history), where these files lived at
`projects/paper_audit_v2/docs/scientific_audit_v2/`. To keep this standalone `paper` integration branch
coherent, the audit branch is recorded as a merged parent (`git merge -s ours`) and only these audit
deliverables were grafted here (no monorepo files imported, nothing deleted or overwritten).

- `final_recommendation_v1.md` — verdict MODIFY, Phase-2 guardrails.
- `claims_vs_evidence_v1.md` — per-claim grading (C1–C11).
- `runtime_data_audit_v1.md` — first-hand runtime/data integrity + replicate/held-out-state findings.
- `offline_method_audit_v1.md` — audit of Claude B's offline method.
