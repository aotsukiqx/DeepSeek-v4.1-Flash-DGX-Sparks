# bp6r3 bisect variants (decode >=42 hunt)

Base: bp6r2-candidate branch image (SM121 DeepGEMM gates + fmla 1.0.0; decode 38.8).
Hypothesis (bounded decode trace 11-08): per-step host cudaEventSynchronize
~44ms from the compact-verify refactor. These are clean `git revert --no-commit`
patches computed against 9d547e19 (apply AFTER the r2 gate patch; verify with
`git apply --check` in the image build tree, exact-commit style like bp4).

Variant matrix (one window, ordered by suspicion):
- V1 = r2 - #48833 (compact verify window; 6 files) — prime suspect
- V2 = r2 - #48831 (draft metadata glue; 1 file, 5 lines)
- V3 = r2 - #48832 (compressors fused; 1 file)
- If none pass decode>=42: the sync pattern is elsewhere in the pin's
  spec rework — escalate to #34031/#34038-era or profile-guided pinpoint.

Decision pending team review (11-08); artifacts ready regardless.
