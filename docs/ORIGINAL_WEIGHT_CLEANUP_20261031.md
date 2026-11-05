# Original Campaign Weight Cleanup

The four recovery trainers hardlinked their retained original weights into
new output directories. Rolling retention and final-five selection remove
recovery paths, while the original campaign paths remain. Removing only the
recovery links does not satisfy the requested cleanup of unused weights.

`cleanup_original_weights28.py` now registers only the original epoch
weights in these four directories under `runs/server28`:

- `a_utility_full20_seed2027`
- `a_random_full20_seed2027`
- `batch10_utility_seed2027`
- `batch10_random_seed2027`

It preserves the common validation-selected weight in each corresponding
`runs/server28-recovery-20261031` directory. The selection is shared across
complete/Miss datasets within each campaign; it is not chosen on test scores.
Other experiments and historical/pretrained weights are outside this scope.

## Verified Registration

At2026-10-31 18:17:15 UTC+08, registration recorded12existing original
files with exact paths, bytes and SHA256. The worker PID197828 was actually
alive18:17:57 and logged `WAITING` for all four final independent verifiers.
Poll interval300seconds; CPU-only, no GPU allocation.

Source SHA256:
`2685ec98c14d09d565709e1a8aa476c8b6f202fdf68946eb843e7c96c5acf7ab`.
Registration SHA256:
`3cc1a95c53ed920f018ed06cd677f33ee3870ab85b644d1b2e5f2e7f65e1ea0b`.
Output: `reports/server28-original-weight-cleanup-20261031`.
Witness: `reports/server28-original-weight-cleanup-witness-20261031.json`.

## Execution Conditions

The queue waits on the four actual registered verifier processes while
their `independent_verification.json` files are absent. If a verifier exits
without its result, the cleanup exits before deleting any original weight.
No automatic restart or altered verification source is accepted.

Only after all four independent verifications exist does it check:

- Registered verifier/queue source hashes and selected-weight identity.
- Each campaign's970development and958official verified trajectories,
  same selected checkpoint and matching official complete/Miss metrics.
- Completed CPU repair audit and corrected analysis hashes, because those
  audits used original epoch6 weights.
- Exact original file inventory, path boundaries, file sizes and SHA256
  against registration before any deletion.
- All four retained recovery paths and hashes before and after cleanup.

It writes `approved_plan.json` before cleanup, then an fsynced per-file
`deleted.jsonl` receipt, followed by `completed.json` only after the four
original directories contain no epoch weights and the selected hashes match.
It records removed path bytes separately from last-link-removed bytes.
Old independent verifier reports remain unchanged and still state their
own original-cleanup limitation; this separate completion receipt is the
evidence for original cleanup when it eventually exists.

The worker does not require threshold attainment as a deletion trigger:
selection/complete evaluation must finish regardless of whether a model
meets the target. It does not certify convergence or paper modules.
No real campaign weights have been deleted by this new queue yet.

## Actual Witness

Before launch, the shared inventory/removal body actually removed4temporary
fixture weights, preserved all4selected fixture weights and an unrelated
fixture weight, and wrote4journal entries. The fixture directory was scoped
under `reports` and removed by `TemporaryDirectory`.
This tests real removal/retention behavior, not future final metric checks
or the end-to-end campaign cleanup. The latter remains unverified.

Preserve this source, the original12registered weights, the registered
verification sources and final selected weights until their respective
stages complete. Do not repurpose these original paths for new experiments
while this cleanup registration is pending.

## Current Campaign Timing

At18:20:59 UTC+08 the candidate Utility was in epoch11/update20181 and
Random in epoch12/update22222. Utility10 and Random11 validation losses
were5.026235371828079 and5.032911457121372 respectively, both worse than
their previous epochs. These are cached128-tuple proxies, not convergence
or official tracking metrics. Earliest plateau stop remains14 because the
post-drop reference resets at11.

The registered gap experiment had35/64completed full trajectories:
GPU0 9/16 with17396/33776registered model calls in finished trajectories;
GPU1 26/48 with17780/37544. All videos remain registered. Based on the
approximately21minutes actually elapsed, revise the final-result window
to18:40-18:55 UTC+08 including parent polling, rather than the conservative
original19:00-20:00 estimate. In-progress trajectory calls are not included
in these completed-call counts; this is a provisional ETA, not a result.
The next candidate validation boundaries are roughly19:30-19:50 UTC+08.
