# Pairing verifier correction before any epoch result

The training protocol stops Utility and Random independently using their own
fixed train882 development losses. The original verifier incorrectly required
equal entire input/step arrays after both arms stop. Different valid stopping
epochs would therefore block all official evaluation despite valid training.
This is a contract error, not a hypothetical fallback: existing paired dense
geometry training actually stopped native at20 and dense at14 epochs
(DENSE_GEOMETRY_TRAINING_20261102.md). Neither current formal arm has yet
completed its first epoch; no formal score or stopping outcome informed this
correction.

Version2 independently checks every arm's entire data/source/loss/weight/
stopping/selection records as before. It requires equal fixed development
inputs, then exact input identities, tensor hashes and mask hashes over the
whole COMMON fitting prefix. Any additional epochs in the longer arm still
undergo all single-arm checks. It explicitly reports both lengths and common
length and does not claim the longer suffix is paired. A difference anywhere
in the common prefix still fails. Both arms must retain five checkpoints.
No training, seed, optimizer, data partition, stopping criterion, selected
checkpoint, official evaluator, official score or cleanup rule changes.

Preserve original verifier/controller files and all their hash receipts.
New verify_partitioned_a_v2.py writes verification_v2.json; run the existing
six-epoch smoke evidence through it. Test equal/unequal lengths and common
input/mask mismatch rejection directly on the same comparison function.
No benchmark is used for these checks.

Retire ONLY old sleeping controllerPID1318842 after verifying actual command,
WAITING_FOR_TRAINING, live original trainers and absence of any evaluation.
Do not stop/restart either trainer or remove any queue/weight/result files.
Register partitioned_a_official_queue_v2.py in a separate queue directory.
Its only functional changes are verifier path/receipt and separate controller
provenance; official result destinations, GPU2/3, final-five reports, full
479/337353 scope, four workers per GPU and source-check requirements persist.
New sources and this amendment are frozen by the new queue. Old queue remains
as superseded evidence. No automatic fallback to an older verifier.
