# Server 28 Live Snapshot

Observed 2026-10-30 19:10 UTC.

## Active work

- Fixed20 utility/random training processes 3453478/3453479 remain live on
  GPUs0/1, epoch5, latest log updates10625/10975. Recorded loss fields are
  finite. Their evaluation controller remains queued behind training.
- Batch10 utility/random candidate controllers 3816049/3816050 and training
  processes 3816056/3816057 remain live on GPUs2/3, epoch3, latest updates
  4600/4750. Recorded losses are finite; only epoch1/2 fixed128 task-loss
  validations exist. Epoch3 validation is pending. Checkpoints epoch01/02
  exist; epoch03 has not completed.
- Historical A utility20 LasHeR-Miss evaluation PID4138772 remains live with
  one worker on each of GPUs0/1. It has8/479 sequence receipts totaling
 62941 frames; all eight completed receipts are LasHeR-Miss. No official
  metrics JSON exists yet.

For the first8 receipts, total worker time is5039.6 seconds over62941 frames
(12.49 frames/worker-second). Linear projection over337353 total frames and
two workers gives roughly3.1 additional hours from this observation, around
22:14 UTC. This is only a rough ETA: the evaluator sorts longest sequences
first, and only the first8 have completed.

## GPU state

Temperatures were84/80/79/73 C and power limits250 W on GPUs0-3. Current
draw was approximately244-249 W. The recorded target temperature remains
85 C.

No process was restarted or modified. This snapshot is health/progress
evidence only, not a benchmark result or validated module.
