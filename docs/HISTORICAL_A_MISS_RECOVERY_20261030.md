# Historical A Miss evaluation and concurrency correction

Goal: fill missing official Miss metrics for the already-trained historical
A utility epoch20 checkpoint. This evaluation does not select candidate
weights and cannot alter the four active training jobs.

## Failed launch

Evaluator PID4080351 attempted the complete479-sequence/337353-frame Miss
evaluation on physical GPU1 with four multiprocessing workers while the
GPU1 trainer remained active. It failed before producing a valid complete
evaluation. The log records two worker model-load CUDA OOMs: a20MiB
allocation with11.75MiB free and another with5.75MiB free. Other workers
then failed during convolution with CUDNN_STATUS_NOT_INITIALIZED, secondary
to memory exhaustion. No metrics or complete receipt exists. Preserve the
failed output/log and never interpret it as a model result.

## Measured concurrency

A separate single-worker GPU1 profile completed one565-frame sequence in
52.04seconds with peak PyTorch allocation499,928,064bytes and reservation
624,951,296bytes.

Evaluator now accepts `--workers-per-gpu` and `--limit`. A limited
two-sequence execution smoke completed with one worker on each of GPUs0,1.
The official inventory, sorted longest-first, selected ``blkboy`shead``12862
frames on GPU1 and `boyshead9684`9684frames on GPU0. Total22546frames;
both receipts contain finite boxes and identify the assigned physical GPU.

First controller PID4100659 stopped without launching full evaluation
because the evaluator writes per-sequence receipts rather than the
expected all_receipts.json. Corrected controller PID4138770 verified the
two receipt files and GPU assignments, then launched full historical Miss
evaluation PID4138772. Output:
`results/server28_historical_a_20261030/miss_single_worker_per_gpu`.
It uses one worker per GPU (two total), with no automatic retry. At
18:29UTC it was live with0 full-run sequence receipts; no metric result yet.

## Correctness scope

The full run uses the same evaluator's strict inventory/mask/GT checks,
native tracker, all479sequence receipts and official toolkit metrics.
Limit mode exits after inference receipts and does not publish benchmark
metrics. GPU0/1 training and frozen runtime are untouched. Before reporting,
verify479sequences,337353frames and all requested LasHeR-Miss and
RGBT234-Miss scores. Historical A complete-input metrics already exist.
