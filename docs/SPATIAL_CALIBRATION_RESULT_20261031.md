# Spatial Calibration: Completed Fresh Four-Video Control

Verified on 2026-10-31 at18:01 UTC+08. All32 registered full trajectories
completed; the independent analyzer verified their metrics, source/GT/
checkpoint/trajectory hashes, model-call counts, blind holds, reconstruction
of the past-only offsets/scales, and native-sham/complete equality.

Source directory:
`reports/server28-spatial-calibration4-20261031-recovered`.
Published report: `reports/spatial-calibration4-recovered-20261031.json`.
SHA256:
`96be9ee99255d28ee98ccafb01f844c82cc49666ec074d0676fd8f2a1046242e`.
Server report, completion receipt and downloaded report agree.

## Results

The prospectively registered development videos are rightbhindbike-gai,
leftwhitebike, whitegirlwithumbrella and manbikecoming:836frames total.
Every video and trajectory was retained. Scores are video-macro percentages.

| Condition | Arm | PR20 | SR |
|---|---|---:|---:|
| Complete | All four arms, exactly equal | 94.507576 | 77.629789 |
| Synthetic block1 outage | Native/sham, exactly equal | 93.983586 | 75.107497 |
| Synthetic block1 outage | Past center | 93.226010 | 74.745948 |
| Synthetic block1 outage | Past center and size | 93.068182 | 75.233551 |

| Outage contrast | PR delta, pp | PR paired-video95% interval | SR delta, pp | SR paired-video95% interval |
|---|---:|---|---:|---|
| Center minus native | -0.757576 | [-1.515152,0.000000] | -0.361548 | [-1.194431,0.241911] |
| Geometry minus native | -0.915404 | [-1.893939,0.000000] | +0.126055 | [-0.117244,0.394120] |
| Geometry minus center | -0.157828 | [-0.757576,0.284091] | +0.487603 | [0.082036,1.199495] |

Geometry improves SR over center on4/4videos, but improves over native on
only2/4. Its PR has0wins/2ties/2losses against native. This supports a small
conditional size-calibration effect relative to the center-only control,
not stable improvement over the actual native baseline. Four correlated
development videos and one E0 weight do not establish official gains,
new-seed robustness, a novel module, or the requested three contributions.
Do not promote the geometry arm on the basis of this result.

## Next Registered Experiment

The existing recovery controller72348 launched gap workers168995/168996
at approximately17:59:50 UTC+08, both actually alive18:01:45 with the
registered command lines. Output:
`reports/server28-calibration-gap4-20261031-recovered`.
The launch links this completed geometry report by its full SHA256.

The unchanged original registration uses fresh development indices60..63:
boyaftertworunboys458frames, twopeople_0109 450frames,
10rightblackboy333frames, leftbottle1143frames.
Four arms times four conditions times four videos gives64trajectories.
Conditions are complete and synthetic block1/block32/block128 outages.
Changing block duration changes observation/template histories as well as
calibration age; this is not isolated causal cache-age identification.
Keep short videos even when some block128 states are unexposed.

Original protocol estimate was60-120minutes after predecessor release,
plus up to300seconds controller delay: approximately19:00-20:00 UTC+08.
Revise only from actual throughput. Keep the registered sources unchanged
until final independent analysis. All four recovery trainers and the
CPU repair audit remained alive18:01:10. CPU full paired result was pending.

Official thresholds, convergence, final-five checkpoint selection, global
original-checkpoint cleanup and three validated modules remain incomplete.
