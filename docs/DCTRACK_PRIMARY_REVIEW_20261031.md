# DCTrack Primary-Paper Correction

Reviewed the actual local PDF bytes with pdftotext on 2026-10-31.
Title: DCTrack: Divide-and-Conquer for Modality-Missing RGBT Tracking.
First author: Shilei Wang. CVPR Workshops 2026, pages 8338 onward.

Source: https://openaccess.thecvf.com/content/CVPR2026W/URVIS/papers/Wang_DCTrack_Divide-and-Conquer_for_Modality-Missing_RGBT_Tracking_CVPRW_2026_paper.pdf

Local evidence: `work/dctrack-cvf-2026.pdf`, outside the publication clone.
SHA256: `ac899ca27264de01218dd7c86ee4b89dddbf218844dcf14d0697d672890cd715`.

## Verified Mechanism

- Section 3.3: availability uses maximum intensity >=5, not image standard deviation.
- Sections 3.4-3.5: IPT, MMCITrack and DAM4SAM variants; substitute the
  available image for a missing stream. DAM4SAM prefers visible input.
- Section 3.5, equations 4-6: linear extrapolation from two past valid states
  blends with Kalman prediction; unstable trajectories use buffered replay.
  This is not a demonstrated future-boundary interpolation method.
- The inspected paper does not specify enough replay/instability details to
  reproduce that branch exactly. No verified DSRTrack code provenance follows.

## Consequence For Our Experiments

Availability routing, image substitution and ordinary motion prediction
cannot independently support our novelty claims. The paper does not verify
our past complete-frame counterfactual center/size calibration or its efficacy.
Current registered experiments stay unchanged; only complete prospective
results can support promotion.

The earlier feature-disentanglement description, and the intermediate
SIAMTFA/DiMP, STD-detection and future-interpolation account, are unsupported
by these PDF bytes and must not enter the paper or implementation plan.
