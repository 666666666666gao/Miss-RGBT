# Global proposal source review

Reviewed 2026-10-30; exactly two author repositories, excluding KeepTrack/GlobalTrack. **Decision: no directly suitable, license-cleared, target-conditioned global proposal implementation for the existing frozen ViT was found.** Siam R-CNN is a usable architectural reference; SPLT is a closer functional match but lacks an explicit repository-wide license. Neither was executed. No source was copied, weights/dependencies downloaded, or GPU server/training files changed.

## 1. Siam R-CNN: retain as architectural reference

Official author repository `RWTHVision/SiamR-CNN`, commit `df9b428aeb90da0c8b2c8076f54f632efb07366c`; root [LICENSE][s-license] is MIT, copyright Visual Computing Institute.

**Paper versus source.** The [CVPR paper][s-paper] describes first-frame/previous-prediction re-detection and tracklet dynamic programming. Actual [train.py][s-train] generates full-image RPN proposals before reference conditioning (`build_graph`, lines 451-495); `roi_heads` concatenates reference and proposal RoI features. Thus the final detections are target-conditioned, but initial candidate generation is target agnostic. This distinction matters when objectness fails to propose the lost target.

**Resolution and cost.** [config.py][s-config] specifies 800-pixel test short edge, maximum 1333 before FPN padding, ResNet-101/FPN, and FPN proposal settings. This preserves substantially more full-image detail than compressing a frame into a small tracking crop, without establishing small-target recall. [README][s-readme] requires TensorFlow-GPU 1.15/TensorPack, supplies checkpoint instructions, and documents supervised training from a COCO detector; hard-example feature precomputation code is explicitly unavailable. No CPU/GPU latency was measured here.

**Original-target acceptance.** [tracking/three_stage_tracker.py][s-tracker] keeps first-frame reference scores, separate evolving target-tracklet scores, distractor tracklets, association thresholds and dynamic programming. `_track` can return a historical box with negative confidence when its selected tracklet is inactive. This is learned association and target evidence, not proof that a proposal is the original object or a standalone calibrated acceptance gate.

**Frozen ViT assessment (inference):** the 256x7x7 RoI interface and multiscale FPN are concrete boundaries worth studying, but replacing ResNet/FPN requires feature adaptation and supervised detector/re-identification head training. There is no supplied frozen-ViT adapter.

## 2. SPLT: reject code reuse pending license clarification

Official author repository `iiau-tracker/SPLT`, commit `a196e603798e9be969d9d985c087c11cad1cda43`. The pinned [tree][p-tree] contains no LICENSE/COPYING/NOTICE file; readable source and vendored dependencies do not establish permission for the authors' tracker code.

**Paper versus source.** The [ICCV paper][p-paper] claims image-wide sliding-window skimming followed by regression/verification. Actual [utils/tracking_utils.py][p-utils], `gen_search_patch_Hao` (47-97), rescales the whole image according to initial target size, enumerates overlapping 256x256 windows with stride 128, and truncates/subsamples to 500. This is target-scale global enumeration, not unrestricted coverage at every scale.

[SPLT_tracker_new.py][p-tracker] (567-725) ranks windows using fixed initial-template skimming features, processes normally three best regions with a template-conditioned 300x300 regressor, then checks the best detection against initial `template_feat`. Acceptance requires regressor score >0.7 and verifier squared distance <0.8. Proposal ranking therefore differs explicitly from original-target acceptance. [Verifier/resnet50_bin.py][p-verifier] L2-normalizes its learned embedding: squared distance equals `2-2*cosine`; this does not validate a new verifier using untrained frozen-ViT cosine.

**Requirements/limits.** [README][p-readme] documents pretrained regressor/verifier checkpoints and RTX 2080Ti testing; [requirements.txt][p-requirements] pins TensorFlow-GPU 1.13.1. Skimming models and training scripts are also present. Compute grows with window count plus regressor/verifier calls; no measured Miss-RGBT throughput exists. Both repositories implement three-channel visual tracking, without supplied RGB-T missing-modality fusion. A frozen-ViT port would require learned heads and independent identity/recall validation; neither paper establishes its success.

## Relevance And Review

This bounded research was delegated under the installed research skill and
reviewed by the main agent. Main-agent checks independently read Siam R-CNN's
pinned MIT license and proposal construction. The SPLT recursive Git tree
was non-truncated, with3618 entries and no LICENSE/COPYING/NOTICE-named file.
That tree check alone does not audit every embedded source notice.
No code is being copied, and no new package or checkpoint is being installed.

The next deployable recovery design must measure image support, proposal
recall, and correct-target acceptance separately. Historical fixed-grid and
cosine controls already failed to establish recovery; these repositories do
not erase that evidence. A target-agnostic objectness route can miss the
tracked target before its identity head runs, while target-ranked sliding
windows can drop useful crops. Any learned head needs train882 supervision,
matched wrong-anchor/distractor controls and a frozen development decision
before official testing. No further recovery job is launched from this review.

[s-license]: https://github.com/RWTHVision/SiamR-CNN/blob/df9b428aeb90da0c8b2c8076f54f632efb07366c/LICENSE
[s-paper]: https://openaccess.thecvf.com/content_CVPR_2020/html/Voigtlaender_Siam_R-CNN_Visual_Tracking_by_Re-Detection_CVPR_2020_paper.html
[s-train]: https://github.com/RWTHVision/SiamR-CNN/blob/df9b428aeb90da0c8b2c8076f54f632efb07366c/train.py
[s-config]: https://github.com/RWTHVision/SiamR-CNN/blob/df9b428aeb90da0c8b2c8076f54f632efb07366c/config.py
[s-readme]: https://github.com/RWTHVision/SiamR-CNN/blob/df9b428aeb90da0c8b2c8076f54f632efb07366c/README.md
[s-tracker]: https://github.com/RWTHVision/SiamR-CNN/blob/df9b428aeb90da0c8b2c8076f54f632efb07366c/tracking/three_stage_tracker.py
[p-tree]: https://github.com/iiau-tracker/SPLT/tree/a196e603798e9be969d9d985c087c11cad1cda43
[p-paper]: https://openaccess.thecvf.com/content_ICCV_2019/html/Yan_Skimming-Perusal_Tracking_A_Framework_for_Real-Time_and_Robust_Long-Term_Tracking_ICCV_2019_paper.html
[p-utils]: https://github.com/iiau-tracker/SPLT/blob/a196e603798e9be969d9d985c087c11cad1cda43/utils/tracking_utils.py
[p-tracker]: https://github.com/iiau-tracker/SPLT/blob/a196e603798e9be969d9d985c087c11cad1cda43/SPLT_tracker_new.py
[p-verifier]: https://github.com/iiau-tracker/SPLT/blob/a196e603798e9be969d9d985c087c11cad1cda43/Verifier/resnet50_bin.py
[p-readme]: https://github.com/iiau-tracker/SPLT/blob/a196e603798e9be969d9d985c087c11cad1cda43/README.md
[p-requirements]: https://github.com/iiau-tracker/SPLT/blob/a196e603798e9be969d9d985c087c11cad1cda43/requirements.txt
