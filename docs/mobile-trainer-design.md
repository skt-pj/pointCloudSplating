# PCS Android mobile 3DGS trainer design

## Scope

PCS owns the high-level on-device training policy. VkSplat is pinned as a low-level Vulkan rasterization/optimization implementation reference; PCS does not treat the upstream desktop/academic trainer policy as the product architecture.

This design intentionally separates correctness fixes in the pinned renderer (for example the cumsum bounds fix) from mobile training policy. Failures must be surfaced by semantic invariants rather than hidden by progressively smaller emergency constants.

## Research basis

- PocketGS, arXiv:2601.17354 — mobile 3DGS requires co-design of geometry, prior-conditioned initialization, and training; its initialization estimates local surface normals with K=16 and tangential scale with K=3 and uses a short mobile optimization regime.
- Taming 3DGS, arXiv:2406.15643 — Gaussian population is treated as an explicit budget rather than an uncontrolled consequence of Adaptive Density Control.
- gsplat / Nerfstudio MCMC strategy — bounded stochastic relocation/growth exposes a maximum Gaussian count and regularization parameters suitable for a fixed working-set policy.
- VkSplat, arXiv:2605.00219 and harry7557558/vksplat — supplies Vulkan compute kernels and gsplat-compatible training operations. Upstream explicitly describes VkSplat as academic/performance work and points practical users toward Spirulae-Splat.
- Spirulae-Splat — practical Vulkan trainer reference for future kernel/algorithm work, but not used as a drop-in Android dependency here.

## v1 training contract

1. Capture remains unchanged: high-resolution Camera2 RGB, synchronized ARCore root-local camera pose, calibrated intrinsics, and optional ARCore Raw Depth.
2. Raw Depth points remain metric input geometry. The VkSplat loader may normalize the scene internally before training.
3. Initial geometry is spatially budgeted deterministically to at most 90,000 points before optimizer state is allocated.
4. Each initial Gaussian is reconstructed as a local surface element:
   - K=16 local covariance neighbors;
   - smallest covariance eigenvector is the surface normal;
   - K=3 nearest-neighbor distance sets tangential scale;
   - Gaussian local Z is aligned to the normal;
   - normal thickness is 0.20 of tangential scale. The 0.20 ratio is a PCS engineering parameter, not a claim from PocketGS.
5. Density control uses VkSplat's MCMC implementation with an explicit 120,000 Gaussian cap. Desktop Default duplicate/split/prune ADC is not used by PCS mobile trainer v1.
6. Training is 750–1,000 iterations depending on saved view count. SH degree is advanced across quarters of the short schedule so SH3 is reached within the mobile budget.
7. The tile working set is budgeted before key/index buffer allocation.
8. At step 0, every 100 steps, and validation, PCS copies `tiles_touched` to CPU and checks:
   - every Gaussian touches between 0 and `grid_width * grid_height` tiles;
   - CPU sum of `tiles_touched` exactly equals the GPU prefix-sum result.
   This detects renderer/prefix-sum corruption as a correctness error rather than misclassifying it as an OOM.
9. Result metadata records the PCS trainer, the pinned VkSplat raster backend, strategy, initialization, Gaussian budget, and peak Vulkan allocation.

## Known geometry gap

This is not a full PocketGS reproduction. PocketGS performs global bundle adjustment over an independent feature-observation graph before initialization. Existing PCS saved datasets contain ARCore camera poses, calibrated RGB and Raw Depth, but do not yet persist a feature track graph suitable for Schur-complement BA.

Therefore mobile trainer v1 uses the existing metric ARCore root-local poses as camera initialization/poses. A future geometry stage should add robust RGB feature tracks and global BA, or an independently validated RGB-D pose refinement. That work must be benchmarked separately; it should not be approximated by silently perturbing ARCore poses in the trainer.

## CI contract

`scripts/verify-mobile-trainer.py` runs after the pinned VkSplat backend is prepared and before the APK build. It fails CI if the architecture drifts back toward desktop Default densification, an unbounded/long training schedule, loss of K16/K3 surface initialization, removal of projection invariants, or loss of the cumsum bounds fix.
