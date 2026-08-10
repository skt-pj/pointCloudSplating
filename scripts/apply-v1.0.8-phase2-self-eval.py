#!/usr/bin/env python3
"""Add machine-readable signed RGB/depth edge diagnostics for Phase 2.

The user must not have to export or inspect overlay JPEGs manually. This patch keeps Phase 3
blocked, but records enough signed/region-wise alignment evidence in phase2_evaluation.json for
remote review from the existing Drive diagnostics file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "app/src/main/java/com/sktpj/pointcloudsplatting/Phase2DatasetEvaluator.java"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"v1.0.8 patch failed: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "        final JSONArray overlayResults = new JSONArray();\n\n        int depthObservationCount;",
        "        final JSONArray overlayResults = new JSONArray();\n"
        "        final List<Double> overlaySignedDxMedians = new ArrayList<>();\n"
        "        final List<Double> overlaySignedDyMedians = new ArrayList<>();\n\n"
        "        int depthObservationCount;",
        "signed overlay collections",
    )

    text = replace_once(
        text,
        "        double edgeMedianPx = Double.NaN;\n        double edgeP90Px = Double.NaN;\n",
        "        double edgeMedianPx = Double.NaN;\n"
        "        double edgeP90Px = Double.NaN;\n"
        "        double systematicDxMedianPx = Double.NaN;\n"
        "        double systematicDyMedianPx = Double.NaN;\n"
        "        double systematicOffsetMagnitudePx = Double.NaN;\n",
        "systematic fields",
    )

    text = replace_once(
        text,
        "                overlayResults.put(overlay.json);\n"
        "                if (Double.isFinite(overlay.edgeErrorPx)) edgeErrors.add(overlay.edgeErrorPx);\n",
        "                overlayResults.put(overlay.json);\n"
        "                if (Double.isFinite(overlay.edgeErrorPx)) edgeErrors.add(overlay.edgeErrorPx);\n"
        "                double signedDx = overlay.json.optDouble(\"signed_dx_median_px\", Double.NaN);\n"
        "                double signedDy = overlay.json.optDouble(\"signed_dy_median_px\", Double.NaN);\n"
        "                if (Double.isFinite(signedDx)) overlaySignedDxMedians.add(signedDx);\n"
        "                if (Double.isFinite(signedDy)) overlaySignedDyMedians.add(signedDy);\n",
        "collect signed overlay metrics",
    )

    text = replace_once(
        text,
        "            if (!edgeErrors.isEmpty()) {\n"
        "                edgeMedianPx = percentile(edgeErrors, 0.50);\n"
        "                edgeP90Px = percentile(edgeErrors, 0.90);\n"
        "            }\n\n"
        "            if (DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX != null",
        "            if (!edgeErrors.isEmpty()) {\n"
        "                edgeMedianPx = percentile(edgeErrors, 0.50);\n"
        "                edgeP90Px = percentile(edgeErrors, 0.90);\n"
        "            }\n"
        "            systematicDxMedianPx = percentile(overlaySignedDxMedians, 0.50);\n"
        "            systematicDyMedianPx = percentile(overlaySignedDyMedians, 0.50);\n"
        "            if (Double.isFinite(systematicDxMedianPx) && Double.isFinite(systematicDyMedianPx)) {\n"
        "                systematicOffsetMagnitudePx = Math.hypot(systematicDxMedianPx, systematicDyMedianPx);\n"
        "            }\n\n"
        "            if (DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX != null",
        "aggregate signed overlay metrics",
    )

    text = replace_once(
        text,
        "            List<Integer> edgeCells = new ArrayList<>();\n"
        "            List<Double> errorsScaledPx = new ArrayList<>();\n",
        "            List<Integer> edgeCells = new ArrayList<>();\n"
        "            List<Double> errorsScaledPx = new ArrayList<>();\n"
        "            List<Double> signedDxScaledPx = new ArrayList<>();\n"
        "            List<Double> signedDyScaledPx = new ArrayList<>();\n"
        "            List<List<Double>> quadrantDx = Arrays.asList(\n"
        "                    new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());\n"
        "            List<List<Double>> quadrantDy = Arrays.asList(\n"
        "                    new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>());\n",
        "per-overlay signed samples",
    )

    text = replace_once(
        text,
        "                    double nearest = nearestGradientEdgeDistance(\n"
        "                            gradient,\n"
        "                            gradientThreshold,\n"
        "                            width,\n"
        "                            height,\n"
        "                            x,\n"
        "                            y,\n"
        "                            EDGE_SEARCH_RADIUS_PX);\n"
        "                    double sourceScale = 0.5 * (\n"
        "                            originalCamera.width / (double) width\n"
        "                                    + originalCamera.height / (double) height);\n"
        "                    errorsScaledPx.add(nearest * sourceScale);\n",
        "                    EdgeOffset nearest = nearestGradientEdgeOffset(\n"
        "                            gradient,\n"
        "                            gradientThreshold,\n"
        "                            width,\n"
        "                            height,\n"
        "                            x,\n"
        "                            y,\n"
        "                            EDGE_SEARCH_RADIUS_PX);\n"
        "                    double sourceScaleX = originalCamera.width / (double) width;\n"
        "                    double sourceScaleY = originalCamera.height / (double) height;\n"
        "                    double sourceScale = 0.5 * (sourceScaleX + sourceScaleY);\n"
        "                    errorsScaledPx.add(nearest.distancePx * sourceScale);\n"
        "                    if (nearest.found) {\n"
        "                        double dxScaled = nearest.dxPx * sourceScaleX;\n"
        "                        double dyScaled = nearest.dyPx * sourceScaleY;\n"
        "                        signedDxScaledPx.add(dxScaled);\n"
        "                        signedDyScaledPx.add(dyScaled);\n"
        "                        int quadrant = (y >= height / 2 ? 2 : 0) + (x >= width / 2 ? 1 : 0);\n"
        "                        quadrantDx.get(quadrant).add(dxScaled);\n"
        "                        quadrantDy.get(quadrant).add(dyScaled);\n"
        "                    }\n",
        "signed nearest-edge offsets",
    )

    text = replace_once(
        text,
        "            json.put(\"depth_edge_alignment_error_px\", edgeError);\n"
        "            json.put(\"depth_edge_alignment_p90_px\", edgeP90);\n"
        "            json.put(\n"
        "                    \"edge_threshold_policy\",",
        "            json.put(\"depth_edge_alignment_error_px\", edgeError);\n"
        "            json.put(\"depth_edge_alignment_p90_px\", edgeP90);\n"
        "            json.put(\"signed_match_count\", signedDxScaledPx.size());\n"
        "            json.put(\"signed_dx_median_px\", finiteJson(percentile(signedDxScaledPx, 0.50)));\n"
        "            json.put(\"signed_dy_median_px\", finiteJson(percentile(signedDyScaledPx, 0.50)));\n"
        "            json.put(\"signed_dx_direction_consensus\", finiteJson(directionConsensus(signedDxScaledPx)));\n"
        "            json.put(\"signed_dy_direction_consensus\", finiteJson(directionConsensus(signedDyScaledPx)));\n"
        "            json.put(\"signed_offset_semantics\",\n"
        "                    \"+dx: nearest RGB edge is right of projected depth edge; +dy: below\");\n"
        "            JSONArray regions = new JSONArray();\n"
        "            String[] regionNames = {\"top_left\", \"top_right\", \"bottom_left\", \"bottom_right\"};\n"
        "            for (int region = 0; region < 4; region++) {\n"
        "                JSONObject regionJson = new JSONObject();\n"
        "                regionJson.put(\"region\", regionNames[region]);\n"
        "                regionJson.put(\"match_count\", quadrantDx.get(region).size());\n"
        "                regionJson.put(\"signed_dx_median_px\", finiteJson(percentile(quadrantDx.get(region), 0.50)));\n"
        "                regionJson.put(\"signed_dy_median_px\", finiteJson(percentile(quadrantDy.get(region), 0.50)));\n"
        "                regions.put(regionJson);\n"
        "            }\n"
        "            json.put(\"regions\", regions);\n"
        "            json.put(\n"
        "                    \"edge_threshold_policy\",",
        "overlay JSON signed metrics",
    )

    text = replace_once(
        text,
        "                overlayJson.put(\"edge_error_p90_px\", finiteJson(edgeP90Px));\n"
        "                overlayJson.put(\n"
        "                        \"hard_threshold_px\",",
        "                overlayJson.put(\"edge_error_p90_px\", finiteJson(edgeP90Px));\n"
        "                JSONObject systematic = new JSONObject();\n"
        "                systematic.put(\"view_median_dx_px\", finiteJson(systematicDxMedianPx));\n"
        "                systematic.put(\"view_median_dy_px\", finiteJson(systematicDyMedianPx));\n"
        "                systematic.put(\"view_median_vector_magnitude_px\", finiteJson(systematicOffsetMagnitudePx));\n"
        "                systematic.put(\"coordinate_semantics\",\n"
        "                        \"+dx RGB edge right of projected depth edge; +dy RGB edge below projected depth edge\");\n"
        "                systematic.put(\"review_source\", \"machine_metrics_in_drive_log_no_manual_file_export\");\n"
        "                overlayJson.put(\"systematic_offset_summary\", systematic);\n"
        "                overlayJson.put(\n"
        "                        \"hard_threshold_px\",",
        "systematic summary JSON",
    )

    text = replace_once(
        text,
        "    private static double nearestGradientEdgeDistance(\n"
        "            byte[] gradient,\n"
        "            int threshold,\n"
        "            int width,\n"
        "            int height,\n"
        "            int centerX,\n"
        "            int centerY,\n"
        "            int radius) {\n"
        "        double bestSq = Double.POSITIVE_INFINITY;\n"
        "        for (int dy = -radius; dy <= radius; dy++) {\n"
        "            int y = centerY + dy;\n"
        "            if (y <= 0 || y >= height - 1) continue;\n"
        "            for (int dx = -radius; dx <= radius; dx++) {\n"
        "                int x = centerX + dx;\n"
        "                if (x <= 0 || x >= width - 1) continue;\n"
        "                if ((gradient[y * width + x] & 0xff) < threshold) continue;\n"
        "                double sq = dx * (double) dx + dy * (double) dy;\n"
        "                if (sq < bestSq) bestSq = sq;\n"
        "            }\n"
        "        }\n"
        "        return Double.isFinite(bestSq) ? Math.sqrt(bestSq) : radius + 1.0;\n"
        "    }\n",
        "    private static EdgeOffset nearestGradientEdgeOffset(\n"
        "            byte[] gradient,\n"
        "            int threshold,\n"
        "            int width,\n"
        "            int height,\n"
        "            int centerX,\n"
        "            int centerY,\n"
        "            int radius) {\n"
        "        double bestSq = Double.POSITIVE_INFINITY;\n"
        "        int bestDx = 0;\n"
        "        int bestDy = 0;\n"
        "        boolean found = false;\n"
        "        for (int dy = -radius; dy <= radius; dy++) {\n"
        "            int y = centerY + dy;\n"
        "            if (y <= 0 || y >= height - 1) continue;\n"
        "            for (int dx = -radius; dx <= radius; dx++) {\n"
        "                int x = centerX + dx;\n"
        "                if (x <= 0 || x >= width - 1) continue;\n"
        "                if ((gradient[y * width + x] & 0xff) < threshold) continue;\n"
        "                double sq = dx * (double) dx + dy * (double) dy;\n"
        "                if (sq < bestSq) {\n"
        "                    bestSq = sq;\n"
        "                    bestDx = dx;\n"
        "                    bestDy = dy;\n"
        "                    found = true;\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "        return found\n"
        "                ? new EdgeOffset(true, bestDx, bestDy, Math.sqrt(bestSq))\n"
        "                : new EdgeOffset(false, Double.NaN, Double.NaN, radius + 1.0);\n"
        "    }\n\n"
        "    private static double directionConsensus(List<Double> values) {\n"
        "        int positive = 0;\n"
        "        int negative = 0;\n"
        "        for (double value : values) {\n"
        "            if (!Double.isFinite(value) || value == 0.0) continue;\n"
        "            if (value > 0.0) positive++; else negative++;\n"
        "        }\n"
        "        int count = positive + negative;\n"
        "        return count == 0 ? Double.NaN : Math.max(positive, negative) / (double) count;\n"
        "    }\n",
        "signed edge helper",
    )

    text = replace_once(
        text,
        "    private static final class OverlayResult {\n",
        "    private static final class EdgeOffset {\n"
        "        final boolean found;\n"
        "        final double dxPx;\n"
        "        final double dyPx;\n"
        "        final double distancePx;\n\n"
        "        EdgeOffset(boolean found, double dxPx, double dyPx, double distancePx) {\n"
        "            this.found = found;\n"
        "            this.dxPx = dxPx;\n"
        "            this.dyPx = dyPx;\n"
        "            this.distancePx = distancePx;\n"
        "        }\n"
        "    }\n\n"
        "    private static final class OverlayResult {\n",
        "EdgeOffset class",
    )

    text = text.replace(
        "v1.0.5 is the first Pixel 10a Phase 2 measurement build, so the RGB/depth-edge pixel threshold\n"
        " * is deliberately not hard-coded yet. Hard geometric gates can pass, but the overall result stays\n"
        " * REVIEW_REQUIRED until a real-device edge-error distribution is measured and the threshold is\n"
        " * frozen in a later build.",
        "The RGB/depth-edge absolute pixel threshold remains intentionally unfrozen until Pixel 10a evidence\n"
        " * is reviewed. v1.0.8 adds signed and quadrant edge offsets to the Drive JSON so that review can be\n"
        " * performed remotely without asking the user to export overlay image files manually."
    )

    PATH.write_text(text, encoding="utf-8")
    print("v1.0.8 Phase 2 self-evaluation metrics applied")


if __name__ == "__main__":
    main()
