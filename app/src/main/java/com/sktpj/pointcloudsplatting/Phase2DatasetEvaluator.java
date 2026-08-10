package com.sktpj.pointcloudsplatting;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Phase 2 geometry/camera-view consistency evaluator.
 *
 * <p>Consumes only Phase 1 observation artifacts. It fuses independent {@code depth_obs_*}
 * observations into a root-local geometry prior, checks cross-depth consistency, projects that
 * geometry into every saved high-resolution RGB camera, and emits numerical/visual diagnostics.
 *
 * <p>This class intentionally does not invoke any 3DGS trainer or differentiable rasterizer.
 * v1.0.5 is the first Pixel 10a Phase 2 measurement build, so the RGB/depth-edge pixel threshold
 * is deliberately not hard-coded yet. Hard geometric gates can pass, but the overall result stays
 * REVIEW_REQUIRED until a real-device edge-error distribution is measured and the threshold is
 * frozen in a later build.
 */
public final class Phase2DatasetEvaluator {
    private static final String TAG = "Phase2Eval";

    private static final float MIN_CONFIDENCE = 0.30f;
    private static final float FUSION_VOXEL_METERS = 0.008f;
    private static final int MAX_FUSED_VOXELS = 320_000;
    private static final int SAMPLE_POINTS_PER_DEPTH = 640;
    private static final int MAX_PROJECTION_POINTS = 24_000;
    private static final float MATCH_CELL_METERS = 0.08f;
    private static final float MAX_MATCH_DISTANCE_METERS = 0.15f;
    private static final int MIN_MUTUAL_MATCHES_FOR_OVERLAP = 8;
    private static final int TEMPORAL_NEIGHBORS_TO_COMPARE = 4;

    // Phase 2 hard gates agreed before implementation.
    private static final double MAX_SURFACE_MEDIAN_METERS = 0.03;
    private static final double MAX_SURFACE_P90_METERS = 0.08;
    private static final double MAX_DEPTH_MEDIAN_METERS = 0.03;
    private static final double MAX_DEPTH_P90_METERS = 0.08;

    private static final int PROJECTION_GRID_W = 128;
    private static final int PROJECTION_GRID_H = 96;
    private static final int REQUIRED_OVERLAYS = 5;
    private static final int MAX_OVERLAY_LONG_EDGE = 1400;
    private static final int EDGE_SEARCH_RADIUS_PX = 8;

    // Intentionally unset for the first Pixel 10a Phase 2 run.
    private static final Double DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX = null;

    private static final int VOXEL_BITS = 21;
    private static final int VOXEL_OFFSET = 1 << 20;
    private static final long VOXEL_MASK = (1L << VOXEL_BITS) - 1L;

    private Phase2DatasetEvaluator() {}

    public static final class Result {
        public final boolean hardGatePassed;
        public final boolean passed;
        public final boolean reviewRequired;
        public final int failureCount;
        public final int warningCount;
        public final File reportFile;
        public final File geometryFile;
        public final JSONObject report;

        Result(
                boolean hardGatePassed,
                boolean passed,
                boolean reviewRequired,
                int failureCount,
                int warningCount,
                File reportFile,
                File geometryFile,
                JSONObject report) {
            this.hardGatePassed = hardGatePassed;
            this.passed = passed;
            this.reviewRequired = reviewRequired;
            this.failureCount = failureCount;
            this.warningCount = warningCount;
            this.reportFile = reportFile;
            this.geometryFile = geometryFile;
            this.report = report;
        }
    }

    public static Result evaluate(File dataset) {
        File reportFile = dataset == null ? null : new File(dataset, "phase2_evaluation.json");
        File geometryFile = dataset == null ? null : new File(dataset, "phase2_geometry_prior.ply");
        Evaluation state = new Evaluation(dataset);

        try {
            state.verifyPhase1();
            if (state.failures.isEmpty()) {
                state.loadAndFuseDepth();
                state.evaluateDepthConsistency();
                state.evaluateRgbProjectionAndOverlays();
            }
        } catch (Throwable error) {
            state.fail(
                    "phase2_evaluator_exception",
                    error.getClass().getSimpleName() + ": " + String.valueOf(error.getMessage()));
            DiagnosticLog.e(TAG, "Phase 2 evaluator failed", error);
        }

        boolean hardPass = state.failures.isEmpty();
        boolean reviewRequired = hardPass && DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX == null;
        boolean passed = hardPass && !reviewRequired;
        String status = passed ? "PASS" : (reviewRequired ? "REVIEW_REQUIRED" : "FAIL");

        JSONObject report = state.toJson(status, hardPass, passed, reviewRequired);
        if (reportFile != null) {
            try {
                writeJson(reportFile, report);
            } catch (Exception error) {
                DiagnosticLog.e(TAG, "Could not write phase2_evaluation.json", error);
                state.fail("phase2_report_write_failed", error.toString());
                hardPass = false;
                reviewRequired = false;
                passed = false;
                status = "FAIL";
                report = state.toJson(status, false, false, false);
                try {
                    writeJson(reportFile, report);
                } catch (Exception ignored) {
                }
            }
        }

        DiagnosticLog.i(
                TAG,
                String.format(
                        Locale.US,
                        "PHASE2_EVAL %s depth=%d/%d fused=%d pairMedianCm=%.2f pairP90Cm=%.2f "
                                + "depthMedianCm=%.2f depthP90Cm=%.2f rgb=%d/%d components=%d "
                                + "overlays=%d edgeMedianPx=%.2f failures=%d warnings=%d",
                        status,
                        state.depthUsedCount,
                        state.depthObservationCount,
                        state.fusedPoints.size(),
                        metersToCm(state.surfaceMedian),
                        metersToCm(state.surfaceP90),
                        metersToCm(state.depthMedian),
                        metersToCm(state.depthP90),
                        state.rgbProjectionValidCount,
                        state.rgbViewCount,
                        state.rgbConnectedComponents,
                        state.overlayResults.length(),
                        finiteOrNegative(state.edgeMedianPx),
                        state.failures.size(),
                        state.warnings.size()));

        DriveDiagnosticLogStore.setPrimaryDataset(dataset);
        DriveDiagnosticLogStore.requestOverwrite();

        return new Result(
                hardPass,
                passed,
                reviewRequired,
                state.failures.size(),
                state.warnings.size(),
                reportFile,
                geometryFile,
                report);
    }

    public static boolean needsEvaluation(File dataset) {
        JSONObject json = readStored(dataset);
        return json == null
                || !BuildConfig.VERSION_NAME.equals(json.optString("evaluator_build", ""));
    }

    public static boolean hasStoredPass(File dataset) {
        JSONObject json = readStored(dataset);
        return json != null
                && json.optBoolean("pass", false)
                && "PASS".equals(json.optString("status", ""));
    }

    public static boolean hasStoredHardGatePass(File dataset) {
        JSONObject json = readStored(dataset);
        return json != null && json.optBoolean("hard_gate_pass", false);
    }

    private static JSONObject readStored(File dataset) {
        if (dataset == null) return null;
        File file = new File(dataset, "phase2_evaluation.json");
        if (!file.isFile()) return null;
        try {
            return new JSONObject(readText(file));
        } catch (Exception ignored) {
            return null;
        }
    }

    private static final class Evaluation {
        final File dataset;
        final List<JSONObject> failures = new ArrayList<>();
        final List<JSONObject> warnings = new ArrayList<>();
        final List<DepthObservation> depthObservations = new ArrayList<>();
        final Map<Long, FusedAccumulator> fusion = new HashMap<>();
        final List<FusedPoint> fusedPoints = new ArrayList<>();
        final List<Double> surfaceResiduals = new ArrayList<>();
        final List<Double> cameraDepthResiduals = new ArrayList<>();
        final JSONArray overlappingPairs = new JSONArray();
        final JSONArray rgbViews = new JSONArray();
        final JSONArray overlayResults = new JSONArray();

        int depthObservationCount;
        int depthUsedCount;
        int depthExcludedMetadataCount;
        int depthExcludedQualityCount;
        long rawDepthPointCount;
        long confidenceRejectedCount;
        long rangeRejectedCount;
        long capacityRejectedCount;
        int isolatedVoxelRejectedCount;

        int overlappingPairCount;
        long mutualSurfaceMatchCount;
        long cameraDepthResidualCount;
        double surfaceMedian = Double.NaN;
        double surfaceP90 = Double.NaN;
        double depthMedian = Double.NaN;
        double depthP90 = Double.NaN;

        int rgbViewCount;
        int rgbProjectionValidCount;
        int rgbZeroProjectionCount;
        int rgbConnectedComponents;
        long rgbProjectedPointTotal;
        long rgbVisiblePointTotal;
        double coverageMedian = Double.NaN;
        double coverageMin = Double.NaN;
        double coverageMax = Double.NaN;
        double edgeMedianPx = Double.NaN;
        double edgeP90Px = Double.NaN;

        double minX = Double.POSITIVE_INFINITY;
        double minY = Double.POSITIVE_INFINITY;
        double minZ = Double.POSITIVE_INFINITY;
        double maxX = Double.NEGATIVE_INFINITY;
        double maxY = Double.NEGATIVE_INFINITY;
        double maxZ = Double.NEGATIVE_INFINITY;

        Evaluation(File dataset) {
            this.dataset = dataset;
            if (dataset == null || !dataset.isDirectory()) {
                fail("dataset_unavailable", "dataset directory unavailable");
            }
        }

        void verifyPhase1() {
            if (dataset == null || !dataset.isDirectory()) return;
            if (!Phase1DatasetEvaluator.hasStoredPass(dataset)) {
                fail("phase1_not_pass", "Phase 1 evaluation is not PASS");
            }
        }

        void loadAndFuseDepth() throws Exception {
            File[] metadataFiles = dataset.listFiles(
                    (dir, name) -> name.startsWith("depth_obs_") && name.endsWith(".json"));
            if (metadataFiles == null) metadataFiles = new File[0];
            Arrays.sort(metadataFiles, Comparator.comparing(File::getName));
            depthObservationCount = metadataFiles.length;
            if (depthObservationCount == 0) {
                fail("depth_observations_missing", "no independent depth_obs_*.json observations");
                return;
            }

            for (int i = 0; i < metadataFiles.length; i++) {
                File metadataFile = metadataFiles[i];
                JSONObject metadata;
                try {
                    metadata = new JSONObject(readText(metadataFile));
                } catch (Exception error) {
                    depthExcludedMetadataCount++;
                    fail("depth_metadata_invalid", metadataFile.getName() + ": " + error);
                    continue;
                }

                String plyName = metadata.optString("point_cloud", "");
                File ply = new File(dataset, plyName);
                double[] rootFromCamera = parseColumnMajorMatrix(
                        metadata.optJSONArray("root_from_depth_camera_column_major"));
                if (!"raw_depth".equals(metadata.optString("observation_type", ""))
                        || plyName.isEmpty()
                        || !ply.isFile()
                        || rootFromCamera == null) {
                    depthExcludedMetadataCount++;
                    fail("depth_observation_incomplete", metadataFile.getName());
                    continue;
                }

                long timestamp = metadata.optLong(
                        "android_camera_timestamp_ns",
                        metadata.optLong("raw_depth_timestamp_ns", i + 1L));
                DepthObservation observation = new DepthObservation(
                        metadata.optString("depth_observation_id", metadataFile.getName()),
                        timestamp,
                        rootFromCamera,
                        metadataFile,
                        ply);
                try {
                    readDepthObservation(observation);
                    depthObservations.add(observation);
                    depthUsedCount++;
                } catch (Exception error) {
                    String message = String.valueOf(error.getMessage());
                    if ("no confidence-filtered sample points".equals(message)) {
                        // The observation is structurally valid (Phase 1 already proved that),
                        // but contributes no geometry after the Phase 2 quality filter. Exclude
                        // it with an explicit reason; do not misclassify normal low-confidence
                        // sensor output as a corrupt PLY or a dataset hard failure.
                        depthExcludedQualityCount++;
                        warn("depth_observation_no_usable_points",
                                ply.getName() + ": all points rejected by Phase 2 confidence filter");
                    } else {
                        depthExcludedMetadataCount++;
                        fail("depth_ply_read_failed", ply.getName() + ": " + message);
                    }
                }
            }

            if (depthUsedCount == 0) {
                fail("depth_all_excluded", "no usable independent Depth observations");
                return;
            }

            depthObservations.sort(Comparator.comparingLong(value -> value.timestampNs));
            removeIsolatedFusionVoxels();
            if (fusedPoints.size() < 64) {
                fail("geometry_prior_too_small", "fused geometry has only " + fusedPoints.size() + " points");
                return;
            }

            writeGeometryPrior(new File(dataset, "phase2_geometry_prior.ply"), fusedPoints);
        }

        void readDepthObservation(DepthObservation observation) throws Exception {
            PlyHeader header = readPlyHeader(observation.plyFile);
            if (header.xOffset < 0 || header.yOffset < 0 || header.zOffset < 0
                    || header.confidenceOffset < 0) {
                throw new IOException("required XYZ/confidence properties missing");
            }
            long bodyBytes = (long) header.vertexCount * header.recordBytes;
            if (bodyBytes <= 0L
                    || observation.plyFile.length() < header.headerBytes + bodyBytes) {
                throw new IOException("truncated PLY");
            }
            rawDepthPointCount += header.vertexCount;

            try (FileInputStream input = new FileInputStream(observation.plyFile);
                 FileChannel channel = input.getChannel()) {
                MappedByteBuffer body = channel.map(
                        FileChannel.MapMode.READ_ONLY,
                        header.headerBytes,
                        bodyBytes);
                body.order(ByteOrder.LITTLE_ENDIAN);
                int accepted = 0;
                for (int pointIndex = 0; pointIndex < header.vertexCount; pointIndex++) {
                    int base = pointIndex * header.recordBytes;
                    float x = body.getFloat(base + header.xOffset);
                    float y = body.getFloat(base + header.yOffset);
                    float z = body.getFloat(base + header.zOffset);
                    float confidence = body.getFloat(base + header.confidenceOffset);
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)
                            || !Float.isFinite(confidence)) {
                        fail("depth_nonfinite_point",
                                observation.plyFile.getName() + " point=" + pointIndex);
                        continue;
                    }
                    if (confidence < MIN_CONFIDENCE) {
                        confidenceRejectedCount++;
                        continue;
                    }

                    double cameraDistance = distanceToCamera(
                            x, y, z, observation.rootFromCamera);
                    // Deliberately broad sanity bound; Raw Depth confidence remains the real quality filter.
                    if (!Double.isFinite(cameraDistance)
                            || cameraDistance < 0.05
                            || cameraDistance > 20.0) {
                        rangeRejectedCount++;
                        continue;
                    }

                    int red = header.redOffset >= 0 ? body.get(base + header.redOffset) & 0xff : 128;
                    int green = header.greenOffset >= 0 ? body.get(base + header.greenOffset) & 0xff : 128;
                    int blue = header.blueOffset >= 0 ? body.get(base + header.blueOffset) & 0xff : 128;

                    accepted++;
                    observation.offerSample(x, y, z, confidence, accepted);

                    int qx = fastFloor(x / FUSION_VOXEL_METERS);
                    int qy = fastFloor(y / FUSION_VOXEL_METERS);
                    int qz = fastFloor(z / FUSION_VOXEL_METERS);
                    if (Math.abs(qx) >= VOXEL_OFFSET
                            || Math.abs(qy) >= VOXEL_OFFSET
                            || Math.abs(qz) >= VOXEL_OFFSET) {
                        rangeRejectedCount++;
                        continue;
                    }
                    long key = pack(qx, qy, qz);
                    FusedAccumulator accumulator = fusion.get(key);
                    if (accumulator == null) {
                        if (fusion.size() >= MAX_FUSED_VOXELS) {
                            capacityRejectedCount++;
                            continue;
                        }
                        accumulator = new FusedAccumulator(key, qx, qy, qz);
                        fusion.put(key, accumulator);
                    }
                    double weight = Math.max(MIN_CONFIDENCE, confidence);
                    accumulator.x += x * weight;
                    accumulator.y += y * weight;
                    accumulator.z += z * weight;
                    accumulator.r += red * weight;
                    accumulator.g += green * weight;
                    accumulator.b += blue * weight;
                    accumulator.confidence += confidence * weight;
                    accumulator.weight += weight;
                    accumulator.hits++;
                }
                observation.acceptedPointCount = accepted;
            }
            if (observation.samplePoints.isEmpty()) {
                throw new IOException("no confidence-filtered sample points");
            }
        }

        void removeIsolatedFusionVoxels() {
            for (FusedAccumulator value : fusion.values()) {
                boolean keep = value.hits >= 2;
                if (!keep) {
                    outer:
                    for (int dz = -1; dz <= 1; dz++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                int qx = value.qx + dx;
                                int qy = value.qy + dy;
                                int qz = value.qz + dz;
                                if (Math.abs(qx) >= VOXEL_OFFSET
                                        || Math.abs(qy) >= VOXEL_OFFSET
                                        || Math.abs(qz) >= VOXEL_OFFSET) {
                                    continue;
                                }
                                if (fusion.containsKey(pack(qx, qy, qz))) {
                                    keep = true;
                                    break outer;
                                }
                            }
                        }
                    }
                }
                if (!keep) {
                    isolatedVoxelRejectedCount++;
                    continue;
                }
                double inv = 1.0 / Math.max(1e-9, value.weight);
                FusedPoint point = new FusedPoint(
                        (float) (value.x * inv),
                        (float) (value.y * inv),
                        (float) (value.z * inv),
                        clampByte((int) Math.round(value.r * inv)),
                        clampByte((int) Math.round(value.g * inv)),
                        clampByte((int) Math.round(value.b * inv)),
                        (float) Math.max(0.0, Math.min(1.0, value.confidence * inv)));
                fusedPoints.add(point);
                minX = Math.min(minX, point.x);
                minY = Math.min(minY, point.y);
                minZ = Math.min(minZ, point.z);
                maxX = Math.max(maxX, point.x);
                maxY = Math.max(maxY, point.y);
                maxZ = Math.max(maxZ, point.z);
            }
            fusion.clear();
            if (capacityRejectedCount > 0) {
                warn(
                        "fusion_capacity_reached",
                        "rejected " + capacityRejectedCount
                                + " new voxels after mobile cap=" + MAX_FUSED_VOXELS);
            }
        }

        void evaluateDepthConsistency() throws JSONException {
            if (depthObservations.size() < 2) {
                fail("depth_overlap_unavailable", "need at least two usable Depth observations");
                return;
            }

            for (int i = 0; i < depthObservations.size(); i++) {
                DepthObservation a = depthObservations.get(i);
                int maxJ = Math.min(
                        depthObservations.size(),
                        i + 1 + TEMPORAL_NEIGHBORS_TO_COMPARE);
                for (int j = i + 1; j < maxJ; j++) {
                    DepthObservation b = depthObservations.get(j);
                    PairMetrics metrics = compareMutualNearest(a, b);
                    if (metrics.mutualCount < MIN_MUTUAL_MATCHES_FOR_OVERLAP) continue;

                    overlappingPairCount++;
                    mutualSurfaceMatchCount += metrics.mutualCount;
                    for (double value : metrics.surfaceResiduals) surfaceResiduals.add(value);
                    for (double value : metrics.cameraDepthResiduals) {
                        cameraDepthResiduals.add(value);
                    }
                    cameraDepthResidualCount += metrics.cameraDepthResiduals.size();

                    JSONObject pair = new JSONObject();
                    pair.put("a", a.id);
                    pair.put("b", b.id);
                    pair.put("mutual_matches", metrics.mutualCount);
                    pair.put("surface_median_m", percentile(metrics.surfaceResiduals, 0.50));
                    pair.put("surface_p90_m", percentile(metrics.surfaceResiduals, 0.90));
                    pair.put(
                            "camera_depth_median_m",
                            percentile(metrics.cameraDepthResiduals, 0.50));
                    overlappingPairs.put(pair);
                }
            }

            if (overlappingPairCount == 0 || surfaceResiduals.isEmpty()) {
                fail("depth_overlap_pairs_missing", "no overlapping Depth observation pairs found");
                return;
            }
            if (cameraDepthResiduals.isEmpty()) {
                fail("depth_renderback_samples_missing", "no camera-space leave-one-out depth residuals");
                return;
            }

            surfaceMedian = percentile(surfaceResiduals, 0.50);
            surfaceP90 = percentile(surfaceResiduals, 0.90);
            depthMedian = percentile(cameraDepthResiduals, 0.50);
            depthP90 = percentile(cameraDepthResiduals, 0.90);

            if (surfaceMedian > MAX_SURFACE_MEDIAN_METERS) {
                fail(
                        "surface_residual_median",
                        String.format(Locale.US, "%.4fm > %.4fm", surfaceMedian, MAX_SURFACE_MEDIAN_METERS));
            }
            if (surfaceP90 > MAX_SURFACE_P90_METERS) {
                fail(
                        "surface_residual_p90",
                        String.format(Locale.US, "%.4fm > %.4fm", surfaceP90, MAX_SURFACE_P90_METERS));
            }
            if (depthMedian > MAX_DEPTH_MEDIAN_METERS) {
                fail(
                        "camera_depth_residual_median",
                        String.format(Locale.US, "%.4fm > %.4fm", depthMedian, MAX_DEPTH_MEDIAN_METERS));
            }
            if (depthP90 > MAX_DEPTH_P90_METERS) {
                fail(
                        "camera_depth_residual_p90",
                        String.format(Locale.US, "%.4fm > %.4fm", depthP90, MAX_DEPTH_P90_METERS));
            }

            warn(
                    "raw_depth_pixel_grid_not_persisted",
                    "v1.0.4 saved root-local Depth points but not the raw depth pixel grid/intrinsics. "
                            + "Phase 2 therefore uses camera-space leave-one-out nearest-surface depth "
                            + "residuals; it does not pretend to perform exact per-pixel render-back.");
        }

        PairMetrics compareMutualNearest(DepthObservation a, DepthObservation b) {
            List<SamplePoint> aPoints = a.samplePoints;
            List<SamplePoint> bPoints = b.samplePoints;
            SpatialIndex bIndex = new SpatialIndex(bPoints, MATCH_CELL_METERS);
            SpatialIndex aIndex = new SpatialIndex(aPoints, MATCH_CELL_METERS);

            Nearest[] aToB = new Nearest[aPoints.size()];
            for (int i = 0; i < aPoints.size(); i++) {
                aToB[i] = bIndex.nearest(aPoints.get(i), MAX_MATCH_DISTANCE_METERS);
            }
            Nearest[] bToA = new Nearest[bPoints.size()];
            for (int i = 0; i < bPoints.size(); i++) {
                bToA[i] = aIndex.nearest(bPoints.get(i), MAX_MATCH_DISTANCE_METERS);
            }

            PairMetrics out = new PairMetrics();
            for (int ai = 0; ai < aToB.length; ai++) {
                Nearest nearest = aToB[ai];
                if (nearest.index < 0 || nearest.index >= bToA.length) continue;
                Nearest reverse = bToA[nearest.index];
                if (reverse.index != ai) continue;

                SamplePoint pa = aPoints.get(ai);
                SamplePoint pb = bPoints.get(nearest.index);
                out.mutualCount++;
                out.surfaceResiduals.add(nearest.distance);

                double da = forwardDepthInCamera(pa, a.rootFromCamera);
                double dbInA = forwardDepthInCamera(pb, a.rootFromCamera);
                if (da > 0.0 && dbInA > 0.0) {
                    out.cameraDepthResiduals.add(Math.abs(da - dbInA));
                }

                double db = forwardDepthInCamera(pb, b.rootFromCamera);
                double daInB = forwardDepthInCamera(pa, b.rootFromCamera);
                if (db > 0.0 && daInB > 0.0) {
                    out.cameraDepthResiduals.add(Math.abs(db - daInB));
                }
            }
            return out;
        }

        void evaluateRgbProjectionAndOverlays() throws Exception {
            File transformsFile = new File(dataset, "transforms.json");
            if (!transformsFile.isFile()) {
                fail("transforms_missing", "transforms.json missing");
                return;
            }
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.optJSONArray("frames");
            if (frames == null) {
                fail("rgb_frames_missing", "transforms.json has no frames");
                return;
            }
            rgbViewCount = frames.length();
            if (rgbViewCount == 0) {
                fail("rgb_frames_empty", "no RGB camera views");
                return;
            }

            List<FusedPoint> projectionPoints = sampleFusedPoints(fusedPoints, MAX_PROJECTION_POINTS);
            List<BitSet> visibleSets = new ArrayList<>();
            List<Double> coverages = new ArrayList<>();

            for (int i = 0; i < frames.length(); i++) {
                JSONObject frame = frames.getJSONObject(i);
                CameraView camera = CameraView.from(frame);
                ProjectionResult projection = projectCamera(camera, projectionPoints);
                visibleSets.add(projection.visiblePointIndices);
                coverages.add(projection.coverage);
                rgbProjectedPointTotal += projection.projectedCount;
                rgbVisiblePointTotal += projection.visiblePointCount;

                JSONObject view = new JSONObject();
                view.put("index", i);
                view.put("file_path", frame.optString("file_path", ""));
                view.put("projected_points", projection.projectedCount);
                view.put("visible_points", projection.visiblePointCount);
                view.put("image_coverage", projection.coverage);
                rgbViews.put(view);

                if (projection.projectedCount <= 0) {
                    rgbZeroProjectionCount++;
                    fail("rgb_zero_projection", "RGB view " + i + " has zero projected geometry points");
                } else {
                    rgbProjectionValidCount++;
                }
            }

            rgbConnectedComponents = countConnectedComponents(visibleSets);
            if (rgbConnectedComponents != 1) {
                fail(
                        "rgb_overlap_graph_disconnected",
                        "RGB overlap graph has " + rgbConnectedComponents + " connected components");
            }

            coverageMedian = percentile(coverages, 0.50);
            coverageMin = coverages.isEmpty() ? Double.NaN : Collections.min(coverages);
            coverageMax = coverages.isEmpty() ? Double.NaN : Collections.max(coverages);

            if (rgbViewCount < REQUIRED_OVERLAYS) {
                fail(
                        "overlay_views_insufficient",
                        "need at least " + REQUIRED_OVERLAYS + " distinct RGB views for overlays");
                return;
            }

            int[] overlayIndices = selectOverlayIndices(rgbViewCount, REQUIRED_OVERLAYS);
            List<Double> edgeErrors = new ArrayList<>();
            for (int ordinal = 0; ordinal < overlayIndices.length; ordinal++) {
                int viewIndex = overlayIndices[ordinal];
                JSONObject frame = frames.getJSONObject(viewIndex);
                OverlayResult overlay = createOverlay(
                        frame,
                        viewIndex,
                        ordinal + 1,
                        projectionPoints);
                overlayResults.put(overlay.json);
                if (Double.isFinite(overlay.edgeErrorPx)) edgeErrors.add(overlay.edgeErrorPx);
            }

            if (overlayResults.length() < REQUIRED_OVERLAYS) {
                fail(
                        "overlay_generation_incomplete",
                        "generated " + overlayResults.length() + "/" + REQUIRED_OVERLAYS + " overlays");
            }

            if (!edgeErrors.isEmpty()) {
                edgeMedianPx = percentile(edgeErrors, 0.50);
                edgeP90Px = percentile(edgeErrors, 0.90);
            }

            if (DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX != null
                    && Double.isFinite(edgeP90Px)
                    && edgeP90Px > DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX) {
                fail(
                        "depth_edge_alignment",
                        "edge p90=" + edgeP90Px
                                + "px > " + DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX + "px");
            }
        }

        ProjectionResult projectCamera(CameraView camera, List<FusedPoint> points) {
            double[] zBuffer = new double[PROJECTION_GRID_W * PROJECTION_GRID_H];
            Arrays.fill(zBuffer, Double.POSITIVE_INFINITY);
            int[] pointBuffer = new int[zBuffer.length];
            Arrays.fill(pointBuffer, -1);
            int projected = 0;

            for (int i = 0; i < points.size(); i++) {
                FusedPoint p = points.get(i);
                CameraPoint cp = worldToCamera(p.x, p.y, p.z, camera.c2w);
                if (!(cp.depth > 0.0)) continue;
                double u = camera.cx + camera.fx * cp.x / cp.depth;
                double v = camera.cy - camera.fy * cp.y / cp.depth;
                if (!(u >= 0.0 && u < camera.width && v >= 0.0 && v < camera.height)) continue;
                projected++;
                int gx = Math.min(PROJECTION_GRID_W - 1,
                        Math.max(0, (int) (u * PROJECTION_GRID_W / camera.width)));
                int gy = Math.min(PROJECTION_GRID_H - 1,
                        Math.max(0, (int) (v * PROJECTION_GRID_H / camera.height)));
                int cell = gy * PROJECTION_GRID_W + gx;
                if (cp.depth < zBuffer[cell]) {
                    zBuffer[cell] = cp.depth;
                    pointBuffer[cell] = i;
                }
            }

            BitSet visible = new BitSet(points.size());
            int occupied = 0;
            for (int index : pointBuffer) {
                if (index >= 0) {
                    visible.set(index);
                    occupied++;
                }
            }
            double coverage = occupied / (double) pointBuffer.length;
            return new ProjectionResult(projected, occupied, coverage, visible);
        }

        int countConnectedComponents(List<BitSet> visibleSets) {
            int n = visibleSets.size();
            if (n == 0) return 0;
            boolean[] visited = new boolean[n];
            int components = 0;
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            for (int start = 0; start < n; start++) {
                if (visited[start]) continue;
                components++;
                visited[start] = true;
                queue.add(start);
                while (!queue.isEmpty()) {
                    int current = queue.removeFirst();
                    for (int other = 0; other < n; other++) {
                        if (visited[other] || current == other) continue;
                        BitSet intersection = (BitSet) visibleSets.get(current).clone();
                        intersection.and(visibleSets.get(other));
                        if (!intersection.isEmpty()) {
                            visited[other] = true;
                            queue.addLast(other);
                        }
                    }
                }
            }
            return components;
        }

        OverlayResult createOverlay(
                JSONObject frame,
                int viewIndex,
                int ordinal,
                List<FusedPoint> points) throws Exception {
            CameraView originalCamera = CameraView.from(frame);
            File source = new File(dataset, frame.getString("file_path"));
            if (!source.isFile()) throw new IOException("missing overlay JPEG " + source.getName());

            Bitmap bitmap = decodeForOverlay(source, originalCamera.width, originalCamera.height);
            if (bitmap == null) throw new IOException("could not decode overlay JPEG " + source.getName());

            Bitmap mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            if (mutable == null) {
                bitmap.recycle();
                throw new IOException("could not allocate overlay bitmap");
            }
            if (mutable != bitmap) bitmap.recycle();

            double sx = mutable.getWidth() / (double) originalCamera.width;
            double sy = mutable.getHeight() / (double) originalCamera.height;
            CameraView camera = originalCamera.scaled(sx, sy, mutable.getWidth(), mutable.getHeight());

            int width = mutable.getWidth();
            int height = mutable.getHeight();
            int analysisW = Math.min(360, width);
            int analysisH = Math.max(1, (int) Math.round(height * (analysisW / (double) width)));
            float[] zBuffer = new float[analysisW * analysisH];
            Arrays.fill(zBuffer, Float.POSITIVE_INFINITY);
            int projected = 0;
            for (FusedPoint p : points) {
                CameraPoint cp = worldToCamera(p.x, p.y, p.z, camera.c2w);
                if (!(cp.depth > 0.0)) continue;
                double u = camera.cx + camera.fx * cp.x / cp.depth;
                double v = camera.cy - camera.fy * cp.y / cp.depth;
                if (!(u >= 0.0 && u < width && v >= 0.0 && v < height)) continue;
                projected++;
                int gx = Math.min(analysisW - 1, Math.max(0, (int) (u * analysisW / width)));
                int gy = Math.min(analysisH - 1, Math.max(0, (int) (v * analysisH / height)));
                int index = gy * analysisW + gx;
                if (cp.depth < zBuffer[index]) zBuffer[index] = (float) cp.depth;
            }

            byte[] gray = grayscale(mutable);
            byte[] gradient = sobelMagnitude(gray, width, height);
            int gradientThreshold = percentileByte(gradient, 0.85);

            List<Integer> edgeCells = new ArrayList<>();
            List<Double> errorsScaledPx = new ArrayList<>();
            for (int gy = 1; gy < analysisH - 1; gy++) {
                for (int gx = 1; gx < analysisW - 1; gx++) {
                    int index = gy * analysisW + gx;
                    float depth = zBuffer[index];
                    if (!Float.isFinite(depth)) continue;
                    boolean edge = false;
                    float threshold = Math.max(0.03f, depth * 0.05f);
                    int[] neighbors = {
                            index - 1,
                            index + 1,
                            index - analysisW,
                            index + analysisW
                    };
                    for (int neighbor : neighbors) {
                        float other = zBuffer[neighbor];
                        if (!Float.isFinite(other) || Math.abs(depth - other) > threshold) {
                            edge = true;
                            break;
                        }
                    }
                    if (!edge) continue;
                    edgeCells.add(index);
                    int x = Math.min(width - 1,
                            Math.max(0, (int) Math.round((gx + 0.5) * width / analysisW)));
                    int y = Math.min(height - 1,
                            Math.max(0, (int) Math.round((gy + 0.5) * height / analysisH)));
                    double nearest = nearestGradientEdgeDistance(
                            gradient,
                            gradientThreshold,
                            width,
                            height,
                            x,
                            y,
                            EDGE_SEARCH_RADIUS_PX);
                    double sourceScale = 0.5 * (
                            originalCamera.width / (double) width
                                    + originalCamera.height / (double) height);
                    errorsScaledPx.add(nearest * sourceScale);
                }
            }

            Canvas canvas = new Canvas(mutable);
            Paint paint = new Paint();
            paint.setColor(Color.argb(220, 255, 80, 40));
            paint.setStrokeWidth(2.0f);
            paint.setStyle(Paint.Style.FILL);
            for (int index : edgeCells) {
                int gx = index % analysisW;
                int gy = index / analysisW;
                float x = (float) ((gx + 0.5) * width / analysisW);
                float y = (float) ((gy + 0.5) * height / analysisH);
                canvas.drawCircle(x, y, 1.8f, paint);
            }

            File target = new File(
                    dataset,
                    String.format(Locale.US, "phase2_overlay_%02d_view_%03d.jpg", ordinal, viewIndex));
            try (FileOutputStream out = new FileOutputStream(target)) {
                if (!mutable.compress(Bitmap.CompressFormat.JPEG, 92, out)) {
                    throw new IOException("overlay JPEG compression failed");
                }
            }
            mutable.recycle();

            double edgeError = percentile(errorsScaledPx, 0.50);
            double edgeP90 = percentile(errorsScaledPx, 0.90);
            int visible = countFinite(zBuffer);
            double coverage = visible / (double) (analysisW * analysisH);

            JSONObject json = new JSONObject();
            json.put("view_index", viewIndex);
            json.put("source_image", source.getName());
            json.put("overlay", target.getName());
            json.put("projected_points", projected);
            json.put("visible_cells", visible);
            json.put("image_coverage", coverage);
            json.put("analysis_grid", analysisW + "x" + analysisH);
            json.put("depth_edge_points", edgeCells.size());
            json.put("rgb_edge_gradient_threshold", gradientThreshold);
            json.put("depth_edge_alignment_error_px", edgeError);
            json.put("depth_edge_alignment_p90_px", edgeP90);
            json.put(
                    "edge_threshold_policy",
                    DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX == null
                            ? "baseline_measurement_only_no_hard_threshold"
                            : "hard_threshold_applied");
            return new OverlayResult(edgeError, json);
        }

        JSONObject toJson(
                String status,
                boolean hardGatePass,
                boolean pass,
                boolean reviewRequired) {
            JSONObject out = new JSONObject();
            try {
                out.put("format_version", 1);
                out.put("phase", "phase2");
                out.put("evaluator_build", BuildConfig.VERSION_NAME);
                out.put("status", status);
                out.put("pass", pass);
                out.put("hard_gate_pass", hardGatePass);
                out.put("review_required", reviewRequired);
                out.put("next_phase_allowed", pass);
                out.put("phase1_pass", dataset != null && Phase1DatasetEvaluator.hasStoredPass(dataset));

                JSONObject fusionJson = new JSONObject();
                fusionJson.put("source", "independent_depth_obs_only");
                fusionJson.put("confidence_min", MIN_CONFIDENCE);
                fusionJson.put("voxel_m", FUSION_VOXEL_METERS);
                fusionJson.put("max_voxels", MAX_FUSED_VOXELS);
                fusionJson.put("depth_observation_count", depthObservationCount);
                fusionJson.put("depth_observation_used_count", depthUsedCount);
                fusionJson.put("depth_observation_excluded_metadata_count", depthExcludedMetadataCount);
                fusionJson.put("depth_observation_excluded_quality_count", depthExcludedQualityCount);
                fusionJson.put("raw_point_count", rawDepthPointCount);
                fusionJson.put("confidence_rejected_count", confidenceRejectedCount);
                fusionJson.put("range_rejected_count", rangeRejectedCount);
                fusionJson.put("capacity_rejected_count", capacityRejectedCount);
                fusionJson.put("isolated_voxel_rejected_count", isolatedVoxelRejectedCount);
                fusionJson.put("final_point_count", fusedPoints.size());
                fusionJson.put("output", "phase2_geometry_prior.ply");
                JSONObject bounds = new JSONObject();
                bounds.put("min_x", finiteJson(minX));
                bounds.put("min_y", finiteJson(minY));
                bounds.put("min_z", finiteJson(minZ));
                bounds.put("max_x", finiteJson(maxX));
                bounds.put("max_y", finiteJson(maxY));
                bounds.put("max_z", finiteJson(maxZ));
                fusionJson.put("scene_bounds_m", bounds);
                out.put("geometry_fusion", fusionJson);

                JSONObject pairJson = new JSONObject();
                pairJson.put("method", "temporal_neighbor_mutual_nearest_surface");
                pairJson.put("temporal_neighbor_span", TEMPORAL_NEIGHBORS_TO_COMPARE);
                pairJson.put("match_radius_m", MAX_MATCH_DISTANCE_METERS);
                pairJson.put("overlap_min_mutual_matches", MIN_MUTUAL_MATCHES_FOR_OVERLAP);
                pairJson.put("overlapping_pair_count", overlappingPairCount);
                pairJson.put("mutual_match_count", mutualSurfaceMatchCount);
                pairJson.put("median_m", finiteJson(surfaceMedian));
                pairJson.put("p90_m", finiteJson(surfaceP90));
                pairJson.put("accept_median_m", MAX_SURFACE_MEDIAN_METERS);
                pairJson.put("accept_p90_m", MAX_SURFACE_P90_METERS);
                pairJson.put("pairs", overlappingPairs);
                out.put("depth_surface_consistency", pairJson);

                JSONObject depthJson = new JSONObject();
                depthJson.put("method", "camera_space_leave_one_out_nearest_surface_depth");
                depthJson.put("exact_raw_pixel_renderback", false);
                depthJson.put(
                        "limitation",
                        "raw depth pixel grid/intrinsics were not persisted in Phase 1 v1.0.4; "
                                + "this metric compares overlapping independent observations after "
                                + "transforming matched surfaces into each source depth-camera frame");
                depthJson.put("sample_count", cameraDepthResidualCount);
                depthJson.put("median_m", finiteJson(depthMedian));
                depthJson.put("p90_m", finiteJson(depthP90));
                depthJson.put("accept_median_m", MAX_DEPTH_MEDIAN_METERS);
                depthJson.put("accept_p90_m", MAX_DEPTH_P90_METERS);
                out.put("depth_renderback_consistency", depthJson);

                JSONObject rgbJson = new JSONObject();
                rgbJson.put("view_count", rgbViewCount);
                rgbJson.put("projection_valid_view_count", rgbProjectionValidCount);
                rgbJson.put("zero_projection_view_count", rgbZeroProjectionCount);
                rgbJson.put("overlap_connected_components", rgbConnectedComponents);
                rgbJson.put("projected_point_total", rgbProjectedPointTotal);
                rgbJson.put("visible_point_total", rgbVisiblePointTotal);
                rgbJson.put("coverage_median", finiteJson(coverageMedian));
                rgbJson.put("coverage_min", finiteJson(coverageMin));
                rgbJson.put("coverage_max", finiteJson(coverageMax));
                rgbJson.put("views", rgbViews);
                out.put("rgb_projection", rgbJson);

                JSONObject overlayJson = new JSONObject();
                overlayJson.put("required_count", REQUIRED_OVERLAYS);
                overlayJson.put("generated_count", overlayResults.length());
                overlayJson.put("edge_error_median_px", finiteJson(edgeMedianPx));
                overlayJson.put("edge_error_p90_px", finiteJson(edgeP90Px));
                overlayJson.put(
                        "hard_threshold_px",
                        DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX == null
                                ? JSONObject.NULL
                                : DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX);
                overlayJson.put(
                        "threshold_policy",
                        DEPTH_EDGE_ALIGNMENT_HARD_THRESHOLD_PX == null
                                ? "FIRST_PIXEL10A_BASELINE_REQUIRED_DO_NOT_PASS_PHASE2_YET"
                                : "fixed_from_prior_pixel10a_measurement");
                overlayJson.put("overlays", overlayResults);
                out.put("projection_overlays", overlayJson);

                out.put("failure_count", failures.size());
                out.put("warning_count", warnings.size());
                out.put("failures", new JSONArray(failures));
                out.put("warnings", new JSONArray(warnings));
            } catch (JSONException error) {
                try {
                    out.put("status", "FAIL");
                    out.put("pass", false);
                    out.put("hard_gate_pass", false);
                    out.put("serialization_error", error.toString());
                } catch (JSONException ignored) {
                }
            }
            return out;
        }

        void fail(String code, String detail) {
            failures.add(issue(code, detail));
        }

        void warn(String code, String detail) {
            warnings.add(issue(code, detail));
        }

        JSONObject issue(String code, String detail) {
            JSONObject json = new JSONObject();
            try {
                json.put("code", code);
                json.put("detail", detail);
            } catch (JSONException ignored) {
            }
            return json;
        }
    }

    private static final class DepthObservation {
        final String id;
        final long timestampNs;
        final double[] rootFromCamera;
        final File metadataFile;
        final File plyFile;
        final List<SamplePoint> samplePoints = new ArrayList<>(SAMPLE_POINTS_PER_DEPTH);
        int acceptedPointCount;

        DepthObservation(
                String id,
                long timestampNs,
                double[] rootFromCamera,
                File metadataFile,
                File plyFile) {
            this.id = id;
            this.timestampNs = timestampNs;
            this.rootFromCamera = rootFromCamera;
            this.metadataFile = metadataFile;
            this.plyFile = plyFile;
        }

        void offerSample(float x, float y, float z, float confidence, int acceptedCount) {
            SamplePoint point = new SamplePoint(x, y, z, confidence);
            if (samplePoints.size() < SAMPLE_POINTS_PER_DEPTH) {
                samplePoints.add(point);
                return;
            }
            long mixed = mix64(
                    acceptedCount * 0x9E3779B97F4A7C15L
                            ^ timestampNs
                            ^ id.hashCode());
            int slot = (int) Math.floorMod(mixed, acceptedCount);
            if (slot < SAMPLE_POINTS_PER_DEPTH) {
                samplePoints.set(slot, point);
            }
        }
    }

    private static final class SamplePoint {
        final float x;
        final float y;
        final float z;
        final float confidence;

        SamplePoint(float x, float y, float z, float confidence) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.confidence = confidence;
        }
    }

    private static final class FusedAccumulator {
        final long key;
        final int qx;
        final int qy;
        final int qz;
        double x, y, z;
        double r, g, b;
        double confidence;
        double weight;
        int hits;

        FusedAccumulator(long key, int qx, int qy, int qz) {
            this.key = key;
            this.qx = qx;
            this.qy = qy;
            this.qz = qz;
        }
    }

    private static final class FusedPoint {
        final float x, y, z;
        final int r, g, b;
        final float confidence;

        FusedPoint(float x, float y, float z, int r, int g, int b, float confidence) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.r = r;
            this.g = g;
            this.b = b;
            this.confidence = confidence;
        }
    }

    private static final class PairMetrics {
        int mutualCount;
        final List<Double> surfaceResiduals = new ArrayList<>();
        final List<Double> cameraDepthResiduals = new ArrayList<>();
    }

    private static final class Nearest {
        final int index;
        final double distance;

        Nearest(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }
    }

    private static final class SpatialIndex {
        final List<SamplePoint> points;
        final float cellMeters;
        final Map<Long, IntList> cells = new HashMap<>();

        SpatialIndex(List<SamplePoint> points, float cellMeters) {
            this.points = points;
            this.cellMeters = cellMeters;
            for (int i = 0; i < points.size(); i++) {
                SamplePoint p = points.get(i);
                int qx = fastFloor(p.x / cellMeters);
                int qy = fastFloor(p.y / cellMeters);
                int qz = fastFloor(p.z / cellMeters);
                if (Math.abs(qx) >= VOXEL_OFFSET
                        || Math.abs(qy) >= VOXEL_OFFSET
                        || Math.abs(qz) >= VOXEL_OFFSET) {
                    continue;
                }
                long key = pack(qx, qy, qz);
                IntList list = cells.get(key);
                if (list == null) {
                    list = new IntList();
                    cells.put(key, list);
                }
                list.add(i);
            }
        }

        Nearest nearest(SamplePoint query, float maxDistance) {
            int qx = fastFloor(query.x / cellMeters);
            int qy = fastFloor(query.y / cellMeters);
            int qz = fastFloor(query.z / cellMeters);
            double bestSq = maxDistance * (double) maxDistance;
            int bestIndex = -1;
            int radius = Math.max(1, (int) Math.ceil(maxDistance / cellMeters));
            for (int dz = -radius; dz <= radius; dz++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int x = qx + dx;
                        int y = qy + dy;
                        int z = qz + dz;
                        if (Math.abs(x) >= VOXEL_OFFSET
                                || Math.abs(y) >= VOXEL_OFFSET
                                || Math.abs(z) >= VOXEL_OFFSET) {
                            continue;
                        }
                        IntList list = cells.get(pack(x, y, z));
                        if (list == null) continue;
                        for (int k = 0; k < list.size; k++) {
                            int index = list.values[k];
                            SamplePoint p = points.get(index);
                            double ddx = query.x - p.x;
                            double ddy = query.y - p.y;
                            double ddz = query.z - p.z;
                            double sq = ddx * ddx + ddy * ddy + ddz * ddz;
                            if (sq < bestSq) {
                                bestSq = sq;
                                bestIndex = index;
                            }
                        }
                    }
                }
            }
            return new Nearest(bestIndex, bestIndex < 0 ? Double.NaN : Math.sqrt(bestSq));
        }
    }

    private static final class IntList {
        int[] values = new int[8];
        int size;

        void add(int value) {
            if (size == values.length) values = Arrays.copyOf(values, values.length * 2);
            values[size++] = value;
        }
    }

    private static final class CameraView {
        final int width;
        final int height;
        final double fx, fy, cx, cy;
        final double[] c2w;

        CameraView(
                int width,
                int height,
                double fx,
                double fy,
                double cx,
                double cy,
                double[] c2w) {
            this.width = width;
            this.height = height;
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
            this.c2w = c2w;
        }

        static CameraView from(JSONObject frame) throws JSONException {
            int width = frame.getInt("w");
            int height = frame.getInt("h");
            double fx = frame.getDouble("fl_x");
            double fy = frame.getDouble("fl_y");
            double cx = frame.getDouble("cx");
            double cy = frame.getDouble("cy");
            double[] c2w = parseRowMatrix(frame.getJSONArray("transform_matrix"));
            if (width <= 0 || height <= 0
                    || !(fx > 0.0) || !(fy > 0.0)
                    || c2w == null) {
                throw new JSONException("invalid camera view");
            }
            return new CameraView(width, height, fx, fy, cx, cy, c2w);
        }

        CameraView scaled(double sx, double sy, int width, int height) {
            return new CameraView(
                    width,
                    height,
                    fx * sx,
                    fy * sy,
                    cx * sx,
                    cy * sy,
                    c2w);
        }
    }

    private static final class CameraPoint {
        final double x, y, depth;

        CameraPoint(double x, double y, double depth) {
            this.x = x;
            this.y = y;
            this.depth = depth;
        }
    }

    private static final class ProjectionResult {
        final int projectedCount;
        final int visiblePointCount;
        final double coverage;
        final BitSet visiblePointIndices;

        ProjectionResult(
                int projectedCount,
                int visiblePointCount,
                double coverage,
                BitSet visiblePointIndices) {
            this.projectedCount = projectedCount;
            this.visiblePointCount = visiblePointCount;
            this.coverage = coverage;
            this.visiblePointIndices = visiblePointIndices;
        }
    }

    private static final class OverlayResult {
        final double edgeErrorPx;
        final JSONObject json;

        OverlayResult(double edgeErrorPx, JSONObject json) {
            this.edgeErrorPx = edgeErrorPx;
            this.json = json;
        }
    }

    private static final class PlyHeader {
        final int headerBytes;
        final int vertexCount;
        final int recordBytes;
        final int xOffset, yOffset, zOffset;
        final int redOffset, greenOffset, blueOffset;
        final int confidenceOffset;

        PlyHeader(
                int headerBytes,
                int vertexCount,
                int recordBytes,
                int xOffset,
                int yOffset,
                int zOffset,
                int redOffset,
                int greenOffset,
                int blueOffset,
                int confidenceOffset) {
            this.headerBytes = headerBytes;
            this.vertexCount = vertexCount;
            this.recordBytes = recordBytes;
            this.xOffset = xOffset;
            this.yOffset = yOffset;
            this.zOffset = zOffset;
            this.redOffset = redOffset;
            this.greenOffset = greenOffset;
            this.blueOffset = blueOffset;
            this.confidenceOffset = confidenceOffset;
        }
    }

    private static PlyHeader readPlyHeader(File file) throws IOException {
        try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
            ByteArrayOutputStream raw = new ByteArrayOutputStream();
            List<String> lines = new ArrayList<>();
            StringBuilder line = new StringBuilder();
            int c;
            while ((c = input.read()) != -1) {
                raw.write(c);
                if (c == '\n') {
                    String value = line.toString().trim();
                    lines.add(value);
                    line.setLength(0);
                    if ("end_header".equals(value)) break;
                } else if (c != '\r') {
                    line.append((char) c);
                }
                if (raw.size() > 64 * 1024) throw new IOException("PLY header too large");
            }

            boolean binary = false;
            boolean inVertex = false;
            int vertices = -1;
            int offset = 0;
            int x = -1, y = -1, z = -1;
            int r = -1, g = -1, b = -1, confidence = -1;
            for (String value : lines) {
                if ("format binary_little_endian 1.0".equals(value)) {
                    binary = true;
                } else if (value.startsWith("element ")) {
                    String[] fields = value.split("\\s+");
                    inVertex = fields.length == 3 && "vertex".equals(fields[1]);
                    if (inVertex) {
                        vertices = Integer.parseInt(fields[2]);
                        offset = 0;
                    }
                } else if (inVertex && value.startsWith("property ")) {
                    String[] fields = value.split("\\s+");
                    if (fields.length != 3 || "list".equals(fields[1])) {
                        throw new IOException("unsupported PLY property");
                    }
                    int size = plyTypeSize(fields[1]);
                    switch (fields[2]) {
                        case "x": x = offset; break;
                        case "y": y = offset; break;
                        case "z": z = offset; break;
                        case "red": r = offset; break;
                        case "green": g = offset; break;
                        case "blue": b = offset; break;
                        case "confidence": confidence = offset; break;
                        default: break;
                    }
                    offset += size;
                }
            }
            if (!binary || vertices < 0 || offset <= 0) {
                throw new IOException("unsupported PLY " + file.getName());
            }
            return new PlyHeader(raw.size(), vertices, offset, x, y, z, r, g, b, confidence);
        }
    }

    private static int plyTypeSize(String type) throws IOException {
        switch (type) {
            case "char":
            case "uchar":
            case "int8":
            case "uint8":
                return 1;
            case "short":
            case "ushort":
            case "int16":
            case "uint16":
                return 2;
            case "int":
            case "uint":
            case "float":
            case "int32":
            case "uint32":
            case "float32":
                return 4;
            case "double":
            case "float64":
            case "int64":
            case "uint64":
                return 8;
            default:
                throw new IOException("unsupported PLY type " + type);
        }
    }

    private static void writeGeometryPrior(File file, List<FusedPoint> points) throws IOException {
        String header =
                "ply\n"
                        + "format binary_little_endian 1.0\n"
                        + "comment Phase 2 fused independent Raw Depth observations\n"
                        + "comment coordinate_system datasetRootAnchor local\n"
                        + "element vertex " + points.size() + "\n"
                        + "property float x\n"
                        + "property float y\n"
                        + "property float z\n"
                        + "property uchar red\n"
                        + "property uchar green\n"
                        + "property uchar blue\n"
                        + "property float confidence\n"
                        + "end_header\n";
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(header.getBytes(StandardCharsets.US_ASCII));
            ByteBuffer buffer = ByteBuffer.allocate(19 * 4096).order(ByteOrder.LITTLE_ENDIAN);
            for (FusedPoint point : points) {
                if (buffer.remaining() < 19) {
                    out.write(buffer.array(), 0, buffer.position());
                    buffer.clear();
                }
                buffer.putFloat(point.x);
                buffer.putFloat(point.y);
                buffer.putFloat(point.z);
                buffer.put((byte) point.r);
                buffer.put((byte) point.g);
                buffer.put((byte) point.b);
                buffer.putFloat(point.confidence);
            }
            if (buffer.position() > 0) out.write(buffer.array(), 0, buffer.position());
        }
    }

    private static List<FusedPoint> sampleFusedPoints(List<FusedPoint> input, int max) {
        if (input.size() <= max) return new ArrayList<>(input);
        List<FusedPoint> out = new ArrayList<>(max);
        double step = input.size() / (double) max;
        for (int i = 0; i < max; i++) {
            out.add(input.get(Math.min(input.size() - 1, (int) Math.floor(i * step))));
        }
        return out;
    }

    private static CameraPoint worldToCamera(double x, double y, double z, double[] c2w) {
        double dx = x - c2w[12];
        double dy = y - c2w[13];
        double dz = z - c2w[14];
        double cx = c2w[0] * dx + c2w[1] * dy + c2w[2] * dz;
        double cy = c2w[4] * dx + c2w[5] * dy + c2w[6] * dz;
        double cz = c2w[8] * dx + c2w[9] * dy + c2w[10] * dz;
        return new CameraPoint(cx, cy, -cz);
    }

    private static double forwardDepthInCamera(SamplePoint p, double[] c2w) {
        return worldToCamera(p.x, p.y, p.z, c2w).depth;
    }

    private static double distanceToCamera(float x, float y, float z, double[] c2w) {
        double dx = x - c2w[12];
        double dy = y - c2w[13];
        double dz = z - c2w[14];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private static double[] parseColumnMajorMatrix(JSONArray values) {
        if (values == null || values.length() != 16) return null;
        double[] out = new double[16];
        try {
            for (int i = 0; i < 16; i++) {
                out[i] = values.getDouble(i);
                if (!Double.isFinite(out[i])) return null;
            }
            return out;
        } catch (JSONException error) {
            return null;
        }
    }

    private static double[] parseRowMatrix(JSONArray rows) throws JSONException {
        if (rows == null || rows.length() != 4) return null;
        double[] out = new double[16];
        for (int row = 0; row < 4; row++) {
            JSONArray values = rows.getJSONArray(row);
            if (values.length() != 4) return null;
            for (int col = 0; col < 4; col++) {
                double value = values.getDouble(col);
                if (!Double.isFinite(value)) return null;
                out[col * 4 + row] = value;
            }
        }
        return out;
    }

    private static Bitmap decodeForOverlay(File source, int sourceW, int sourceH) {
        int sample = 1;
        while (Math.max(sourceW / (sample * 2), sourceH / (sample * 2))
                >= MAX_OVERLAY_LONG_EDGE) {
            sample *= 2;
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = Math.max(1, sample);
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inScaled = false;
        return BitmapFactory.decodeFile(source.getAbsolutePath(), options);
    }

    private static byte[] grayscale(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int[] row = new int[width];
        byte[] gray = new byte[width * height];
        for (int y = 0; y < height; y++) {
            bitmap.getPixels(row, 0, width, 0, y, width, 1);
            int offset = y * width;
            for (int x = 0; x < width; x++) {
                int color = row[x];
                int r = Color.red(color);
                int g = Color.green(color);
                int b = Color.blue(color);
                gray[offset + x] = (byte) ((77 * r + 150 * g + 29 * b) >> 8);
            }
        }
        return gray;
    }

    private static byte[] sobelMagnitude(byte[] gray, int width, int height) {
        byte[] out = new byte[gray.length];
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int i = y * width + x;
                int a = gray[i - width - 1] & 0xff;
                int b = gray[i - width] & 0xff;
                int c = gray[i - width + 1] & 0xff;
                int d = gray[i - 1] & 0xff;
                int f = gray[i + 1] & 0xff;
                int g = gray[i + width - 1] & 0xff;
                int h = gray[i + width] & 0xff;
                int j = gray[i + width + 1] & 0xff;
                int gx = -a + c - 2 * d + 2 * f - g + j;
                int gy = -a - 2 * b - c + g + 2 * h + j;
                int magnitude = Math.min(255, (Math.abs(gx) + Math.abs(gy)) / 4);
                out[i] = (byte) magnitude;
            }
        }
        return out;
    }

    private static int percentileByte(byte[] values, double q) {
        int[] histogram = new int[256];
        int count = 0;
        for (byte value : values) {
            int v = value & 0xff;
            if (v == 0) continue;
            histogram[v]++;
            count++;
        }
        if (count == 0) return 255;
        int target = Math.max(1, (int) Math.ceil(count * q));
        int seen = 0;
        for (int i = 0; i < histogram.length; i++) {
            seen += histogram[i];
            if (seen >= target) return Math.max(16, i);
        }
        return 255;
    }

    private static double nearestGradientEdgeDistance(
            byte[] gradient,
            int threshold,
            int width,
            int height,
            int centerX,
            int centerY,
            int radius) {
        double bestSq = Double.POSITIVE_INFINITY;
        for (int dy = -radius; dy <= radius; dy++) {
            int y = centerY + dy;
            if (y <= 0 || y >= height - 1) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                int x = centerX + dx;
                if (x <= 0 || x >= width - 1) continue;
                if ((gradient[y * width + x] & 0xff) < threshold) continue;
                double sq = dx * (double) dx + dy * (double) dy;
                if (sq < bestSq) bestSq = sq;
            }
        }
        return Double.isFinite(bestSq) ? Math.sqrt(bestSq) : radius + 1.0;
    }

    private static int countFinite(float[] values) {
        int count = 0;
        for (float value : values) if (Float.isFinite(value)) count++;
        return count;
    }

    private static int[] selectOverlayIndices(int count, int required) {
        LinkedHashSet<Integer> values = new LinkedHashSet<>();
        for (int i = 0; i < required; i++) {
            int index = (int) Math.round(i * (count - 1.0) / (required - 1.0));
            values.add(Math.max(0, Math.min(count - 1, index)));
        }
        int cursor = 0;
        while (values.size() < required && cursor < count) values.add(cursor++);
        int[] out = new int[values.size()];
        int i = 0;
        for (int value : values) out[i++] = value;
        return out;
    }

    private static double percentile(List<Double> values, double q) {
        if (values == null || values.isEmpty()) return Double.NaN;
        List<Double> finite = new ArrayList<>(values.size());
        for (double value : values) if (Double.isFinite(value)) finite.add(value);
        if (finite.isEmpty()) return Double.NaN;
        Collections.sort(finite);
        double position = Math.max(0.0, Math.min(1.0, q)) * (finite.size() - 1);
        int lower = (int) Math.floor(position);
        int upper = Math.min(finite.size() - 1, lower + 1);
        double t = position - lower;
        return finite.get(lower) * (1.0 - t) + finite.get(upper) * t;
    }

    private static Object finiteJson(double value) {
        return Double.isFinite(value) ? value : JSONObject.NULL;
    }

    private static double metersToCm(double value) {
        return Double.isFinite(value) ? value * 100.0 : -1.0;
    }

    private static double finiteOrNegative(double value) {
        return Double.isFinite(value) ? value : -1.0;
    }

    private static int fastFloor(float value) {
        int truncated = (int) value;
        return value < truncated ? truncated - 1 : truncated;
    }

    private static long pack(int x, int y, int z) {
        long px = (x + VOXEL_OFFSET) & VOXEL_MASK;
        long py = (y + VOXEL_OFFSET) & VOXEL_MASK;
        long pz = (z + VOXEL_OFFSET) & VOXEL_MASK;
        return (px << (VOXEL_BITS * 2)) | (py << VOXEL_BITS) | pz;
    }

    private static long mix64(long value) {
        value ^= value >>> 33;
        value *= 0xff51afd7ed558ccdL;
        value ^= value >>> 33;
        value *= 0xc4ceb9fe1a85ec53L;
        value ^= value >>> 33;
        return value;
    }

    private static int clampByte(int value) {
        return Math.max(0, Math.min(255, value));
    }

    private static void writeJson(File file, JSONObject json) throws IOException, JSONException {
        try (FileOutputStream out = new FileOutputStream(file, false)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
    }

    private static String readText(File file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) out.append(line).append('\n');
        }
        return out.toString();
    }
}
