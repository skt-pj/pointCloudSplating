package com.sktpj.pointcloudsplatting;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Phone-side high-quality refinement that actually consumes the saved high-resolution JPEGs,
 * saved camera-to-world poses and intrinsics.
 *
 * <p>This stage is deliberately separate from the future full Vulkan differentiable trainer.
 * It improves the depth prior in three material ways: (1) high-resolution RGB is reprojected onto
 * the Gaussian geometry using the saved cameras, (2) first-order view-dependent SH is fitted by
 * weighted multi-view least squares, and (3) local surface covariance seeds anisotropic scale and
 * rotation. The output is a substantially better 3DGS-compatible preview, but it is not marked as
 * final 3DGS because position/opacity/density-control are not yet optimized through a rasterized
 * L1+SSIM backward pass.</p>
 */
public final class HighQualityGaussianTrainer {
    private static final String TAG = "HighQualityGaussian";
    private static final String PRIOR_FILE = "depth_prior.ply";
    private static final String OUTPUT_FILE = "splat.ply";
    private static final float SH_C0 = 0.28209479177387814f;
    private static final float SH_C1 = 0.4886025119029199f;
    private static final int SH_REST_COUNT = 45;
    private static final int OUTPUT_FLOATS_PER_GAUSSIAN = 62;
    private static final int OUTPUT_BYTES_PER_GAUSSIAN = OUTPUT_FLOATS_PER_GAUSSIAN * Float.BYTES;
    private static final float HASH_CELL_METERS = 0.015f;
    private static final float MIN_SCALE_METERS = 0.0015f;
    private static final float MAX_SCALE_METERS = 0.040f;
    private static final int VISIBILITY_GRID_MAX_WIDTH = 640;
    private static final int ALIGNMENT_SAMPLE_LIMIT = 1200;
    private static final float MIN_OBSERVATION_WEIGHT = 0.08f;

    private HighQualityGaussianTrainer() {}

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        public final boolean success;
        public final String message;
        public final int gaussianCount;
        public final int texturedGaussianCount;
        public final int frameCount;
        public final double photometricRmse;
        public final File outputFile;

        private Result(
                boolean success,
                String message,
                int gaussianCount,
                int texturedGaussianCount,
                int frameCount,
                double photometricRmse,
                File outputFile) {
            this.success = success;
            this.message = message;
            this.gaussianCount = gaussianCount;
            this.texturedGaussianCount = texturedGaussianCount;
            this.frameCount = frameCount;
            this.photometricRmse = photometricRmse;
            this.outputFile = outputFile;
        }

        static Result fail(String message) {
            return new Result(false, message, 0, 0, 0, Double.NaN, null);
        }

        static Result ok(
                File outputFile,
                int gaussianCount,
                int texturedGaussianCount,
                int frameCount,
                double photometricRmse) {
            return new Result(
                    true,
                    "high-resolution RGB refinement ready",
                    gaussianCount,
                    texturedGaussianCount,
                    frameCount,
                    photometricRmse,
                    outputFile);
        }
    }

    public static Result train(File datasetDirectory, ProgressListener listener) {
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return Result.fail("dataset directory unavailable");
        }
        File priorFile = new File(datasetDirectory, PRIOR_FILE);
        File transformsFile = new File(datasetDirectory, "transforms.json");
        if (!priorFile.isFile()) {
            return Result.fail("depth_prior.ply is missing");
        }
        if (!transformsFile.isFile()) {
            return Result.fail("transforms.json is missing");
        }

        try {
            notifyProgress(listener, 2, "Depth priorを読み込んでいます...");
            List<Gaussian> gaussians = readPrior(priorFile);
            if (gaussians.isEmpty()) {
                return Result.fail("depth prior has no Gaussians");
            }

            notifyProgress(listener, 8, "局所形状からanisotropic Gaussianを初期化しています...");
            estimateLocalAnisotropy(gaussians);

            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frameArray = transforms.getJSONArray("frames");
            List<CameraFrame> frames = parseFrames(datasetDirectory, frameArray);
            if (frames.isEmpty()) {
                return Result.fail("no RGB camera frames available");
            }

            int count = gaussians.size();
            // Symmetric 4x4 normal matrix has 10 unique entries.
            float[] ata = new float[count * 10];
            float[] rhs = new float[count * 12];
            float[] yty = new float[count * 3];
            float[] rgbSum = new float[count * 3];
            float[] weightSum = new float[count];
            int[] observations = new int[count];

            for (int frameIndex = 0; frameIndex < frames.size(); frameIndex++) {
                CameraFrame frame = frames.get(frameIndex);
                int basePercent = 12 + Math.round(58f * frameIndex / Math.max(1, frames.size()));
                notifyProgress(listener, basePercent,
                        "高解像度RGBを投影しています: " + (frameIndex + 1) + "/" + frames.size());
                Bitmap bitmap = decodeFullResolution(frame.imageFile);
                if (bitmap == null) {
                    throw new IOException("failed to decode " + frame.imageFile.getName());
                }
                try {
                    frame.prepareForBitmap(bitmap.getWidth(), bitmap.getHeight());
                    Alignment alignment = estimateAlignment(gaussians, frame, bitmap);
                    frame.cx += alignment.dx;
                    frame.cy += alignment.dy;
                    frame.fx *= alignment.focalScale;
                    frame.fy *= alignment.focalScale;
                    DiagnosticLog.i(TAG,
                            String.format(Locale.US,
                                    "RGB alignment frame=%s dx=%.1f dy=%.1f focalScale=%.4f score=%.5f",
                                    frame.imageFile.getName(), alignment.dx, alignment.dy,
                                    alignment.focalScale, alignment.score));
                    accumulateFrameObservations(
                            gaussians, frame, bitmap, ata, rhs, yty,
                            rgbSum, weightSum, observations);
                } finally {
                    bitmap.recycle();
                }
            }

            notifyProgress(listener, 73, "Multi-view SH appearanceを最適化しています...");
            FitStats fit = solveAppearance(
                    gaussians, ata, rhs, yty, rgbSum, weightSum, observations);

            notifyProgress(listener, 88, "高品質Gaussian PLYを書き出しています...");
            File output = new File(datasetDirectory, OUTPUT_FILE);
            writeGaussianPly(output, gaussians);
            writeResult(datasetDirectory, frames, gaussians, fit, output);

            notifyProgress(listener, 100, "高解像度RGB反映完了");
            DiagnosticLog.i(TAG,
                    String.format(Locale.US,
                            "HQ refinement ready gaussians=%d textured=%d frames=%d rmse=%.5f output=%s",
                            gaussians.size(), fit.texturedCount, frames.size(), fit.rmse,
                            output.getAbsolutePath()));
            return Result.ok(
                    output, gaussians.size(), fit.texturedCount, frames.size(), fit.rmse);
        } catch (IOException | JSONException | RuntimeException e) {
            DiagnosticLog.e(TAG, "High-quality Gaussian refinement failed", e);
            return Result.fail("high-quality refinement failed: " + e.getClass().getSimpleName()
                    + (e.getMessage() == null ? "" : " (" + e.getMessage() + ")"));
        }
    }

    private static List<CameraFrame> parseFrames(File dataset, JSONArray frames)
            throws JSONException, IOException {
        List<CameraFrame> out = new ArrayList<>(frames.length());
        for (int i = 0; i < frames.length(); i++) {
            JSONObject json = frames.getJSONObject(i);
            File image = new File(dataset, json.getString("file_path"));
            if (!image.isFile()) {
                throw new IOException("missing RGB frame " + image.getName());
            }
            double[][] c2w = parseMatrix(json.getJSONArray("transform_matrix"));
            CameraFrame frame = new CameraFrame(
                    image,
                    json.getInt("w"),
                    json.getInt("h"),
                    json.getDouble("fl_x"),
                    json.getDouble("fl_y"),
                    json.getDouble("cx"),
                    json.getDouble("cy"),
                    c2w);
            out.add(frame);
        }
        return out;
    }

    private static double[][] parseMatrix(JSONArray rows) throws JSONException, IOException {
        if (rows.length() != 4) {
            throw new IOException("camera transform is not 4x4");
        }
        double[][] matrix = new double[4][4];
        for (int r = 0; r < 4; r++) {
            JSONArray row = rows.getJSONArray(r);
            if (row.length() != 4) {
                throw new IOException("camera transform is not 4x4");
            }
            for (int c = 0; c < 4; c++) {
                matrix[r][c] = row.getDouble(c);
            }
        }
        return matrix;
    }

    private static Bitmap decodeFullResolution(File image) {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inScaled = false;
        return BitmapFactory.decodeFile(image.getAbsolutePath(), options);
    }

    private static void accumulateFrameObservations(
            List<Gaussian> gaussians,
            CameraFrame frame,
            Bitmap bitmap,
            float[] ata,
            float[] rhs,
            float[] yty,
            float[] rgbSum,
            float[] weightSum,
            int[] observations) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int gridWidth = Math.min(VISIBILITY_GRID_MAX_WIDTH, width);
        int gridHeight = Math.max(1, Math.round((float) height * gridWidth / width));
        float[] zBuffer = new float[gridWidth * gridHeight];
        Arrays.fill(zBuffer, Float.POSITIVE_INFINITY);

        Projection projection = new Projection();
        for (Gaussian gaussian : gaussians) {
            if (!frame.project(gaussian.x, gaussian.y, gaussian.z, projection)) {
                continue;
            }
            if (projection.u < 0 || projection.v < 0
                    || projection.u >= width || projection.v >= height) {
                continue;
            }
            int gx = clampInt((int) (projection.u * gridWidth / width), 0, gridWidth - 1);
            int gy = clampInt((int) (projection.v * gridHeight / height), 0, gridHeight - 1);
            int index = gy * gridWidth + gx;
            if (projection.depth < zBuffer[index]) {
                zBuffer[index] = projection.depth;
            }
        }

        for (int i = 0; i < gaussians.size(); i++) {
            Gaussian gaussian = gaussians.get(i);
            if (!frame.project(gaussian.x, gaussian.y, gaussian.z, projection)) {
                continue;
            }
            if (projection.u < 1 || projection.v < 1
                    || projection.u >= width - 1 || projection.v >= height - 1) {
                continue;
            }
            int gx = clampInt((int) (projection.u * gridWidth / width), 0, gridWidth - 1);
            int gy = clampInt((int) (projection.v * gridHeight / height), 0, gridHeight - 1);
            float nearest = zBuffer[gy * gridWidth + gx];
            float visibilityTolerance = Math.max(0.025f, projection.depth * 0.012f);
            if (projection.depth > nearest + visibilityTolerance) {
                continue;
            }

            float dirX = gaussian.x - (float) frame.cameraX;
            float dirY = gaussian.y - (float) frame.cameraY;
            float dirZ = gaussian.z - (float) frame.cameraZ;
            float invDir = inverseLength(dirX, dirY, dirZ);
            if (invDir == 0f) {
                continue;
            }
            dirX *= invDir;
            dirY *= invDir;
            dirZ *= invDir;

            float incidence = Math.abs(
                    gaussian.normalX * (-dirX)
                            + gaussian.normalY * (-dirY)
                            + gaussian.normalZ * (-dirZ));
            float weight = Math.max(MIN_OBSERVATION_WEIGHT,
                    (0.20f + 0.80f * incidence) * gaussian.priorAlpha);

            int color = bitmap.getPixel(
                    clampInt(Math.round(projection.u), 0, width - 1),
                    clampInt(Math.round(projection.v), 0, height - 1));
            float red = ((color >> 16) & 0xFF) / 255f;
            float green = ((color >> 8) & 0xFF) / 255f;
            float blue = (color & 0xFF) / 255f;

            float b0 = SH_C0;
            float b1 = -SH_C1 * dirY;
            float b2 = SH_C1 * dirZ;
            float b3 = -SH_C1 * dirX;
            accumulateNormalMatrix(ata, i * 10, weight, b0, b1, b2, b3);
            int rhsBase = i * 12;
            accumulateRhs(rhs, rhsBase, weight, b0, b1, b2, b3, red - 0.5f);
            accumulateRhs(rhs, rhsBase + 4, weight, b0, b1, b2, b3, green - 0.5f);
            accumulateRhs(rhs, rhsBase + 8, weight, b0, b1, b2, b3, blue - 0.5f);
            int rgbBase = i * 3;
            rgbSum[rgbBase] += weight * red;
            rgbSum[rgbBase + 1] += weight * green;
            rgbSum[rgbBase + 2] += weight * blue;
            yty[rgbBase] += weight * square(red - 0.5f);
            yty[rgbBase + 1] += weight * square(green - 0.5f);
            yty[rgbBase + 2] += weight * square(blue - 0.5f);
            weightSum[i] += weight;
            observations[i]++;
        }
    }

    private static Alignment estimateAlignment(
            List<Gaussian> gaussians, CameraFrame frame, Bitmap bitmap) {
        List<Integer> samples = chooseAlignmentSamples(gaussians, frame, bitmap.getWidth(), bitmap.getHeight());
        if (samples.size() < 32) {
            return Alignment.identity();
        }

        Alignment best = new Alignment(0f, 0f, 1f, Float.POSITIVE_INFINITY);
        float[] focalScales = new float[] {0.985f, 1.0f, 1.015f};
        for (float focalScale : focalScales) {
            for (int dy = -24; dy <= 24; dy += 6) {
                for (int dx = -24; dx <= 24; dx += 6) {
                    float score = alignmentScore(
                            gaussians, samples, frame, bitmap, dx, dy, focalScale);
                    if (score < best.score) {
                        best = new Alignment(dx, dy, focalScale, score);
                    }
                }
            }
        }
        Alignment refined = best;
        for (int dy = Math.round(best.dy) - 5; dy <= Math.round(best.dy) + 5; dy++) {
            for (int dx = Math.round(best.dx) - 5; dx <= Math.round(best.dx) + 5; dx++) {
                float score = alignmentScore(
                        gaussians, samples, frame, bitmap, dx, dy, best.focalScale);
                if (score < refined.score) {
                    refined = new Alignment(dx, dy, best.focalScale, score);
                }
            }
        }
        return refined;
    }

    private static List<Integer> chooseAlignmentSamples(
            List<Gaussian> gaussians, CameraFrame frame, int width, int height) {
        List<Integer> out = new ArrayList<>(Math.min(ALIGNMENT_SAMPLE_LIMIT, gaussians.size()));
        int stride = Math.max(1, gaussians.size() / ALIGNMENT_SAMPLE_LIMIT);
        Projection projection = new Projection();
        for (int i = 0; i < gaussians.size(); i += stride) {
            Gaussian gaussian = gaussians.get(i);
            if (gaussian.priorAlpha < 0.35f) {
                continue;
            }
            if (frame.project(gaussian.x, gaussian.y, gaussian.z, projection)
                    && projection.u >= 32 && projection.u < width - 32
                    && projection.v >= 32 && projection.v < height - 32) {
                out.add(i);
                if (out.size() >= ALIGNMENT_SAMPLE_LIMIT) {
                    break;
                }
            }
        }
        return out;
    }

    private static float alignmentScore(
            List<Gaussian> gaussians,
            List<Integer> sampleIndices,
            CameraFrame frame,
            Bitmap bitmap,
            float dx,
            float dy,
            float focalScale) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        double oldFx = frame.fx;
        double oldFy = frame.fy;
        double oldCx = frame.cx;
        double oldCy = frame.cy;
        frame.fx = oldFx * focalScale;
        frame.fy = oldFy * focalScale;
        frame.cx = oldCx + dx;
        frame.cy = oldCy + dy;
        Projection projection = new Projection();
        double error = 0.0;
        int used = 0;
        for (int index : sampleIndices) {
            Gaussian gaussian = gaussians.get(index);
            if (!frame.project(gaussian.x, gaussian.y, gaussian.z, projection)) {
                continue;
            }
            int u = Math.round(projection.u);
            int v = Math.round(projection.v);
            if (u < 0 || v < 0 || u >= width || v >= height) {
                continue;
            }
            int color = bitmap.getPixel(u, v);
            float r = ((color >> 16) & 0xFF) / 255f;
            float g = ((color >> 8) & 0xFF) / 255f;
            float b = (color & 0xFF) / 255f;
            // Robust L1 matching against the synchronized ARCore CPU color carried by the prior.
            error += Math.min(0.35,
                    Math.abs(r - gaussian.baseRed)
                            + Math.abs(g - gaussian.baseGreen)
                            + Math.abs(b - gaussian.baseBlue));
            used++;
        }
        frame.fx = oldFx;
        frame.fy = oldFy;
        frame.cx = oldCx;
        frame.cy = oldCy;
        return used == 0 ? Float.POSITIVE_INFINITY : (float) (error / used);
    }

    private static FitStats solveAppearance(
            List<Gaussian> gaussians,
            float[] ata,
            float[] rhs,
            float[] yty,
            float[] rgbSum,
            float[] weightSum,
            int[] observations) {
        int textured = 0;
        double residualSse = 0.0;
        double residualWeight = 0.0;
        for (int i = 0; i < gaussians.size(); i++) {
            Gaussian gaussian = gaussians.get(i);
            float weight = weightSum[i];
            if (weight <= 1e-6f || observations[i] == 0) {
                gaussian.setDcFromRgb(gaussian.baseRed, gaussian.baseGreen, gaussian.baseBlue);
                continue;
            }
            textured++;
            int rgbBase = i * 3;
            float meanR = clamp01(rgbSum[rgbBase] / weight);
            float meanG = clamp01(rgbSum[rgbBase + 1] / weight);
            float meanB = clamp01(rgbSum[rgbBase + 2] / weight);

            if (observations[i] < 3) {
                gaussian.setDcFromRgb(meanR, meanG, meanB);
                continue;
            }

            int aBase = i * 10;
            float[] matrix = expandSymmetric4(ata, aBase);
            float ridge = Math.max(1e-4f, weight * 2e-4f);
            matrix[0] += ridge * 0.15f;
            matrix[5] += ridge;
            matrix[10] += ridge;
            matrix[15] += ridge;

            int rhsBase = i * 12;
            float[] red = solve4(matrix, rhs, rhsBase);
            float[] green = solve4(matrix, rhs, rhsBase + 4);
            float[] blue = solve4(matrix, rhs, rhsBase + 8);
            if (red == null || green == null || blue == null) {
                gaussian.setDcFromRgb(meanR, meanG, meanB);
                continue;
            }
            gaussian.setSh(red, green, blue);

            residualSse += channelResidual(ata, aBase, rhs, rhsBase, yty[rgbBase], red);
            residualSse += channelResidual(ata, aBase, rhs, rhsBase + 4, yty[rgbBase + 1], green);
            residualSse += channelResidual(ata, aBase, rhs, rhsBase + 8, yty[rgbBase + 2], blue);
            residualWeight += 3.0 * weight;
        }
        double rmse = residualWeight <= 0.0
                ? Double.NaN : Math.sqrt(Math.max(0.0, residualSse) / residualWeight);
        return new FitStats(textured, rmse);
    }

    private static double channelResidual(
            float[] ata,
            int aBase,
            float[] rhs,
            int rhsBase,
            float yty,
            float[] x) {
        float[] a = expandSymmetric4(ata, aBase);
        double xAx = 0.0;
        double xTb = 0.0;
        for (int r = 0; r < 4; r++) {
            xTb += x[r] * rhs[rhsBase + r];
            for (int c = 0; c < 4; c++) {
                xAx += x[r] * a[r * 4 + c] * x[c];
            }
        }
        return Math.max(0.0, yty - 2.0 * xTb + xAx);
    }

    private static void estimateLocalAnisotropy(List<Gaussian> gaussians) {
        Map<Long, Integer> spatial = new HashMap<>(gaussians.size() * 2);
        for (int i = 0; i < gaussians.size(); i++) {
            Gaussian g = gaussians.get(i);
            g.qx = fastFloor(g.x / HASH_CELL_METERS);
            g.qy = fastFloor(g.y / HASH_CELL_METERS);
            g.qz = fastFloor(g.z / HASH_CELL_METERS);
            spatial.put(packCell(g.qx, g.qy, g.qz), i);
        }

        for (int i = 0; i < gaussians.size(); i++) {
            Gaussian center = gaussians.get(i);
            double cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
            int neighbors = 0;
            for (int dz = -1; dz <= 1; dz++) {
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        Integer index = spatial.get(packCell(
                                center.qx + dx, center.qy + dy, center.qz + dz));
                        if (index == null) {
                            continue;
                        }
                        Gaussian neighbor = gaussians.get(index);
                        double px = neighbor.x - center.x;
                        double py = neighbor.y - center.y;
                        double pz = neighbor.z - center.z;
                        double dist2 = px * px + py * py + pz * pz;
                        if (dist2 <= 1e-10 || dist2 > square(HASH_CELL_METERS * 2.2f)) {
                            continue;
                        }
                        cxx += px * px;
                        cyy += py * py;
                        czz += pz * pz;
                        cxy += px * py;
                        cxz += px * pz;
                        cyz += py * pz;
                        neighbors++;
                    }
                }
            }
            if (neighbors < 4) {
                center.normalX = 0f;
                center.normalY = 0f;
                center.normalZ = 1f;
                continue;
            }
            double inv = 1.0 / neighbors;
            double[][] covariance = new double[][] {
                    {cxx * inv, cxy * inv, cxz * inv},
                    {cxy * inv, cyy * inv, cyz * inv},
                    {cxz * inv, cyz * inv, czz * inv}
            };
            Eigen3 eigen = jacobiEigen(covariance);
            int[] order = sortEigenDescending(eigen.values);
            double[] axis0 = column(eigen.vectors, order[0]);
            double[] axis1 = column(eigen.vectors, order[1]);
            double[] axis2 = column(eigen.vectors, order[2]);
            if (determinant(axis0, axis1, axis2) < 0.0) {
                axis2[0] = -axis2[0];
                axis2[1] = -axis2[1];
                axis2[2] = -axis2[2];
            }

            float tangent0 = clampScale((float) (Math.sqrt(Math.max(1e-9, eigen.values[order[0]])) * 1.15));
            float tangent1 = clampScale((float) (Math.sqrt(Math.max(1e-9, eigen.values[order[1]])) * 1.15));
            float normal = clamp(
                    (float) (Math.sqrt(Math.max(1e-10, eigen.values[order[2]])) * 0.55),
                    MIN_SCALE_METERS,
                    Math.max(MIN_SCALE_METERS, Math.min(tangent0, tangent1) * 0.38f));
            center.logScaleX = (float) Math.log(tangent0);
            center.logScaleY = (float) Math.log(tangent1);
            center.logScaleZ = (float) Math.log(normal);
            center.normalX = (float) axis2[0];
            center.normalY = (float) axis2[1];
            center.normalZ = (float) axis2[2];
            float[] quaternion = quaternionFromAxes(axis0, axis1, axis2);
            center.rotW = quaternion[0];
            center.rotX = quaternion[1];
            center.rotY = quaternion[2];
            center.rotZ = quaternion[3];
        }
    }

    private static List<Gaussian> readPrior(File file) throws IOException {
        try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
            PlyHeader header = readFloatPlyHeader(input);
            int xIndex = requireProperty(header, "x");
            int yIndex = requireProperty(header, "y");
            int zIndex = requireProperty(header, "z");
            int rIndex = requireProperty(header, "f_dc_0");
            int gIndex = requireProperty(header, "f_dc_1");
            int bIndex = requireProperty(header, "f_dc_2");
            int opacityIndex = requireProperty(header, "opacity");
            int sxIndex = requireProperty(header, "scale_0");
            int syIndex = requireProperty(header, "scale_1");
            int szIndex = requireProperty(header, "scale_2");

            long bytes = (long) header.vertexCount * header.properties.size() * Float.BYTES;
            if (file.length() - header.bodyOffset < bytes) {
                throw new IOException("truncated depth_prior.ply");
            }
            MappedByteBuffer body = input.getChannel().map(
                    FileChannel.MapMode.READ_ONLY, header.bodyOffset, bytes);
            body.order(ByteOrder.LITTLE_ENDIAN);
            float[] values = new float[header.properties.size()];
            List<Gaussian> gaussians = new ArrayList<>(header.vertexCount);
            for (int i = 0; i < header.vertexCount; i++) {
                for (int p = 0; p < values.length; p++) {
                    values[p] = body.getFloat();
                }
                float x = values[xIndex];
                float y = values[yIndex];
                float z = values[zIndex];
                if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) {
                    continue;
                }
                Gaussian gaussian = new Gaussian();
                gaussian.x = x;
                gaussian.y = y;
                gaussian.z = z;
                gaussian.baseRed = clamp01(0.5f + SH_C0 * values[rIndex]);
                gaussian.baseGreen = clamp01(0.5f + SH_C0 * values[gIndex]);
                gaussian.baseBlue = clamp01(0.5f + SH_C0 * values[bIndex]);
                gaussian.opacityLogit = values[opacityIndex];
                gaussian.priorAlpha = clamp(sigmoid(gaussian.opacityLogit), 0.05f, 0.99f);
                gaussian.logScaleX = values[sxIndex];
                gaussian.logScaleY = values[syIndex];
                gaussian.logScaleZ = values[szIndex];
                gaussian.rotW = 1f;
                gaussian.setDcFromRgb(gaussian.baseRed, gaussian.baseGreen, gaussian.baseBlue);
                gaussians.add(gaussian);
            }
            return gaussians;
        }
    }

    private static void writeGaussianPly(File file, List<Gaussian> gaussians) throws IOException {
        StringBuilder header = new StringBuilder();
        header.append("ply\n")
                .append("format binary_little_endian 1.0\n")
                .append("comment pointCloudSplating high-resolution multi-view Gaussian refinement\n")
                .append("comment Depth geometry + saved JPEG/Pose/intrinsics + anisotropic covariance + SH1 fit\n")
                .append("element vertex ").append(gaussians.size()).append('\n')
                .append("property float x\n")
                .append("property float y\n")
                .append("property float z\n")
                .append("property float nx\n")
                .append("property float ny\n")
                .append("property float nz\n")
                .append("property float f_dc_0\n")
                .append("property float f_dc_1\n")
                .append("property float f_dc_2\n");
        for (int i = 0; i < SH_REST_COUNT; i++) {
            header.append("property float f_rest_").append(i).append('\n');
        }
        header.append("property float opacity\n")
                .append("property float scale_0\n")
                .append("property float scale_1\n")
                .append("property float scale_2\n")
                .append("property float rot_0\n")
                .append("property float rot_1\n")
                .append("property float rot_2\n")
                .append("property float rot_3\n")
                .append("end_header\n");

        try (FileOutputStream output = new FileOutputStream(file);
             FileChannel channel = output.getChannel()) {
            output.write(header.toString().getBytes(StandardCharsets.US_ASCII));
            ByteBuffer buffer = ByteBuffer.allocate(1 << 20).order(ByteOrder.LITTLE_ENDIAN);
            for (Gaussian gaussian : gaussians) {
                if (buffer.remaining() < OUTPUT_BYTES_PER_GAUSSIAN) {
                    buffer.flip();
                    while (buffer.hasRemaining()) {
                        channel.write(buffer);
                    }
                    buffer.clear();
                }
                buffer.putFloat(gaussian.x);
                buffer.putFloat(gaussian.y);
                buffer.putFloat(gaussian.z);
                buffer.putFloat(gaussian.normalX);
                buffer.putFloat(gaussian.normalY);
                buffer.putFloat(gaussian.normalZ);
                buffer.putFloat(gaussian.shR[0]);
                buffer.putFloat(gaussian.shG[0]);
                buffer.putFloat(gaussian.shB[0]);
                for (int rest = 0; rest < SH_REST_COUNT; rest++) {
                    float value = 0f;
                    if (rest < 3) {
                        value = gaussian.shR[rest + 1];
                    } else if (rest >= 15 && rest < 18) {
                        value = gaussian.shG[rest - 15 + 1];
                    } else if (rest >= 30 && rest < 33) {
                        value = gaussian.shB[rest - 30 + 1];
                    }
                    buffer.putFloat(value);
                }
                buffer.putFloat(gaussian.opacityLogit);
                buffer.putFloat(gaussian.logScaleX);
                buffer.putFloat(gaussian.logScaleY);
                buffer.putFloat(gaussian.logScaleZ);
                buffer.putFloat(gaussian.rotW);
                buffer.putFloat(gaussian.rotX);
                buffer.putFloat(gaussian.rotY);
                buffer.putFloat(gaussian.rotZ);
            }
            buffer.flip();
            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }
        }
    }

    private static void writeResult(
            File dataset,
            List<CameraFrame> frames,
            List<Gaussian> gaussians,
            FitStats fit,
            File output) throws IOException, JSONException {
        JSONObject result = new JSONObject();
        result.put("format_version", 3);
        result.put("status", "HQ_RGB_REFINED");
        result.put("backend", "android_highres_multiview_gaussian_v1");
        result.put("output", output.getName());
        result.put("gaussian_count", gaussians.size());
        result.put("textured_gaussian_count", fit.texturedCount);
        result.put("rgb_frame_count", frames.size());
        result.put("rgb_role", "high_resolution_photometric_observation");
        result.put("camera_source", "saved_ARCore_pose_and_intrinsics");
        result.put("geometry_source", "ARCore_Raw_Depth_prior");
        result.put("anisotropic_initialization", true);
        result.put("sh_degree_fitted", 1);
        result.put("high_order_sh_zero", true);
        result.put("per_frame_projection_alignment", true);
        result.put("photometric_optimization", true);
        result.put("photometric_fit", "weighted_multiview_SH1_least_squares");
        result.put("photometric_rmse", Double.isFinite(fit.rmse) ? fit.rmse : JSONObject.NULL);
        result.put("rasterized_image_loss", false);
        result.put("l1_ssim_backward", false);
        result.put("density_control", false);
        result.put("final_3dgs", false);
        result.put("note",
                "High-resolution JPEG/Pose/intrinsics are now used for real multi-view appearance refinement. "
                        + "Full final 3DGS remains reserved for differentiable rasterized L1+SSIM optimization "
                        + "with position/opacity/scale/rotation/SH updates and density control.");
        writeJson(new File(dataset, "3dgs_result.json"), result);
    }

    private static PlyHeader readFloatPlyHeader(RandomAccessFile input) throws IOException {
        PlyHeader header = new PlyHeader();
        String first = input.readLine();
        if (!"ply".equals(first)) {
            throw new IOException("not a PLY file");
        }
        boolean inVertex = false;
        String line;
        while ((line = input.readLine()) != null) {
            String trimmed = line.trim();
            if (trimmed.startsWith("format ")) {
                header.binaryLittleEndian = trimmed.startsWith("format binary_little_endian");
            } else if (trimmed.startsWith("element ")) {
                String[] tokens = trimmed.split("\\s+");
                inVertex = tokens.length >= 3 && "vertex".equals(tokens[1]);
                if (inVertex) {
                    header.vertexCount = Integer.parseInt(tokens[2]);
                }
            } else if (inVertex && trimmed.startsWith("property ")) {
                String[] tokens = trimmed.split("\\s+");
                if (tokens.length != 3
                        || !("float".equals(tokens[1]) || "float32".equals(tokens[1]))) {
                    throw new IOException("float Gaussian properties only");
                }
                header.properties.add(tokens[2]);
            } else if ("end_header".equals(trimmed)) {
                header.bodyOffset = input.getFilePointer();
                break;
            }
        }
        if (!header.binaryLittleEndian || header.vertexCount <= 0 || header.bodyOffset <= 0) {
            throw new IOException("invalid Gaussian PLY header");
        }
        return header;
    }

    private static int requireProperty(PlyHeader header, String name) throws IOException {
        int index = header.properties.indexOf(name);
        if (index < 0) {
            throw new IOException("missing PLY property " + name);
        }
        return index;
    }

    private static void accumulateNormalMatrix(
            float[] ata, int offset, float w, float b0, float b1, float b2, float b3) {
        ata[offset] += w * b0 * b0;
        ata[offset + 1] += w * b0 * b1;
        ata[offset + 2] += w * b0 * b2;
        ata[offset + 3] += w * b0 * b3;
        ata[offset + 4] += w * b1 * b1;
        ata[offset + 5] += w * b1 * b2;
        ata[offset + 6] += w * b1 * b3;
        ata[offset + 7] += w * b2 * b2;
        ata[offset + 8] += w * b2 * b3;
        ata[offset + 9] += w * b3 * b3;
    }

    private static void accumulateRhs(
            float[] rhs, int offset, float w, float b0, float b1, float b2, float b3, float target) {
        rhs[offset] += w * b0 * target;
        rhs[offset + 1] += w * b1 * target;
        rhs[offset + 2] += w * b2 * target;
        rhs[offset + 3] += w * b3 * target;
    }

    private static float[] expandSymmetric4(float[] packed, int offset) {
        float[] a = new float[16];
        a[0] = packed[offset];
        a[1] = a[4] = packed[offset + 1];
        a[2] = a[8] = packed[offset + 2];
        a[3] = a[12] = packed[offset + 3];
        a[5] = packed[offset + 4];
        a[6] = a[9] = packed[offset + 5];
        a[7] = a[13] = packed[offset + 6];
        a[10] = packed[offset + 7];
        a[11] = a[14] = packed[offset + 8];
        a[15] = packed[offset + 9];
        return a;
    }

    private static float[] solve4(float[] matrix, float[] rhs, int rhsOffset) {
        double[][] augmented = new double[4][5];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                augmented[r][c] = matrix[r * 4 + c];
            }
            augmented[r][4] = rhs[rhsOffset + r];
        }
        for (int col = 0; col < 4; col++) {
            int pivot = col;
            double best = Math.abs(augmented[col][col]);
            for (int r = col + 1; r < 4; r++) {
                double value = Math.abs(augmented[r][col]);
                if (value > best) {
                    best = value;
                    pivot = r;
                }
            }
            if (best < 1e-10) {
                return null;
            }
            if (pivot != col) {
                double[] temp = augmented[pivot];
                augmented[pivot] = augmented[col];
                augmented[col] = temp;
            }
            double inv = 1.0 / augmented[col][col];
            for (int c = col; c < 5; c++) {
                augmented[col][c] *= inv;
            }
            for (int r = 0; r < 4; r++) {
                if (r == col) continue;
                double factor = augmented[r][col];
                for (int c = col; c < 5; c++) {
                    augmented[r][c] -= factor * augmented[col][c];
                }
            }
        }
        float[] out = new float[4];
        for (int i = 0; i < 4; i++) {
            out[i] = clamp((float) augmented[i][4], -4f, 4f);
        }
        return out;
    }

    private static Eigen3 jacobiEigen(double[][] input) {
        double[][] a = new double[3][3];
        double[][] v = new double[][] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int r = 0; r < 3; r++) {
            System.arraycopy(input[r], 0, a[r], 0, 3);
        }
        for (int iteration = 0; iteration < 14; iteration++) {
            int p = 0;
            int q = 1;
            double max = Math.abs(a[0][1]);
            if (Math.abs(a[0][2]) > max) {
                p = 0; q = 2; max = Math.abs(a[0][2]);
            }
            if (Math.abs(a[1][2]) > max) {
                p = 1; q = 2; max = Math.abs(a[1][2]);
            }
            if (max < 1e-12) {
                break;
            }
            double phi = 0.5 * Math.atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
            double c = Math.cos(phi);
            double s = Math.sin(phi);
            for (int k = 0; k < 3; k++) {
                double apk = a[p][k];
                double aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; k++) {
                double akp = a[k][p];
                double akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; k++) {
                double vkp = v[k][p];
                double vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
        return new Eigen3(new double[] {a[0][0], a[1][1], a[2][2]}, v);
    }

    private static int[] sortEigenDescending(double[] values) {
        int[] order = new int[] {0, 1, 2};
        for (int i = 0; i < order.length; i++) {
            for (int j = i + 1; j < order.length; j++) {
                if (values[order[j]] > values[order[i]]) {
                    int temp = order[i];
                    order[i] = order[j];
                    order[j] = temp;
                }
            }
        }
        return order;
    }

    private static double[] column(double[][] matrix, int col) {
        return new double[] {matrix[0][col], matrix[1][col], matrix[2][col]};
    }

    private static double determinant(double[] a, double[] b, double[] c) {
        return a[0] * (b[1] * c[2] - b[2] * c[1])
                - b[0] * (a[1] * c[2] - a[2] * c[1])
                + c[0] * (a[1] * b[2] - a[2] * b[1]);
    }

    private static float[] quaternionFromAxes(double[] x, double[] y, double[] z) {
        double m00 = x[0], m01 = y[0], m02 = z[0];
        double m10 = x[1], m11 = y[1], m12 = z[1];
        double m20 = x[2], m21 = y[2], m22 = z[2];
        double qw, qx, qy, qz;
        double trace = m00 + m11 + m22;
        if (trace > 0.0) {
            double s = Math.sqrt(trace + 1.0) * 2.0;
            qw = 0.25 * s;
            qx = (m21 - m12) / s;
            qy = (m02 - m20) / s;
            qz = (m10 - m01) / s;
        } else if (m00 > m11 && m00 > m22) {
            double s = Math.sqrt(1.0 + m00 - m11 - m22) * 2.0;
            qw = (m21 - m12) / s;
            qx = 0.25 * s;
            qy = (m01 + m10) / s;
            qz = (m02 + m20) / s;
        } else if (m11 > m22) {
            double s = Math.sqrt(1.0 + m11 - m00 - m22) * 2.0;
            qw = (m02 - m20) / s;
            qx = (m01 + m10) / s;
            qy = 0.25 * s;
            qz = (m12 + m21) / s;
        } else {
            double s = Math.sqrt(1.0 + m22 - m00 - m11) * 2.0;
            qw = (m10 - m01) / s;
            qx = (m02 + m20) / s;
            qy = (m12 + m21) / s;
            qz = 0.25 * s;
        }
        double length = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (length < 1e-9) {
            return new float[] {1f, 0f, 0f, 0f};
        }
        return new float[] {
                (float) (qw / length),
                (float) (qx / length),
                (float) (qy / length),
                (float) (qz / length)
        };
    }

    private static long packCell(int x, int y, int z) {
        long bx = ((long) x - Integer.MIN_VALUE) & 0x1FFFFFL;
        long by = ((long) y - Integer.MIN_VALUE) & 0x1FFFFFL;
        long bz = ((long) z - Integer.MIN_VALUE) & 0x1FFFFFL;
        return bx | (by << 21) | (bz << 42);
    }

    private static int fastFloor(float value) {
        int i = (int) value;
        return value < i ? i - 1 : i;
    }

    private static float clampScale(float value) {
        return clamp(value, 0.004f, MAX_SCALE_METERS);
    }

    private static float inverseLength(float x, float y, float z) {
        float length2 = x * x + y * y + z * z;
        return length2 <= 1e-12f ? 0f : (float) (1.0 / Math.sqrt(length2));
    }

    private static float sigmoid(float x) {
        if (x >= 0f) {
            float e = (float) Math.exp(-x);
            return 1f / (1f + e);
        }
        float e = (float) Math.exp(x);
        return e / (1f + e);
    }

    private static float square(float x) {
        return x * x;
    }

    private static float clamp01(float value) {
        return clamp(value, 0f, 1f);
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void notifyProgress(ProgressListener listener, int percent, String message) {
        if (listener != null) {
            listener.onProgress(Math.max(0, Math.min(100, percent)), message);
        }
    }

    private static String readText(File file) throws IOException {
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(file), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line).append('\n');
            }
        }
        return out.toString();
    }

    private static void writeJson(File file, JSONObject json) throws IOException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));
        } catch (JSONException e) {
            throw new IOException("failed to serialize result JSON", e);
        }
    }

    private static final class Gaussian {
        float x, y, z;
        float baseRed, baseGreen, baseBlue;
        float priorAlpha;
        float opacityLogit;
        float logScaleX, logScaleY, logScaleZ;
        float rotW = 1f, rotX, rotY, rotZ;
        float normalX, normalY, normalZ = 1f;
        int qx, qy, qz;
        final float[] shR = new float[4];
        final float[] shG = new float[4];
        final float[] shB = new float[4];

        void setDcFromRgb(float r, float g, float b) {
            Arrays.fill(shR, 0f);
            Arrays.fill(shG, 0f);
            Arrays.fill(shB, 0f);
            shR[0] = (clamp01(r) - 0.5f) / SH_C0;
            shG[0] = (clamp01(g) - 0.5f) / SH_C0;
            shB[0] = (clamp01(b) - 0.5f) / SH_C0;
        }

        void setSh(float[] r, float[] g, float[] b) {
            for (int i = 0; i < 4; i++) {
                shR[i] = clamp(r[i], -4f, 4f);
                shG[i] = clamp(g[i], -4f, 4f);
                shB[i] = clamp(b[i], -4f, 4f);
            }
        }
    }

    private static final class CameraFrame {
        final File imageFile;
        final int metadataWidth;
        final int metadataHeight;
        double fx, fy, cx, cy;
        final double[][] c2w;
        final double cameraX, cameraY, cameraZ;

        CameraFrame(
                File imageFile,
                int metadataWidth,
                int metadataHeight,
                double fx,
                double fy,
                double cx,
                double cy,
                double[][] c2w) {
            this.imageFile = imageFile;
            this.metadataWidth = metadataWidth;
            this.metadataHeight = metadataHeight;
            this.fx = fx;
            this.fy = fy;
            this.cx = cx;
            this.cy = cy;
            this.c2w = c2w;
            this.cameraX = c2w[0][3];
            this.cameraY = c2w[1][3];
            this.cameraZ = c2w[2][3];
        }

        void prepareForBitmap(int width, int height) {
            if (metadataWidth <= 0 || metadataHeight <= 0) {
                return;
            }
            double sx = width / (double) metadataWidth;
            double sy = height / (double) metadataHeight;
            fx *= sx;
            fy *= sy;
            cx *= sx;
            cy *= sy;
        }

        boolean project(float wx, float wy, float wz, Projection out) {
            double dx = wx - cameraX;
            double dy = wy - cameraY;
            double dz = wz - cameraZ;
            // c2w rotation is orthonormal, so world->camera is R^T.
            double camX = c2w[0][0] * dx + c2w[1][0] * dy + c2w[2][0] * dz;
            double camY = c2w[0][1] * dx + c2w[1][1] * dy + c2w[2][1] * dz;
            double camZ = c2w[0][2] * dx + c2w[1][2] * dy + c2w[2][2] * dz;
            double depth = -camZ;
            if (!(depth > 0.02) || !Double.isFinite(depth)) {
                return false;
            }
            out.depth = (float) depth;
            out.u = (float) (fx * (camX / depth) + cx);
            // OpenGL camera +Y is up; image pixel +Y is down.
            out.v = (float) (cy - fy * (camY / depth));
            return Float.isFinite(out.u) && Float.isFinite(out.v);
        }
    }

    private static final class Projection {
        float u, v, depth;
    }

    private static final class Alignment {
        final float dx, dy, focalScale, score;

        Alignment(float dx, float dy, float focalScale, float score) {
            this.dx = dx;
            this.dy = dy;
            this.focalScale = focalScale;
            this.score = score;
        }

        static Alignment identity() {
            return new Alignment(0f, 0f, 1f, Float.NaN);
        }
    }

    private static final class FitStats {
        final int texturedCount;
        final double rmse;

        FitStats(int texturedCount, double rmse) {
            this.texturedCount = texturedCount;
            this.rmse = rmse;
        }
    }

    private static final class Eigen3 {
        final double[] values;
        final double[][] vectors;

        Eigen3(double[] values, double[][] vectors) {
            this.values = values;
            this.vectors = vectors;
        }
    }

    private static final class PlyHeader {
        boolean binaryLittleEndian;
        int vertexCount;
        long bodyOffset;
        final List<String> properties = new ArrayList<>();
    }
}
