package com.sktpj.pointcloudsplatting;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Geometry-prior initializer for phone-side 3DGS.
 *
 * <p>Raw Depth is intentionally a prior rather than the final reconstruction. The initializer
 * keeps substantially more local geometry than the old 15 mm grid so the later RGB optimization
 * is not forced to recover thin structures from an already over-smoothed point set.</p>
 */
public final class GaussianSplatTrainer {
    private static final String TAG = "GaussianSplatTrainer";
    private static final float VOXEL_SIZE_METERS = 0.007f;
    private static final float MIN_CONFIDENCE = 0.30f;
    private static final int MAX_GAUSSIANS = 320_000;
    private static final int MAX_INPUT_SAMPLES = 1_600_000;
    private static final int VOXEL_BITS = 21;
    private static final int VOXEL_OFFSET = 1 << (VOXEL_BITS - 1);
    private static final int VOXEL_MAX = VOXEL_OFFSET - 1;
    private static final int VOXEL_MIN = -VOXEL_OFFSET;
    private static final long VOXEL_MASK = (1L << VOXEL_BITS) - 1L;
    private static final float SH_C0 = 0.28209479177387814f;
    private static final int SH_REST_COUNT = 45;
    private static final int OUTPUT_FLOATS_PER_GAUSSIAN = 62;
    private static final int OUTPUT_BYTES_PER_GAUSSIAN = OUTPUT_FLOATS_PER_GAUSSIAN * Float.BYTES;

    private GaussianSplatTrainer() {}

    public interface ProgressListener { void onProgress(int percent, String message); }

    public static final class Result {
        public final boolean success;
        public final String message;
        public final int gaussianCount;
        public final File outputFile;

        private Result(boolean success, String message, int gaussianCount, File outputFile) {
            this.success = success;
            this.message = message;
            this.gaussianCount = gaussianCount;
            this.outputFile = outputFile;
        }

        static Result ok(File outputFile, int gaussianCount) {
            return new Result(true, "initialized " + gaussianCount + " depth-prior Gaussians",
                    gaussianCount, outputFile);
        }

        static Result fail(String message) { return new Result(false, message, 0, null); }
    }

    public static Result train(File datasetDirectory, ProgressListener listener) {
        if (datasetDirectory == null || !datasetDirectory.isDirectory()) {
            return Result.fail("dataset directory unavailable");
        }
        File[] files = datasetDirectory.listFiles((dir, name) ->
                name.startsWith("frame_") && name.endsWith(".ply"));
        if (files == null || files.length == 0) {
            return Result.fail("Raw Depth PLY is missing");
        }
        Arrays.sort(files, Comparator.comparing(File::getName));
        try {
            notifyProgress(listener, 2, "Raw Depthを確認しています...");
            List<PlyHeader> headers = new ArrayList<>(files.length);
            long totalPoints = 0L;
            for (File file : files) {
                PlyHeader header = readPlyHeader(file);
                headers.add(header);
                totalPoints += header.vertexCount;
            }
            if (totalPoints == 0L) return Result.fail("Raw Depth contains no points");

            int sampleStride = (int) Math.max(1L,
                    (totalPoints + MAX_INPUT_SAMPLES - 1L) / MAX_INPUT_SAMPLES);
            Map<Long, VoxelAccumulator> voxels = new HashMap<>(
                    Math.min(MAX_GAUSSIANS * 2, 700_000));
            long globalPointOffset = 0L;
            long acceptedPoints = 0L;
            for (int fileIndex = 0; fileIndex < files.length; fileIndex++) {
                acceptedPoints += accumulateFile(files[fileIndex], headers.get(fileIndex), voxels,
                        sampleStride, globalPointOffset);
                globalPointOffset += headers.get(fileIndex).vertexCount;
                notifyProgress(listener, 5 + Math.round(65f * (fileIndex + 1) / files.length),
                        "Depth融合中: " + (fileIndex + 1) + "/" + files.length);
            }
            if (voxels.isEmpty()) return Result.fail("no confident Raw Depth points available");

            notifyProgress(listener, 75, "3Dの初期形状を作っています...");
            List<Gaussian> gaussians = buildGaussians(voxels);
            File output = new File(datasetDirectory, "splat.ply");
            writeGaussianPly(output, gaussians);
            writeResultJson(datasetDirectory, files.length, totalPoints, acceptedPoints,
                    sampleStride, gaussians.size(), output.getName());
            notifyProgress(listener, 100, "初期形状を準備しました");
            DiagnosticLog.i(TAG, String.format(Locale.US,
                    "Depth prior ready gaussians=%d depthFrames=%d sampledPoints=%d stride=%d voxel=%.4fm cap=%d",
                    gaussians.size(), files.length, acceptedPoints, sampleStride,
                    VOXEL_SIZE_METERS, MAX_GAUSSIANS));
            return Result.ok(output, gaussians.size());
        } catch (IOException | RuntimeException e) {
            DiagnosticLog.e(TAG, "Depth prior initialization failed", e);
            return Result.fail("Depth prior initialization failed: " + e.getClass().getSimpleName());
        }
    }

    private static long accumulateFile(File file, PlyHeader header,
            Map<Long, VoxelAccumulator> voxels, int sampleStride, long globalPointOffset)
            throws IOException {
        if (!header.matchesDepthLayout()) {
            throw new IOException("Unsupported depth PLY layout: " + file.getName());
        }
        long accepted = 0L;
        try (FileInputStream input = new FileInputStream(file);
             FileChannel channel = input.getChannel()) {
            long bodyBytes = (long) header.vertexCount * header.recordBytes;
            if (file.length() - header.headerBytes < bodyBytes) {
                throw new IOException("Truncated PLY body: " + file.getName());
            }
            ByteBuffer body = channel.map(FileChannel.MapMode.READ_ONLY, header.headerBytes, bodyBytes);
            body.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < header.vertexCount; i++) {
                long globalIndex = globalPointOffset + i;
                float x = body.getFloat();
                float y = body.getFloat();
                float z = body.getFloat();
                int red = body.get() & 0xFF;
                int green = body.get() & 0xFF;
                int blue = body.get() & 0xFF;
                float confidence = body.getFloat();
                if ((globalIndex % sampleStride) != 0L || confidence < MIN_CONFIDENCE
                        || !Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) continue;
                int qx = fastFloor(x / VOXEL_SIZE_METERS);
                int qy = fastFloor(y / VOXEL_SIZE_METERS);
                int qz = fastFloor(z / VOXEL_SIZE_METERS);
                if (!voxelCoordinateSupported(qx) || !voxelCoordinateSupported(qy)
                        || !voxelCoordinateSupported(qz)) continue;
                long key = packVoxel(qx, qy, qz);
                VoxelAccumulator accumulator = voxels.get(key);
                if (accumulator == null) {
                    if (voxels.size() >= MAX_GAUSSIANS) continue;
                    accumulator = new VoxelAccumulator();
                    voxels.put(key, accumulator);
                }
                accumulator.add(x, y, z, red, green, blue, confidence);
                accepted++;
            }
        }
        return accepted;
    }

    private static List<Gaussian> buildGaussians(Map<Long, VoxelAccumulator> voxels) {
        List<Gaussian> out = new ArrayList<>(voxels.size());
        for (VoxelAccumulator accumulator : voxels.values()) {
            if (accumulator.weightSum <= 1e-9) continue;
            double invWeight = 1.0 / accumulator.weightSum;
            float confidence = clamp((float) (accumulator.confidenceWeightedSum * invWeight), 0f, 1f);
            float alpha = clamp(0.12f + confidence * 0.76f, 0.12f, 0.94f);
            float scale = VOXEL_SIZE_METERS * (0.46f + (1f - confidence) * 0.28f);
            out.add(new Gaussian(
                    (float) (accumulator.xWeightedSum * invWeight),
                    (float) (accumulator.yWeightedSum * invWeight),
                    (float) (accumulator.zWeightedSum * invWeight),
                    clamp((float) (accumulator.redWeightedSum * invWeight / 255.0), 0f, 1f),
                    clamp((float) (accumulator.greenWeightedSum * invWeight / 255.0), 0f, 1f),
                    clamp((float) (accumulator.blueWeightedSum * invWeight / 255.0), 0f, 1f),
                    logit(alpha), (float) Math.log(Math.max(scale, 1e-5f))));
        }
        return out;
    }

    private static void writeGaussianPly(File file, List<Gaussian> gaussians) throws IOException {
        StringBuilder header = new StringBuilder();
        header.append("ply\nformat binary_little_endian 1.0\n")
                .append("comment pointCloudSplating dense Raw Depth geometry prior\n")
                .append("comment geometry initialization only; not completed 3DGS\n")
                .append("element vertex ").append(gaussians.size()).append('\n')
                .append("property float x\nproperty float y\nproperty float z\n")
                .append("property float nx\nproperty float ny\nproperty float nz\n")
                .append("property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n");
        for (int i = 0; i < SH_REST_COUNT; i++) header.append("property float f_rest_").append(i).append('\n');
        header.append("property float opacity\nproperty float scale_0\nproperty float scale_1\n")
                .append("property float scale_2\nproperty float rot_0\nproperty float rot_1\n")
                .append("property float rot_2\nproperty float rot_3\nend_header\n");
        try (FileOutputStream output = new FileOutputStream(file);
             FileChannel channel = output.getChannel()) {
            output.write(header.toString().getBytes(StandardCharsets.US_ASCII));
            ByteBuffer buffer = ByteBuffer.allocate(1 << 20).order(ByteOrder.LITTLE_ENDIAN);
            for (Gaussian gaussian : gaussians) {
                if (buffer.remaining() < OUTPUT_BYTES_PER_GAUSSIAN) flush(channel, buffer);
                buffer.putFloat(gaussian.x).putFloat(gaussian.y).putFloat(gaussian.z);
                buffer.putFloat(0f).putFloat(0f).putFloat(0f);
                buffer.putFloat(rgbToSh(gaussian.red)).putFloat(rgbToSh(gaussian.green))
                        .putFloat(rgbToSh(gaussian.blue));
                for (int i = 0; i < SH_REST_COUNT; i++) buffer.putFloat(0f);
                buffer.putFloat(gaussian.opacityLogit);
                buffer.putFloat(gaussian.logScale).putFloat(gaussian.logScale).putFloat(gaussian.logScale);
                buffer.putFloat(1f).putFloat(0f).putFloat(0f).putFloat(0f);
            }
            flush(channel, buffer);
        }
    }

    private static void flush(FileChannel channel, ByteBuffer buffer) throws IOException {
        buffer.flip();
        while (buffer.hasRemaining()) channel.write(buffer);
        buffer.clear();
    }

    private static void writeResultJson(File datasetDirectory, int depthFrameCount,
            long rawPointCount, long acceptedPointCount, int sampleStride, int gaussianCount,
            String outputName) throws IOException {
        String json = String.format(Locale.US,
                "{\n  \"format_version\": 3,\n  \"status\": \"DEPTH_PRIOR_TEMP\",\n"
                        + "  \"backend\": \"android_dense_depth_prior_v3\",\n  \"output\": \"%s\",\n"
                        + "  \"depth_frame_count\": %d,\n  \"raw_point_count\": %d,\n"
                        + "  \"accepted_point_count\": %d,\n  \"sample_stride\": %d,\n"
                        + "  \"gaussian_count\": %d,\n  \"voxel_size_m\": %.6f,\n"
                        + "  \"confidence_min\": %.3f,\n  \"photometric_optimization\": false,\n"
                        + "  \"final_3dgs\": false,\n"
                        + "  \"note\": \"Dense Raw Depth geometry prior only. RGB optimization still required.\"\n}\n",
                outputName, depthFrameCount, rawPointCount, acceptedPointCount, sampleStride,
                gaussianCount, VOXEL_SIZE_METERS, MIN_CONFIDENCE);
        try (FileOutputStream out = new FileOutputStream(new File(datasetDirectory, "3dgs_result.json"))) {
            out.write(json.getBytes(StandardCharsets.UTF_8));
        }
    }

    private static PlyHeader readPlyHeader(File file) throws IOException {
        try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
            ByteArrayOutputStream raw = new ByteArrayOutputStream(1024);
            List<String> lines = new ArrayList<>();
            StringBuilder line = new StringBuilder();
            int value;
            while ((value = input.read()) != -1) {
                raw.write(value);
                if (value == '\n') {
                    String current = line.toString().trim();
                    lines.add(current);
                    line.setLength(0);
                    if ("end_header".equals(current)) break;
                } else if (value != '\r') line.append((char) value);
                if (raw.size() > 64 * 1024) throw new IOException("PLY header too large: " + file.getName());
            }
            if (lines.isEmpty() || !"ply".equals(lines.get(0))) throw new IOException("Invalid PLY: " + file.getName());
            boolean binaryLittleEndian = false;
            int vertexCount = -1;
            List<String> properties = new ArrayList<>();
            boolean inVertexElement = false;
            for (String current : lines) {
                if ("format binary_little_endian 1.0".equals(current)) binaryLittleEndian = true;
                else if (current.startsWith("element ")) {
                    String[] parts = current.split("\\s+");
                    inVertexElement = parts.length == 3 && "vertex".equals(parts[1]);
                    if (inVertexElement) vertexCount = Integer.parseInt(parts[2]);
                } else if (inVertexElement && current.startsWith("property ")) properties.add(current);
            }
            if (!binaryLittleEndian || vertexCount < 0) throw new IOException("Unsupported PLY encoding: " + file.getName());
            return new PlyHeader(raw.size(), vertexCount, properties);
        }
    }

    private static long packVoxel(int x, int y, int z) {
        long bx = (x + VOXEL_OFFSET) & VOXEL_MASK;
        long by = (y + VOXEL_OFFSET) & VOXEL_MASK;
        long bz = (z + VOXEL_OFFSET) & VOXEL_MASK;
        return bx | (by << VOXEL_BITS) | (bz << (VOXEL_BITS * 2));
    }
    private static boolean voxelCoordinateSupported(int value) { return value >= VOXEL_MIN && value <= VOXEL_MAX; }
    private static int fastFloor(float value) { int i = (int) value; return value < i ? i - 1 : i; }
    private static float rgbToSh(float rgb) { return (clamp(rgb, 0f, 1f) - 0.5f) / SH_C0; }
    private static float logit(float alpha) { float v = clamp(alpha, 1e-4f, 1f - 1e-4f); return (float) Math.log(v / (1f - v)); }
    private static float clamp(float value, float min, float max) { return Math.max(min, Math.min(max, value)); }
    private static void notifyProgress(ProgressListener l, int p, String m) { if (l != null) l.onProgress(Math.max(0, Math.min(100, p)), m); }

    private static final class PlyHeader {
        final long headerBytes; final int vertexCount; final List<String> properties; final int recordBytes;
        PlyHeader(long headerBytes, int vertexCount, List<String> properties) {
            this.headerBytes = headerBytes; this.vertexCount = vertexCount; this.properties = properties;
            this.recordBytes = 3 * Float.BYTES + 3 + Float.BYTES;
        }
        boolean matchesDepthLayout() {
            return properties.size() == 7 && propertyIs(0,"float","x") && propertyIs(1,"float","y")
                    && propertyIs(2,"float","z") && propertyIs(3,"uchar","red")
                    && propertyIs(4,"uchar","green") && propertyIs(5,"uchar","blue")
                    && propertyIs(6,"float","confidence");
        }
        private boolean propertyIs(int i, String type, String name) {
            String[] p = properties.get(i).split("\\s+");
            return p.length == 3 && "property".equals(p[0]) && type.equals(p[1]) && name.equals(p[2]);
        }
    }

    private static final class VoxelAccumulator {
        double xWeightedSum, yWeightedSum, zWeightedSum;
        double redWeightedSum, greenWeightedSum, blueWeightedSum;
        double confidenceWeightedSum, weightSum;
        void add(float x, float y, float z, int r, int g, int b, float confidence) {
            double w = Math.max(0.08, confidence * confidence);
            xWeightedSum += x*w; yWeightedSum += y*w; zWeightedSum += z*w;
            redWeightedSum += r*w; greenWeightedSum += g*w; blueWeightedSum += b*w;
            confidenceWeightedSum += confidence*w; weightSum += w;
        }
    }

    private static final class Gaussian {
        final float x,y,z,red,green,blue,opacityLogit,logScale;
        Gaussian(float x,float y,float z,float red,float green,float blue,float opacityLogit,float logScale) {
            this.x=x; this.y=y; this.z=z; this.red=red; this.green=green; this.blue=blue;
            this.opacityLogit=opacityLogit; this.logScale=logScale;
        }
    }
}
