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
 * Android-only depth-prior Gaussian generator.
 *
 * <p>This is intentionally small enough to run on a phone without PyTorch/CUDA. It fuses the
 * synchronized ARCore Raw Depth point clouds into a bounded voxel set and serializes the result as
 * a standard 3DGS PLY with SH0 color, opacity, isotropic scale and identity rotation. It provides a
 * real splat model that can be opened by standard 3DGS viewers, while leaving future photometric
 * gradient optimization to a Vulkan backend.</p>
 */
public final class GaussianSplatTrainer {
    private static final String TAG = "GaussianSplatTrainer";
    private static final float VOXEL_SIZE_METERS = 0.015f;
    private static final float MIN_CONFIDENCE = 0.35f;
    private static final int MAX_GAUSSIANS = 160_000;
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

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

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
            return new Result(true,
                    "generated " + gaussianCount + " Gaussians", gaussianCount, outputFile);
        }

        static Result fail(String message) {
            return new Result(false, message, 0, null);
        }
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
            notifyProgress(listener, 2, "Raw Depth PLYを確認しています...");
            List<PlyHeader> headers = new ArrayList<>(files.length);
            long totalPoints = 0L;
            for (File file : files) {
                PlyHeader header = readPlyHeader(file);
                headers.add(header);
                totalPoints += header.vertexCount;
            }
            if (totalPoints == 0L) {
                return Result.fail("Raw Depth PLY contains no points");
            }

            long targetSamples = (long) MAX_GAUSSIANS * 4L;
            int sampleStride = (int) Math.max(1L,
                    (totalPoints + targetSamples - 1L) / targetSamples);

            Map<Long, VoxelAccumulator> voxels = new HashMap<>(MAX_GAUSSIANS * 2);
            long globalPointIndex = 0L;
            long acceptedPoints = 0L;
            for (int fileIndex = 0; fileIndex < files.length; fileIndex++) {
                acceptedPoints += accumulateFile(
                        files[fileIndex], headers.get(fileIndex), voxels,
                        sampleStride, globalPointIndex);
                globalPointIndex += headers.get(fileIndex).vertexCount;
                int percent = 5 + Math.round(65f * (fileIndex + 1) / files.length);
                notifyProgress(listener, percent,
                        "Depth融合中: " + (fileIndex + 1) + "/" + files.length);
            }

            if (voxels.isEmpty()) {
                return Result.fail("no confident Raw Depth points available");
            }

            notifyProgress(listener, 75, "3D Gaussianを生成しています...");
            List<Gaussian> gaussians = buildGaussians(voxels);
            if (gaussians.isEmpty()) {
                return Result.fail("Gaussian generation produced no points");
            }

            File output = new File(datasetDirectory, "splat.ply");
            writeGaussianPly(output, gaussians);
            writeResultJson(datasetDirectory, files.length, totalPoints, acceptedPoints,
                    sampleStride, gaussians.size(), output.getName());
            notifyProgress(listener, 100, "3DGS生成完了");
            DiagnosticLog.i(TAG,
                    "Generated splat.ply gaussians=" + gaussians.size()
                            + " depthFrames=" + files.length
                            + " sampledPoints=" + acceptedPoints
                            + " stride=" + sampleStride);
            return Result.ok(output, gaussians.size());
        } catch (IOException | RuntimeException e) {
            DiagnosticLog.e(TAG, "3DGS generation failed", e);
            return Result.fail("3DGS generation failed: " + e.getClass().getSimpleName());
        }
    }

    private static long accumulateFile(
            File file,
            PlyHeader header,
            Map<Long, VoxelAccumulator> voxels,
            int sampleStride,
            long globalPointOffset) throws IOException {
        if (!header.matchesDepthLayout()) {
            throw new IOException("Unsupported depth PLY layout: " + file.getName());
        }

        long accepted = 0L;
        try (FileInputStream input = new FileInputStream(file);
             FileChannel channel = input.getChannel()) {
            long bodyBytes = file.length() - header.headerBytes;
            if (bodyBytes < (long) header.vertexCount * header.recordBytes) {
                throw new IOException("Truncated PLY body: " + file.getName());
            }
            ByteBuffer body = channel.map(
                    FileChannel.MapMode.READ_ONLY, header.headerBytes,
                    (long) header.vertexCount * header.recordBytes);
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

                if ((globalIndex % sampleStride) != 0L
                        || confidence < MIN_CONFIDENCE
                        || !Float.isFinite(x)
                        || !Float.isFinite(y)
                        || !Float.isFinite(z)) {
                    continue;
                }

                int qx = fastFloor(x / VOXEL_SIZE_METERS);
                int qy = fastFloor(y / VOXEL_SIZE_METERS);
                int qz = fastFloor(z / VOXEL_SIZE_METERS);
                if (!voxelCoordinateSupported(qx)
                        || !voxelCoordinateSupported(qy)
                        || !voxelCoordinateSupported(qz)) {
                    continue;
                }
                long key = packVoxel(qx, qy, qz);
                VoxelAccumulator accumulator = voxels.get(key);
                if (accumulator == null) {
                    if (voxels.size() >= MAX_GAUSSIANS) {
                        continue;
                    }
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
            if (accumulator.count == 0) {
                continue;
            }
            float invCount = 1f / accumulator.count;
            float confidence = clamp(accumulator.confidenceSum * invCount, 0f, 1f);
            float alpha = clamp(0.20f + confidence * 0.75f, 0.20f, 0.95f);
            float scale = VOXEL_SIZE_METERS * (0.55f + (1f - confidence) * 0.35f);
            out.add(new Gaussian(
                    (float) (accumulator.xSum * invCount),
                    (float) (accumulator.ySum * invCount),
                    (float) (accumulator.zSum * invCount),
                    clamp((float) (accumulator.redSum * invCount / 255f), 0f, 1f),
                    clamp((float) (accumulator.greenSum * invCount / 255f), 0f, 1f),
                    clamp((float) (accumulator.blueSum * invCount / 255f), 0f, 1f),
                    logit(alpha),
                    (float) Math.log(Math.max(scale, 1e-5f))));
        }
        return out;
    }

    private static void writeGaussianPly(File file, List<Gaussian> gaussians) throws IOException {
        StringBuilder header = new StringBuilder();
        header.append("ply\n")
                .append("format binary_little_endian 1.0\n")
                .append("comment pointCloudSplating Android depth-prior 3DGS\n")
                .append("comment SH degree 3 layout with SH0 color and zero higher-order terms\n")
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
                buffer.putFloat(0f);
                buffer.putFloat(0f);
                buffer.putFloat(0f);
                buffer.putFloat(rgbToSh(gaussian.red));
                buffer.putFloat(rgbToSh(gaussian.green));
                buffer.putFloat(rgbToSh(gaussian.blue));
                for (int i = 0; i < SH_REST_COUNT; i++) {
                    buffer.putFloat(0f);
                }
                buffer.putFloat(gaussian.opacityLogit);
                buffer.putFloat(gaussian.logScale);
                buffer.putFloat(gaussian.logScale);
                buffer.putFloat(gaussian.logScale);
                buffer.putFloat(1f);
                buffer.putFloat(0f);
                buffer.putFloat(0f);
                buffer.putFloat(0f);
            }
            buffer.flip();
            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }
        }
    }

    private static void writeResultJson(
            File datasetDirectory,
            int depthFrameCount,
            long rawPointCount,
            long acceptedPointCount,
            int sampleStride,
            int gaussianCount,
            String outputName) throws IOException {
        String json = String.format(Locale.US,
                "{\n"
                        + "  \"format_version\": 1,\n"
                        + "  \"status\": \"COMPLETE\",\n"
                        + "  \"backend\": \"android_depth_prior_gaussian_v1\",\n"
                        + "  \"output\": \"%s\",\n"
                        + "  \"depth_frame_count\": %d,\n"
                        + "  \"raw_point_count\": %d,\n"
                        + "  \"accepted_point_count\": %d,\n"
                        + "  \"sample_stride\": %d,\n"
                        + "  \"gaussian_count\": %d,\n"
                        + "  \"voxel_size_m\": %.6f,\n"
                        + "  \"sh_degree\": 3,\n"
                        + "  \"photometric_optimization\": false,\n"
                        + "  \"note\": \"Depth-prior fusion generates a standard 3DGS-compatible PLY on-device. Vulkan photometric optimization can replace this backend later without changing the dataset format.\"\n"
                        + "}\n",
                outputName,
                depthFrameCount,
                rawPointCount,
                acceptedPointCount,
                sampleStride,
                gaussianCount,
                VOXEL_SIZE_METERS);
        try (FileOutputStream out = new FileOutputStream(
                new File(datasetDirectory, "3dgs_result.json"))) {
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
                    if ("end_header".equals(current)) {
                        break;
                    }
                } else if (value != '\r') {
                    line.append((char) value);
                }
                if (raw.size() > 64 * 1024) {
                    throw new IOException("PLY header too large: " + file.getName());
                }
            }
            if (lines.isEmpty() || !"ply".equals(lines.get(0))) {
                throw new IOException("Invalid PLY: " + file.getName());
            }
            boolean binaryLittleEndian = false;
            int vertexCount = -1;
            List<String> properties = new ArrayList<>();
            boolean inVertexElement = false;
            for (String current : lines) {
                if ("format binary_little_endian 1.0".equals(current)) {
                    binaryLittleEndian = true;
                } else if (current.startsWith("element ")) {
                    String[] parts = current.split("\\s+");
                    inVertexElement = parts.length == 3 && "vertex".equals(parts[1]);
                    if (inVertexElement) {
                        vertexCount = Integer.parseInt(parts[2]);
                    }
                } else if (inVertexElement && current.startsWith("property ")) {
                    properties.add(current);
                }
            }
            if (!binaryLittleEndian || vertexCount < 0) {
                throw new IOException("Unsupported PLY encoding: " + file.getName());
            }
            int recordBytes = computeRecordBytes(properties);
            return new PlyHeader(raw.size(), vertexCount, recordBytes, properties);
        }
    }

    private static int computeRecordBytes(List<String> properties) throws IOException {
        int bytes = 0;
        for (String property : properties) {
            String[] parts = property.split("\\s+");
            if (parts.length != 3) {
                throw new IOException("Unsupported PLY property: " + property);
            }
            switch (parts[1]) {
                case "float":
                case "float32":
                    bytes += 4;
                    break;
                case "uchar":
                case "uint8":
                    bytes += 1;
                    break;
                default:
                    throw new IOException("Unsupported PLY type: " + parts[1]);
            }
        }
        return bytes;
    }

    private static boolean voxelCoordinateSupported(int value) {
        return value >= VOXEL_MIN && value <= VOXEL_MAX;
    }

    private static long packVoxel(int x, int y, int z) {
        long px = ((long) x + VOXEL_OFFSET) & VOXEL_MASK;
        long py = ((long) y + VOXEL_OFFSET) & VOXEL_MASK;
        long pz = ((long) z + VOXEL_OFFSET) & VOXEL_MASK;
        return (px << (VOXEL_BITS * 2)) | (py << VOXEL_BITS) | pz;
    }

    private static int fastFloor(float value) {
        int integer = (int) value;
        return value < integer ? integer - 1 : integer;
    }

    private static float rgbToSh(float rgb) {
        return (rgb - 0.5f) / SH_C0;
    }

    private static float logit(float value) {
        float clamped = clamp(value, 1e-4f, 1f - 1e-4f);
        return (float) Math.log(clamped / (1f - clamped));
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void notifyProgress(
            ProgressListener listener, int percent, String message) {
        if (listener != null) {
            listener.onProgress(percent, message);
        }
    }

    private static final class PlyHeader {
        final int headerBytes;
        final int vertexCount;
        final int recordBytes;
        final List<String> properties;

        PlyHeader(int headerBytes, int vertexCount, int recordBytes, List<String> properties) {
            this.headerBytes = headerBytes;
            this.vertexCount = vertexCount;
            this.recordBytes = recordBytes;
            this.properties = properties;
        }

        boolean matchesDepthLayout() {
            if (recordBytes != 19 || properties.size() != 7) {
                return false;
            }
            String[] expected = {
                    "property float x",
                    "property float y",
                    "property float z",
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                    "property float confidence"
            };
            for (int i = 0; i < expected.length; i++) {
                if (!expected[i].equals(properties.get(i))) {
                    return false;
                }
            }
            return true;
        }
    }

    private static final class VoxelAccumulator {
        double xSum;
        double ySum;
        double zSum;
        double redSum;
        double greenSum;
        double blueSum;
        float confidenceSum;
        int count;

        void add(float x, float y, float z, int red, int green, int blue, float confidence) {
            xSum += x;
            ySum += y;
            zSum += z;
            redSum += red;
            greenSum += green;
            blueSum += blue;
            confidenceSum += confidence;
            count++;
        }
    }

    private static final class Gaussian {
        final float x;
        final float y;
        final float z;
        final float red;
        final float green;
        final float blue;
        final float opacityLogit;
        final float logScale;

        Gaussian(
                float x, float y, float z,
                float red, float green, float blue,
                float opacityLogit, float logScale) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.red = red;
            this.green = green;
            this.blue = blue;
            this.opacityLogit = opacityLogit;
            this.logScale = logScale;
        }
    }
}
