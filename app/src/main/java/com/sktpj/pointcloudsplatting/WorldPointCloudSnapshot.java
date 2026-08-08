package com.sktpj.pointcloudsplatting;

import com.google.ar.core.Pose;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;

/** Immutable copy of one Raw Depth frame transformed into the session root-anchor frame. */
public final class WorldPointCloudSnapshot {
    private static final int POSITION_FLOATS_PER_POINT = 4;
    private static final int COLOR_FLOATS_PER_POINT = 3;
    private static final int BINARY_BYTES_PER_POINT = 3 * Float.BYTES + 3 + Float.BYTES;

    private final long timestampNs;
    private final float[] xyz;
    private final byte[] rgb;
    private final float[] confidence;

    private WorldPointCloudSnapshot(
            long timestampNs, float[] xyz, byte[] rgb, float[] confidence) {
        this.timestampNs = timestampNs;
        this.xyz = xyz;
        this.rgb = rgb;
        this.confidence = confidence;
    }

    /** Copies the DepthData now, while its ARCore anchor is still active. */
    public static WorldPointCloudSnapshot from(DepthData depth, Pose rootPose) {
        FloatBuffer points = depth.getPoints().duplicate();
        FloatBuffer colors = depth.getColors().duplicate();
        points.position(0);
        colors.position(0);

        int pointCount = Math.min(
                points.remaining() / POSITION_FLOATS_PER_POINT,
                colors.remaining() / COLOR_FLOATS_PER_POINT);

        float[] model = new float[16];
        Pose rootFromDepthCamera = rootPose.inverse().compose(depth.getAnchor().getPose());
        rootFromDepthCamera.toMatrix(model, 0);

        float[] xyz = new float[pointCount * 3];
        byte[] rgb = new byte[pointCount * 3];
        float[] confidence = new float[pointCount];

        for (int i = 0; i < pointCount; i++) {
            float x = points.get();
            float y = points.get();
            float z = points.get();
            float c = points.get();

            // Android/ARCore matrices are OpenGL column-major. Convert the camera-local depth point
            // into the same world coordinate system used by the saved camera poses.
            xyz[i * 3] = model[0] * x + model[4] * y + model[8] * z + model[12];
            xyz[i * 3 + 1] = model[1] * x + model[5] * y + model[9] * z + model[13];
            xyz[i * 3 + 2] = model[2] * x + model[6] * y + model[10] * z + model[14];
            confidence[i] = c;

            rgb[i * 3] = toByte(colors.get());
            rgb[i * 3 + 1] = toByte(colors.get());
            rgb[i * 3 + 2] = toByte(colors.get());
        }

        return new WorldPointCloudSnapshot(depth.getTimestamp(), xyz, rgb, confidence);
    }

    public long getTimestampNs() {
        return timestampNs;
    }

    public int getPointCount() {
        return confidence.length;
    }

    /** Writes a standard binary little-endian PLY: XYZ, RGB and Raw Depth confidence. */
    public void writePly(File file) throws IOException {
        String header =
                "ply\n"
                        + "format binary_little_endian 1.0\n"
                        + "comment ARCore root-anchor local coordinates; camera looks along local -Z\n"
                        + "comment raw_depth_timestamp_ns " + timestampNs + "\n"
                        + "element vertex " + getPointCount() + "\n"
                        + "property float x\n"
                        + "property float y\n"
                        + "property float z\n"
                        + "property uchar red\n"
                        + "property uchar green\n"
                        + "property uchar blue\n"
                        + "property float confidence\n"
                        + "end_header\n";

        ByteBuffer binary = ByteBuffer
                .allocate(getPointCount() * BINARY_BYTES_PER_POINT)
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < getPointCount(); i++) {
            binary.putFloat(xyz[i * 3]);
            binary.putFloat(xyz[i * 3 + 1]);
            binary.putFloat(xyz[i * 3 + 2]);
            binary.put(rgb[i * 3]);
            binary.put(rgb[i * 3 + 1]);
            binary.put(rgb[i * 3 + 2]);
            binary.putFloat(confidence[i]);
        }

        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(header.getBytes(StandardCharsets.US_ASCII));
            out.write(binary.array());
        }
    }

    private static byte toByte(float value) {
        float clamped = Math.max(0f, Math.min(1f, value));
        return (byte) Math.round(clamped * 255f);
    }
}
