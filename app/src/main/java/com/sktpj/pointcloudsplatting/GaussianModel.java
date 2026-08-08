package com.sktpj.pointcloudsplatting;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

final class GaussianModel {
    final int gaussianCount;
    final float[] positions;
    final float[] dc;
    final float[] sh1;
    final float[] alpha;
    final float[] scales;
    final float[] rotations;
    final float sourceRadius;

    GaussianModel(
            int gaussianCount,
            float[] positions,
            float[] dc,
            float[] sh1,
            float[] alpha,
            float[] scales,
            float[] rotations,
            float sourceRadius) {
        this.gaussianCount = gaussianCount;
        this.positions = positions;
        this.dc = dc;
        this.sh1 = sh1;
        this.alpha = alpha;
        this.scales = scales;
        this.rotations = rotations;
        this.sourceRadius = sourceRadius;
    }
}

final class GaussianPlyModelReader {
    private static final String TAG = "GaussianViewer";
    private static final int ROBUST_SAMPLE_LIMIT = 8192;

    static GaussianModel read(File file) throws IOException {
        try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
            Header header = readHeader(input);
            if (!header.binaryLittleEndian || header.vertexCount <= 0) {
                throw new IOException("invalid Gaussian PLY");
            }

            int xIndex = require(header, "x");
            int yIndex = require(header, "y");
            int zIndex = require(header, "z");
            int dcR = require(header, "f_dc_0");
            int dcG = require(header, "f_dc_1");
            int dcB = require(header, "f_dc_2");
            int opacityIndex = require(header, "opacity");
            int sxIndex = require(header, "scale_0");
            int syIndex = require(header, "scale_1");
            int szIndex = require(header, "scale_2");
            int rwIndex = require(header, "rot_0");
            int rxIndex = require(header, "rot_1");
            int ryIndex = require(header, "rot_2");
            int rzIndex = require(header, "rot_3");

            int r1 = header.indexOf("f_rest_0");
            int r2 = header.indexOf("f_rest_1");
            int r3 = header.indexOf("f_rest_2");
            int g1 = header.indexOf("f_rest_15");
            int g2 = header.indexOf("f_rest_16");
            int g3 = header.indexOf("f_rest_17");
            int b1 = header.indexOf("f_rest_30");
            int b2 = header.indexOf("f_rest_31");
            int b3 = header.indexOf("f_rest_32");

            long bodyBytes = (long) header.vertexCount * header.properties.size() * Float.BYTES;
            if (file.length() - header.bodyOffset < bodyBytes) {
                throw new IOException("truncated splat.ply");
            }

            MappedByteBuffer body = input.getChannel().map(
                    FileChannel.MapMode.READ_ONLY, header.bodyOffset, bodyBytes);
            body.order(ByteOrder.LITTLE_ENDIAN);

            float[] xyz = new float[header.vertexCount * 3];
            float[] dc = new float[header.vertexCount * 3];
            float[] sh1 = new float[header.vertexCount * 9];
            float[] alpha = new float[header.vertexCount];
            float[] worldScales = new float[header.vertexCount * 3];
            float[] rotations = new float[header.vertexCount * 4];
            boolean[] valid = new boolean[header.vertexCount];
            float[] values = new float[header.properties.size()];

            int validCount = 0;
            int sh1Count = 0;
            int anisotropicCount = 0;
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

                valid[i] = true;
                validCount++;
                int p = i * 3;
                xyz[p] = x;
                xyz[p + 1] = y;
                xyz[p + 2] = z;
                dc[p] = finiteOrZero(values[dcR]);
                dc[p + 1] = finiteOrZero(values[dcG]);
                dc[p + 2] = finiteOrZero(values[dcB]);

                int s = i * 9;
                sh1[s] = valueAt(values, r1);
                sh1[s + 1] = valueAt(values, r2);
                sh1[s + 2] = valueAt(values, r3);
                sh1[s + 3] = valueAt(values, g1);
                sh1[s + 4] = valueAt(values, g2);
                sh1[s + 5] = valueAt(values, g3);
                sh1[s + 6] = valueAt(values, b1);
                sh1[s + 7] = valueAt(values, b2);
                sh1[s + 8] = valueAt(values, b3);
                float shMagnitude = 0f;
                for (int k = 0; k < 9; k++) {
                    shMagnitude += Math.abs(sh1[s + k]);
                }
                if (shMagnitude > 1e-7f) {
                    sh1Count++;
                }

                alpha[i] = clamp(sigmoid(values[opacityIndex]), 0.001f, 0.999f);
                float sx = positiveScale(values[sxIndex]);
                float sy = positiveScale(values[syIndex]);
                float sz = positiveScale(values[szIndex]);
                worldScales[p] = sx;
                worldScales[p + 1] = sy;
                worldScales[p + 2] = sz;
                float maxScale = Math.max(sx, Math.max(sy, sz));
                float minScale = Math.min(sx, Math.min(sy, sz));
                if (maxScale > minScale * 1.10f) {
                    anisotropicCount++;
                }

                int r = i * 4;
                float qw = finiteOrZero(values[rwIndex]);
                float qx = finiteOrZero(values[rxIndex]);
                float qy = finiteOrZero(values[ryIndex]);
                float qz = finiteOrZero(values[rzIndex]);
                float qLen = (float) Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
                if (!(qLen > 1e-8f) || !Float.isFinite(qLen)) {
                    qw = 1f;
                    qx = qy = qz = 0f;
                } else {
                    float inv = 1f / qLen;
                    qw *= inv;
                    qx *= inv;
                    qy *= inv;
                    qz *= inv;
                }
                rotations[r] = qw;
                rotations[r + 1] = qx;
                rotations[r + 2] = qy;
                rotations[r + 3] = qz;
            }

            if (validCount == 0) {
                throw new IOException("PLY has no finite positions");
            }

            float[] centerRadius = robustCenterRadius(xyz, valid, validCount);
            float centerX = centerRadius[0];
            float centerY = centerRadius[1];
            float centerZ = centerRadius[2];
            float radius = centerRadius[3];

            float[] positionsOut = new float[validCount * 3];
            float[] dcOut = new float[validCount * 3];
            float[] shOut = new float[validCount * 9];
            float[] alphaOut = new float[validCount];
            float[] scalesOut = new float[validCount * 3];
            float[] rotationsOut = new float[validCount * 4];

            int outIndex = 0;
            for (int i = 0; i < header.vertexCount; i++) {
                if (!valid[i]) {
                    continue;
                }
                int srcP = i * 3;
                int dstP = outIndex * 3;
                int srcS = i * 9;
                int dstS = outIndex * 9;
                int srcR = i * 4;
                int dstR = outIndex * 4;

                positionsOut[dstP] = (xyz[srcP] - centerX) / radius;
                positionsOut[dstP + 1] = (xyz[srcP + 1] - centerY) / radius;
                positionsOut[dstP + 2] = (xyz[srcP + 2] - centerZ) / radius;

                dcOut[dstP] = dc[srcP];
                dcOut[dstP + 1] = dc[srcP + 1];
                dcOut[dstP + 2] = dc[srcP + 2];
                System.arraycopy(sh1, srcS, shOut, dstS, 9);
                alphaOut[outIndex] = alpha[i];

                scalesOut[dstP] = clamp(worldScales[srcP] / radius, 1e-6f, 0.35f);
                scalesOut[dstP + 1] = clamp(worldScales[srcP + 1] / radius, 1e-6f, 0.35f);
                scalesOut[dstP + 2] = clamp(worldScales[srcP + 2] / radius, 1e-6f, 0.35f);
                System.arraycopy(rotations, srcR, rotationsOut, dstR, 4);
                outIndex++;
            }

            DiagnosticLog.i(
                    TAG,
                    String.format(
                            Locale.US,
                            "Loaded model gaussians=%d/%d robustCenter=(%.3f,%.3f,%.3f) "
                                    + "radius99=%.3f anisotropic=%d sh1=%d "
                                    + "viewer=ANISOTROPIC_COVARIANCE_SORTED_ES3 appearance=JPEG_SH1",
                            validCount, header.vertexCount, centerX, centerY, centerZ, radius,
                            anisotropicCount, sh1Count));

            return new GaussianModel(
                    validCount, positionsOut, dcOut, shOut, alphaOut, scalesOut, rotationsOut, radius);
        }
    }

    private static float[] robustCenterRadius(float[] xyz, boolean[] valid, int validCount) {
        int stride = Math.max(1, validCount / ROBUST_SAMPLE_LIMIT);
        int capacity = Math.min(validCount, (validCount + stride - 1) / stride + 1);
        float[] xs = new float[capacity];
        float[] ys = new float[capacity];
        float[] zs = new float[capacity];
        int count = 0;
        int seen = 0;
        for (int i = 0; i < valid.length && count < capacity; i++) {
            if (!valid[i]) {
                continue;
            }
            if (seen % stride == 0) {
                xs[count] = xyz[i * 3];
                ys[count] = xyz[i * 3 + 1];
                zs[count] = xyz[i * 3 + 2];
                count++;
            }
            seen++;
        }

        float[] sx = Arrays.copyOf(xs, count);
        float[] sy = Arrays.copyOf(ys, count);
        float[] sz = Arrays.copyOf(zs, count);
        Arrays.sort(sx);
        Arrays.sort(sy);
        Arrays.sort(sz);
        float cx = sx[count / 2];
        float cy = sy[count / 2];
        float cz = sz[count / 2];

        float[] distances = new float[count];
        for (int i = 0; i < count; i++) {
            float dx = xs[i] - cx;
            float dy = ys[i] - cy;
            float dz = zs[i] - cz;
            distances[i] = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        }
        Arrays.sort(distances);
        int index = Math.min(count - 1, Math.max(0, Math.round((count - 1) * 0.99f)));
        float radius = Math.max(0.05f, distances[index] * 1.12f);
        if (!Float.isFinite(radius) || radius <= 0f) {
            radius = 1f;
        }
        return new float[] {cx, cy, cz, radius};
    }

    private static Header readHeader(RandomAccessFile input) throws IOException {
        Header header = new Header();
        if (!"ply".equals(input.readLine())) {
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
                    throw new IOException("viewer supports float Gaussian properties only");
                }
                header.properties.add(tokens[2]);
            } else if ("end_header".equals(trimmed)) {
                header.bodyOffset = input.getFilePointer();
                break;
            }
        }
        if (header.bodyOffset <= 0 || header.properties.isEmpty()) {
            throw new IOException("invalid PLY header");
        }
        return header;
    }

    private static int require(Header header, String name) throws IOException {
        int index = header.indexOf(name);
        if (index < 0) {
            throw new IOException("missing PLY property " + name);
        }
        return index;
    }

    private static float valueAt(float[] values, int index) {
        return index >= 0 ? finiteOrZero(values[index]) : 0f;
    }

    private static float finiteOrZero(float value) {
        return Float.isFinite(value) ? value : 0f;
    }

    private static float positiveScale(float logScale) {
        if (!Float.isFinite(logScale)) {
            return 0.004f;
        }
        float value = (float) Math.exp(Math.max(-20f, Math.min(2f, logScale)));
        return Float.isFinite(value) && value > 0f ? value : 0.004f;
    }

    private static float sigmoid(float x) {
        if (!Float.isFinite(x)) {
            return 0.5f;
        }
        if (x >= 0f) {
            float z = (float) Math.exp(-x);
            return 1f / (1f + z);
        }
        float z = (float) Math.exp(x);
        return z / (1f + z);
    }

    private static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }

    private static final class Header {
        boolean binaryLittleEndian;
        int vertexCount;
        long bodyOffset;
        final List<String> properties = new ArrayList<>();

        int indexOf(String name) {
            return properties.indexOf(name);
        }
    }
}
