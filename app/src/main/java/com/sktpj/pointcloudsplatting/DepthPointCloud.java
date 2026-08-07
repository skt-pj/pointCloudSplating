package com.sktpj.pointcloudsplatting;

import android.media.Image;
import android.opengl.Matrix;

import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.exceptions.NotYetAvailableException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Raw Depth image to world-space point cloud conversion.
 * Derived from google-ar/codelab-raw-depth-api DepthData.java (Apache-2.0).
 */
public final class DepthPointCloud {
    public static final int FLOATS_PER_POINT = 4;
    private static final int MAX_POINTS = 20_000;
    private static final float MIN_CONFIDENCE = 0.30f;
    private static final float MAX_DEPTH_METERS = 1.50f;

    private DepthPointCloud() {}

    public static final class PointCloudFrame {
        public final FloatBuffer points;
        public final int pointCount;

        private PointCloudFrame(FloatBuffer points, int pointCount) {
            this.points = points;
            this.pointCount = pointCount;
        }
    }

    public static PointCloudFrame create(Frame frame, Pose cameraPose) {
        try (Image depthImage = frame.acquireRawDepthImage16Bits();
             Image confidenceImage = frame.acquireRawDepthConfidenceImage()) {
            return convert(depthImage, confidenceImage, frame.getCamera().getTextureIntrinsics(), cameraPose);
        } catch (NotYetAvailableException e) {
            return null;
        }
    }

    private static PointCloudFrame convert(
            Image depth,
            Image confidence,
            CameraIntrinsics textureIntrinsics,
            Pose cameraPose) {

        int depthWidth = depth.getWidth();
        int depthHeight = depth.getHeight();
        int[] intrinsicsDimensions = textureIntrinsics.getImageDimensions();

        float fx = textureIntrinsics.getFocalLength()[0] * depthWidth / intrinsicsDimensions[0];
        float fy = textureIntrinsics.getFocalLength()[1] * depthHeight / intrinsicsDimensions[1];
        float cx = textureIntrinsics.getPrincipalPoint()[0] * depthWidth / intrinsicsDimensions[0];
        float cy = textureIntrinsics.getPrincipalPoint()[1] * depthHeight / intrinsicsDimensions[1];

        int step = Math.max(1, (int) Math.ceil(
                Math.sqrt((double) depthWidth * depthHeight / MAX_POINTS)));
        int sampledWidth = (depthWidth + step - 1) / step;
        int sampledHeight = (depthHeight + step - 1) / step;

        FloatBuffer points = ByteBuffer
                .allocateDirect(sampledWidth * sampledHeight * FLOATS_PER_POINT * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        Image.Plane depthPlane = depth.getPlanes()[0];
        ByteBuffer depthBuffer = depthPlane.getBuffer().duplicate().order(ByteOrder.nativeOrder());
        Image.Plane confidencePlane = confidence.getPlanes()[0];
        ByteBuffer confidenceBuffer = confidencePlane.getBuffer().duplicate();

        float[] modelMatrix = new float[16];
        cameraPose.toMatrix(modelMatrix, 0);
        float[] pointCamera = new float[4];
        float[] pointWorld = new float[4];
        int pointCount = 0;

        for (int y = 0; y < depthHeight; y += step) {
            for (int x = 0; x < depthWidth; x += step) {
                int depthByteIndex = y * depthPlane.getRowStride() + x * depthPlane.getPixelStride();
                if (depthByteIndex < 0 || depthByteIndex + Short.BYTES > depthBuffer.limit()) {
                    continue;
                }

                int depthMillimeters = Short.toUnsignedInt(depthBuffer.getShort(depthByteIndex));
                if (depthMillimeters == 0) {
                    continue;
                }

                float depthMeters = depthMillimeters / 1000.0f;
                int confidenceIndex = y * confidencePlane.getRowStride()
                        + x * confidencePlane.getPixelStride();
                if (confidenceIndex < 0 || confidenceIndex >= confidenceBuffer.limit()) {
                    continue;
                }

                float confidenceNormalized = (confidenceBuffer.get(confidenceIndex) & 0xFF) / 255.0f;
                if (confidenceNormalized < MIN_CONFIDENCE || depthMeters > MAX_DEPTH_METERS) {
                    continue;
                }

                pointCamera[0] = depthMeters * (x - cx) / fx;
                pointCamera[1] = depthMeters * (cy - y) / fy;
                pointCamera[2] = -depthMeters;
                pointCamera[3] = 1.0f;

                Matrix.multiplyMV(pointWorld, 0, modelMatrix, 0, pointCamera, 0);
                points.put(pointWorld[0]);
                points.put(pointWorld[1]);
                points.put(pointWorld[2]);
                points.put(confidenceNormalized);
                pointCount++;
            }
        }

        points.flip();
        return new PointCloudFrame(points, pointCount);
    }
}
