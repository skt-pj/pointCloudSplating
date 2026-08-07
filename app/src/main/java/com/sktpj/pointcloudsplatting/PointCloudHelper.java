/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from google-ar/arcore-android-sdk samples/raw_depth_java.
 */
package com.sktpj.pointcloudsplatting;

import android.media.Image;
import android.media.Image.Plane;

import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

/** Static utilities for ARCore Raw Depth point-cloud and RGB-color conversion. */
public final class PointCloudHelper {
    private static final float[] TEXTURE_COORDS = new float[] {
            0.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            1.0f, 1.0f,
    };

    private PointCloudHelper() {}

    public static FloatBuffer convertRawDepthImagesTo3dPointBuffer(
            Image depth,
            Image confidence,
            CameraIntrinsics cameraTextureIntrinsics,
            int pointLimit) {

        Plane depthImagePlane = depth.getPlanes()[0];
        ShortBuffer depthBuffer =
                depthImagePlane.getBuffer().order(ByteOrder.nativeOrder()).asShortBuffer();

        Plane confidenceImagePlane = confidence.getPlanes()[0];
        ByteBuffer confidenceBuffer =
                confidenceImagePlane.getBuffer().order(ByteOrder.nativeOrder());

        int[] intrinsicsDimensions = cameraTextureIntrinsics.getImageDimensions();
        int depthWidth = depth.getWidth();
        int depthHeight = depth.getHeight();

        float fx = cameraTextureIntrinsics.getFocalLength()[0]
                * depthWidth / intrinsicsDimensions[0];
        float fy = cameraTextureIntrinsics.getFocalLength()[1]
                * depthHeight / intrinsicsDimensions[1];
        float cx = cameraTextureIntrinsics.getPrincipalPoint()[0]
                * depthWidth / intrinsicsDimensions[0];
        float cy = cameraTextureIntrinsics.getPrincipalPoint()[1]
                * depthHeight / intrinsicsDimensions[1];

        int step = calculateImageSubsamplingStep(depthWidth, depthHeight, pointLimit);
        int sampledWidth = (depthWidth + step - 1) / step;
        int sampledHeight = (depthHeight + step - 1) / step;
        FloatBuffer points = FloatBuffer.allocate(
                sampledWidth
                        * sampledHeight
                        * PointCloudRenderer.POSITION_FLOATS_PER_POINT);

        for (int y = 0; y < depthHeight; y += step) {
            for (int x = 0; x < depthWidth; x += step) {
                int depthMillimeters = depthBuffer.get(y * depthWidth + x);
                if (depthMillimeters == 0) {
                    continue;
                }

                float depthMeters = depthMillimeters / 1000.0f;
                points.put(depthMeters * (x - cx) / fx);
                points.put(depthMeters * (cy - y) / fy);
                points.put(-depthMeters);

                byte confidencePixelValue = confidenceBuffer.get(
                        y * confidenceImagePlane.getRowStride()
                                + x * confidenceImagePlane.getPixelStride());
                float confidenceNormalized = ((float) (confidencePixelValue & 0xff)) / 255.0f;
                points.put(confidenceNormalized);
            }
        }

        points.rewind();
        return points;
    }

    public static FloatBuffer getImageCoordinatesForFullTexture(Frame frame) {
        FloatBuffer textureCoords = ByteBuffer
                .allocateDirect(TEXTURE_COORDS.length * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(TEXTURE_COORDS);
        textureCoords.position(0);

        FloatBuffer imageCoords = ByteBuffer
                .allocateDirect(TEXTURE_COORDS.length * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        frame.transformCoordinates2d(
                Coordinates2d.TEXTURE_NORMALIZED,
                textureCoords,
                Coordinates2d.IMAGE_PIXELS,
                imageCoords);
        return imageCoords;
    }

    public static FloatBuffer convertImageToColorBuffer(
            Image color,
            Image depth,
            FloatBuffer imageCoords,
            int pointLimit) {

        int depthWidth = depth.getWidth();
        int depthHeight = depth.getHeight();
        int colorWidth = color.getWidth();

        Plane imagePlaneY = color.getPlanes()[0];
        Plane imagePlaneU = color.getPlanes()[1];
        Plane imagePlaneV = color.getPlanes()[2];

        int rowStrideY = imagePlaneY.getRowStride();
        int rowStrideU = imagePlaneU.getRowStride();
        int rowStrideV = imagePlaneV.getRowStride();
        int pixelStrideY = imagePlaneY.getPixelStride();
        int pixelStrideU = imagePlaneU.getPixelStride();
        int pixelStrideV = imagePlaneV.getPixelStride();

        ByteBuffer colorBufferY = imagePlaneY.getBuffer();
        ByteBuffer colorBufferU = imagePlaneU.getBuffer();
        ByteBuffer colorBufferV = imagePlaneV.getBuffer();

        int colorMinY = Math.round(imageCoords.get(1));
        int colorMaxY = Math.round(imageCoords.get(3));
        int colorRegionHeight = colorMaxY - colorMinY;

        Plane depthImagePlane = depth.getPlanes()[0];
        ShortBuffer depthBuffer =
                depthImagePlane.getBuffer().order(ByteOrder.nativeOrder()).asShortBuffer();

        int step = calculateImageSubsamplingStep(depthWidth, depthHeight, pointLimit);
        int sampledWidth = (depthWidth + step - 1) / step;
        int sampledHeight = (depthHeight + step - 1) / step;
        FloatBuffer colors = FloatBuffer.allocate(
                sampledWidth
                        * sampledHeight
                        * PointCloudRenderer.COLOR_FLOATS_PER_POINT);

        float[] rgb = new float[3];

        for (int y = 0; y < depthHeight; y += step) {
            for (int x = 0; x < depthWidth; x += step) {
                if (depthBuffer.get(y * depthWidth + x) == 0) {
                    continue;
                }

                int colorX = x * colorWidth / depthWidth;
                int colorY = colorMinY + y * colorRegionHeight / depthHeight;
                int colorHalfX = colorX / 2;
                int colorHalfY = colorY / 2;

                int channelValueY =
                        colorBufferY.get(colorY * rowStrideY + colorX * pixelStrideY) & 0xff;
                int channelValueU =
                        colorBufferU.get(colorHalfY * rowStrideU + colorHalfX * pixelStrideU) & 0xff;
                int channelValueV =
                        colorBufferV.get(colorHalfY * rowStrideV + colorHalfX * pixelStrideV) & 0xff;

                convertYuvToRgb(channelValueY, channelValueU, channelValueV, rgb);
                colors.put(rgb[0]);
                colors.put(rgb[1]);
                colors.put(rgb[2]);
            }
        }

        colors.rewind();
        return colors;
    }

    private static int calculateImageSubsamplingStep(int imageWidth, int imageHeight, int n) {
        return Math.max(1, (int) Math.ceil(Math.sqrt((float) imageWidth * imageHeight / n)));
    }

    private static void convertYuvToRgb(int yInt, int uInt, int vInt, float[] rgb) {
        float yFloat = yInt / 255.0f;
        float uFloat = uInt * 0.872f / 255.0f - 0.436f;
        float vFloat = vInt * 1.230f / 255.0f - 0.615f;
        rgb[0] = clamp(yFloat + 1.13983f * vFloat);
        rgb[1] = clamp(yFloat - 0.39465f * uFloat - 0.58060f * vFloat);
        rgb[2] = clamp(yFloat + 2.03211f * uFloat);
    }

    private static float clamp(float val) {
        return Math.max(0.0f, Math.min(1.0f, val));
    }
}
