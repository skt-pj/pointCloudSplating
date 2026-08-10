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

import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.NotYetAvailableException;

import java.nio.FloatBuffer;

/** Stores one ARCore Raw Depth frame as camera-local 3D points plus RGB colors. */
public final class DepthData {
    // ScannerActivity first probes the Raw Depth timestamp, then this factory reacquires the depth
    // image to build the point cloud. ARCore may publish a newer Raw Depth image between those two
    // acquisitions. Without an actual-image timestamp guard, that newer image can be accepted here
    // and then accepted again on the next AR frame, creating duplicate canonical Depth observations.
    private static long lastAcceptedRawDepthTimestampNs = Long.MIN_VALUE;

    private final FloatBuffer points;
    private final FloatBuffer colors;
    private final Pose cameraPose;
    private final long timestamp;

    private DepthData(
            FloatBuffer points,
            FloatBuffer colors,
            long timestamp,
            Pose cameraPose) {
        this.points = points;
        this.colors = colors;
        this.timestamp = timestamp;
        this.cameraPose = cameraPose;
    }

    public static DepthData create(Session session, Frame frame) {
        try (Image cameraImage = frame.acquireCameraImage();
             Image depthImage = frame.acquireRawDepthImage16Bits();
             Image confidenceImage = frame.acquireRawDepthConfidenceImage()) {

            final int maxNumberOfPointsToRender = 15_000;
            final long rawDepthTimestampNs = depthImage.getTimestamp();

            CameraIntrinsics intrinsics = frame.getCamera().getTextureIntrinsics();
            FloatBuffer points = PointCloudHelper.convertRawDepthImagesTo3dPointBuffer(
                    depthImage,
                    confidenceImage,
                    intrinsics,
                    maxNumberOfPointsToRender);

            FloatBuffer imageRegionCoordinates =
                    PointCloudHelper.getImageCoordinatesForFullTexture(frame);
            FloatBuffer colors = PointCloudHelper.convertImageToColorBuffer(
                    cameraImage,
                    depthImage,
                    imageRegionCoordinates,
                    maxNumberOfPointsToRender);

            // De-duplicate using the timestamp of the Raw Depth image actually converted above,
            // not the earlier probe timestamp in ScannerActivity. Only genuinely new Raw Depth
            // estimates are allowed to become DepthData / canonical Depth observations.
            synchronized (DepthData.class) {
                if (rawDepthTimestampNs == lastAcceptedRawDepthTimestampNs) {
                    return null;
                }
                lastAcceptedRawDepthTimestampNs = rawDepthTimestampNs;
            }

            // A depth sample only needs the camera pose at acquisition time. Creating an ARCore
            // Anchor here is both unnecessary and unsafe: Session.createAnchor() throws
            // NotTrackingException during short VIO interruptions, which previously stopped all
            // subsequent Raw Depth updates for the scan. Pose is immutable and remains valid as
            // the transform snapshot for this depth image.
            Pose cameraPose = frame.getCamera().getPose();
            return new DepthData(points, colors, rawDepthTimestampNs, cameraPose);
        } catch (NotYetAvailableException e) {
            return null;
        }
    }

    public FloatBuffer getPoints() {
        points.position(0);
        return points;
    }

    public FloatBuffer getColors() {
        colors.position(0);
        return colors;
    }

    public Pose getCameraPose() {
        return cameraPose;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void getModelMatrix(float[] modelMatrix) {
        cameraPose.toMatrix(modelMatrix, 0);
    }
}
