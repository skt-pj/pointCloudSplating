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

import com.google.ar.core.Anchor;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.Session;
import com.google.ar.core.exceptions.NotYetAvailableException;

import java.nio.FloatBuffer;

/** Stores one ARCore Raw Depth frame as camera-local 3D points plus RGB colors. */
public final class DepthData {
    private final FloatBuffer points;
    private final FloatBuffer colors;
    private final Anchor anchor;
    private final long timestamp;

    private DepthData(
            FloatBuffer points,
            FloatBuffer colors,
            long timestamp,
            Anchor cameraPoseAnchor) {
        this.points = points;
        this.colors = colors;
        this.timestamp = timestamp;
        this.anchor = cameraPoseAnchor;
    }

    public static DepthData create(Session session, Frame frame) {
        try (Image cameraImage = frame.acquireCameraImage();
             Image depthImage = frame.acquireRawDepthImage16Bits();
             Image confidenceImage = frame.acquireRawDepthConfidenceImage()) {

            final int maxNumberOfPointsToRender = 15_000;

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

            Anchor cameraPoseAnchor = session.createAnchor(frame.getCamera().getPose());
            return new DepthData(points, colors, depthImage.getTimestamp(), cameraPoseAnchor);
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

    public Anchor getAnchor() {
        return anchor;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void getModelMatrix(float[] modelMatrix) {
        anchor.getPose().toMatrix(modelMatrix, 0);
    }
}
