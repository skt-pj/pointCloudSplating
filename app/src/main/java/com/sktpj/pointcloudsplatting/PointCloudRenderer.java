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
 * Adapted from google-ar/arcore-android-sdk samples/raw_depth_java Renderer.java.
 */
package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;

/** Renders accumulated Raw Depth frames as RGB-colored 3D points over the live camera view. */
public final class PointCloudRenderer {
    private static final String TAG = "PointCloudRenderer";

    public static final int POSITION_FLOATS_PER_POINT = 4;
    public static final int COLOR_FLOATS_PER_POINT = 3;

    private static final int POSITION_BYTES_PER_POINT = Float.BYTES * POSITION_FLOATS_PER_POINT;
    private static final int COLOR_BYTES_PER_POINT = Float.BYTES * COLOR_FLOATS_PER_POINT;
    private static final int INITIAL_BUFFER_POINTS = 1_000;
    private static final int MAX_FRAMES_STORED = 60;

    private static final Object INSTANCES_LOCK = new Object();
    private static final ArrayList<WeakReference<PointCloudRenderer>> INSTANCES = new ArrayList<>();

    private final ArrayList<DepthData> depthFrames = new ArrayList<>();

    private int positionBuffer;
    private int positionBufferSize;
    private int colorBuffer;
    private int colorBufferSize;

    private int program;
    private int positionAttribute;
    private int colorAttribute;
    private int modelViewProjectionUniform;
    private int pointSizeUniform;
    private int confidenceThresholdUniform;

    private float minConfidence = 0.1f;

    public PointCloudRenderer() {
        synchronized (INSTANCES_LOCK) {
            cleanupDeadInstancesLocked();
            INSTANCES.add(new WeakReference<>(this));
        }
    }

    /** Drops the in-memory rolling Raw Depth preview before the heavy native 3DGS allocation. */
    public static int clearAllFramesForModelProcessing() {
        int cleared = 0;
        synchronized (INSTANCES_LOCK) {
            Iterator<WeakReference<PointCloudRenderer>> iterator = INSTANCES.iterator();
            while (iterator.hasNext()) {
                PointCloudRenderer renderer = iterator.next().get();
                if (renderer == null) {
                    iterator.remove();
                    continue;
                }
                cleared += renderer.clearFrames();
            }
        }
        DiagnosticLog.i(TAG, "Cleared rolling depth preview frames=" + cleared + " before 3DGS");
        return cleared;
    }

    private static void cleanupDeadInstancesLocked() {
        Iterator<WeakReference<PointCloudRenderer>> iterator = INSTANCES.iterator();
        while (iterator.hasNext()) {
            if (iterator.next().get() == null) iterator.remove();
        }
    }

    private synchronized int clearFrames() {
        int count = depthFrames.size();
        depthFrames.clear();
        return count;
    }

    public void createOnGlThread(Context context) throws IOException {
        int[] buffers = new int[2];
        GLES20.glGenBuffers(2, buffers, 0);
        positionBuffer = buffers[0];
        colorBuffer = buffers[1];

        positionBufferSize = INITIAL_BUFFER_POINTS * POSITION_BYTES_PER_POINT;
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, positionBuffer);
        GLES20.glBufferData(
                GLES20.GL_ARRAY_BUFFER, positionBufferSize, null, GLES20.GL_DYNAMIC_DRAW);

        colorBufferSize = INITIAL_BUFFER_POINTS * COLOR_BYTES_PER_POINT;
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, colorBuffer);
        GLES20.glBufferData(
                GLES20.GL_ARRAY_BUFFER, colorBufferSize, null, GLES20.GL_DYNAMIC_DRAW);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);

        int vertexShader = ShaderUtil.loadShader(
                TAG, context, GLES20.GL_VERTEX_SHADER, "shaders/point_cloud.vert");
        int fragmentShader = ShaderUtil.loadShader(
                TAG, context, GLES20.GL_FRAGMENT_SHADER, "shaders/point_cloud.frag");

        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);

        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] == 0) {
            throw new IOException("Point shader link failed: " + GLES20.glGetProgramInfoLog(program));
        }

        positionAttribute = GLES20.glGetAttribLocation(program, "a_Position");
        colorAttribute = GLES20.glGetAttribLocation(program, "a_Color");
        modelViewProjectionUniform = GLES20.glGetUniformLocation(program, "u_ModelViewProjection");
        pointSizeUniform = GLES20.glGetUniformLocation(program, "u_PointSize");
        confidenceThresholdUniform = GLES20.glGetUniformLocation(program, "u_ConfidenceThreshold");
        ShaderUtil.checkGlError(TAG, "createOnGlThread");
    }

    public synchronized void update(DepthData depth) {
        depthFrames.add(depth);
        while (depthFrames.size() > MAX_FRAMES_STORED) {
            depthFrames.remove(0);
        }
    }

    public synchronized int getStoredFrameCount() {
        return depthFrames.size();
    }

    public synchronized void draw(float[] viewMatrix, float[] projectionMatrix) {
        if (depthFrames.isEmpty()) {
            return;
        }

        float[] modelMatrix = new float[16];
        float[] modelView = new float[16];
        float[] modelViewProjection = new float[16];

        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        GLES20.glDepthMask(true);
        GLES20.glUseProgram(program);

        for (DepthData depthFrame : depthFrames) {
            int numPoints = depthFrame.getPoints().remaining() / POSITION_FLOATS_PER_POINT;
            if (numPoints == 0) {
                continue;
            }

            int positionBytesNeeded = numPoints * POSITION_BYTES_PER_POINT;
            while (positionBytesNeeded > positionBufferSize) {
                positionBufferSize *= 2;
            }
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, positionBuffer);
            GLES20.glBufferData(
                    GLES20.GL_ARRAY_BUFFER, positionBufferSize, null, GLES20.GL_DYNAMIC_DRAW);
            GLES20.glBufferSubData(
                    GLES20.GL_ARRAY_BUFFER, 0, positionBytesNeeded, depthFrame.getPoints());

            int colorBytesNeeded = numPoints * COLOR_BYTES_PER_POINT;
            while (colorBytesNeeded > colorBufferSize) {
                colorBufferSize *= 2;
            }
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, colorBuffer);
            GLES20.glBufferData(
                    GLES20.GL_ARRAY_BUFFER, colorBufferSize, null, GLES20.GL_DYNAMIC_DRAW);
            GLES20.glBufferSubData(
                    GLES20.GL_ARRAY_BUFFER, 0, colorBytesNeeded, depthFrame.getColors());

            depthFrame.getModelMatrix(modelMatrix);
            Matrix.multiplyMM(modelView, 0, viewMatrix, 0, modelMatrix, 0);
            Matrix.multiplyMM(modelViewProjection, 0, projectionMatrix, 0, modelView, 0);

            GLES20.glEnableVertexAttribArray(positionAttribute);
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, positionBuffer);
            GLES20.glVertexAttribPointer(
                    positionAttribute,
                    POSITION_FLOATS_PER_POINT,
                    GLES20.GL_FLOAT,
                    false,
                    POSITION_BYTES_PER_POINT,
                    0);

            GLES20.glEnableVertexAttribArray(colorAttribute);
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, colorBuffer);
            GLES20.glVertexAttribPointer(
                    colorAttribute,
                    COLOR_FLOATS_PER_POINT,
                    GLES20.GL_FLOAT,
                    false,
                    COLOR_BYTES_PER_POINT,
                    0);

            GLES20.glUniformMatrix4fv(
                    modelViewProjectionUniform, 1, false, modelViewProjection, 0);
            GLES20.glUniform1f(pointSizeUniform, 5.0f);
            GLES20.glUniform1f(confidenceThresholdUniform, minConfidence);
            GLES20.glDrawArrays(GLES20.GL_POINTS, 0, numPoints);

            GLES20.glDisableVertexAttribArray(positionAttribute);
            GLES20.glDisableVertexAttribArray(colorAttribute);
        }

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);
        ShaderUtil.checkGlError(TAG, "draw");
    }
}
