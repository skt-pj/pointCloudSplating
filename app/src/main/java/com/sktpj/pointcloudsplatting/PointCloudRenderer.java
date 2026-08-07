package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

import com.google.ar.core.Camera;

import java.io.IOException;
import java.nio.FloatBuffer;

/** Point cloud renderer derived from google-ar/codelab-raw-depth-api DepthRenderer.java (Apache-2.0). */
public final class PointCloudRenderer {
    private static final String TAG = "PointCloudRenderer";
    private static final int BYTES_PER_POINT = Float.BYTES * DepthPointCloud.FLOATS_PER_POINT;
    private static final int INITIAL_BUFFER_POINTS = 1_000;

    private int vertexBuffer;
    private int vertexBufferSizeBytes;
    private int program;
    private int positionAttribute;
    private int mvpUniform;
    private int pointSizeUniform;
    private int pointCount;

    public void createOnGlThread(Context context) throws IOException {
        int[] buffers = new int[1];
        GLES20.glGenBuffers(1, buffers, 0);
        vertexBuffer = buffers[0];
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
        vertexBufferSizeBytes = INITIAL_BUFFER_POINTS * BYTES_PER_POINT;
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, vertexBufferSizeBytes, null, GLES20.GL_DYNAMIC_DRAW);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);

        int vertex = ShaderUtil.loadShader(TAG, context, GLES20.GL_VERTEX_SHADER,
                "shaders/point_cloud.vert");
        int fragment = ShaderUtil.loadShader(TAG, context, GLES20.GL_FRAGMENT_SHADER,
                "shaders/point_cloud.frag");
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertex);
        GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program);

        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] == 0) {
            throw new IOException("Point shader link failed: " + GLES20.glGetProgramInfoLog(program));
        }

        positionAttribute = GLES20.glGetAttribLocation(program, "a_Position");
        mvpUniform = GLES20.glGetUniformLocation(program, "u_ModelViewProjection");
        pointSizeUniform = GLES20.glGetUniformLocation(program, "u_PointSize");
        ShaderUtil.checkGlError(TAG, "createOnGlThread");
    }

    public void update(FloatBuffer points) {
        points.position(0);
        pointCount = points.remaining() / DepthPointCloud.FLOATS_PER_POINT;
        int bytesNeeded = pointCount * BYTES_PER_POINT;

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
        if (bytesNeeded > vertexBufferSizeBytes) {
            while (bytesNeeded > vertexBufferSizeBytes) {
                vertexBufferSizeBytes *= 2;
            }
            GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, vertexBufferSizeBytes,
                    null, GLES20.GL_DYNAMIC_DRAW);
        }
        if (bytesNeeded > 0) {
            GLES20.glBufferSubData(GLES20.GL_ARRAY_BUFFER, 0, bytesNeeded, points);
        }
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);
        ShaderUtil.checkGlError(TAG, "update");
    }

    public void draw(Camera camera) {
        if (pointCount == 0) {
            return;
        }

        float[] projection = new float[16];
        float[] view = new float[16];
        float[] mvp = new float[16];
        camera.getProjectionMatrix(projection, 0, 0.1f, 100.0f);
        camera.getViewMatrix(view, 0);
        Matrix.multiplyMM(mvp, 0, projection, 0, view, 0);

        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        GLES20.glDepthMask(true);
        GLES20.glUseProgram(program);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vertexBuffer);
        GLES20.glEnableVertexAttribArray(positionAttribute);
        GLES20.glVertexAttribPointer(positionAttribute, 4, GLES20.GL_FLOAT,
                false, BYTES_PER_POINT, 0);
        GLES20.glUniformMatrix4fv(mvpUniform, 1, false, mvp, 0);
        GLES20.glUniform1f(pointSizeUniform, 5.0f);
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, pointCount);
        GLES20.glDisableVertexAttribArray(positionAttribute);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);
        ShaderUtil.checkGlError(TAG, "draw");
    }
}
