package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;

import com.google.ar.core.Coordinates2d;
import com.google.ar.core.Frame;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/** Draws the ARCore camera texture full-screen behind the accumulated point cloud. */
public final class CameraBackgroundRenderer {
    private static final String TAG = "CameraBackground";

    private static final float[] QUAD_COORDS = {
            -1.0f, -1.0f,
             1.0f, -1.0f,
            -1.0f,  1.0f,
             1.0f,  1.0f
    };

    private final FloatBuffer quadCoords = floatBuffer(QUAD_COORDS);
    private final FloatBuffer textureCoords = floatBuffer(new float[8]);

    private int textureId = -1;
    private int program;
    private int positionAttribute;
    private int textureCoordinateAttribute;
    private int textureUniform;

    public void createOnGlThread(Context context) throws IOException {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MIN_FILTER,
                GLES20.GL_LINEAR);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MAG_FILTER,
                GLES20.GL_LINEAR);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_S,
                GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(
                GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_T,
                GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);

        int vertex = ShaderUtil.loadShader(
                TAG, context, GLES20.GL_VERTEX_SHADER, "shaders/camera_background.vert");
        int fragment = ShaderUtil.loadShader(
                TAG, context, GLES20.GL_FRAGMENT_SHADER, "shaders/camera_background.frag");
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertex);
        GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program);

        int[] linked = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            throw new IOException("Camera shader link failed: " + GLES20.glGetProgramInfoLog(program));
        }

        positionAttribute = GLES20.glGetAttribLocation(program, "a_Position");
        textureCoordinateAttribute = GLES20.glGetAttribLocation(program, "a_TexCoord");
        textureUniform = GLES20.glGetUniformLocation(program, "u_CameraTexture");
        ShaderUtil.checkGlError(TAG, "createOnGlThread");
    }

    public int getTextureId() {
        return textureId;
    }

    public void draw(Frame frame) {
        if (textureId < 0 || frame.getTimestamp() == 0) {
            return;
        }

        quadCoords.rewind();
        textureCoords.rewind();
        frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                quadCoords,
                Coordinates2d.TEXTURE_NORMALIZED,
                textureCoords);
        quadCoords.rewind();
        textureCoords.rewind();

        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glDepthMask(false);
        GLES20.glUseProgram(program);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glUniform1i(textureUniform, 0);

        GLES20.glEnableVertexAttribArray(positionAttribute);
        GLES20.glVertexAttribPointer(
                positionAttribute, 2, GLES20.GL_FLOAT, false, 0, quadCoords);
        GLES20.glEnableVertexAttribArray(textureCoordinateAttribute);
        GLES20.glVertexAttribPointer(
                textureCoordinateAttribute, 2, GLES20.GL_FLOAT, false, 0, textureCoords);

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

        GLES20.glDisableVertexAttribArray(positionAttribute);
        GLES20.glDisableVertexAttribArray(textureCoordinateAttribute);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);
        GLES20.glDepthMask(true);
        ShaderUtil.checkGlError(TAG, "draw");
    }

    private static FloatBuffer floatBuffer(float[] values) {
        FloatBuffer buffer = ByteBuffer.allocateDirect(values.length * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        buffer.put(values);
        buffer.rewind();
        return buffer;
    }
}
