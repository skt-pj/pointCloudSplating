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

/** Camera background renderer derived from google-ar/codelab-raw-depth-api (Apache-2.0). */
public final class BackgroundRenderer {
    private static final String TAG = "BackgroundRenderer";
    private static final float[] QUAD_COORDS = {
            -1.0f, -1.0f,
             1.0f, -1.0f,
            -1.0f,  1.0f,
             1.0f,  1.0f
    };

    private final FloatBuffer quadCoords = directFloatBuffer(QUAD_COORDS.length);
    private final FloatBuffer quadTexCoords = directFloatBuffer(8);
    private int textureId = -1;
    private int program;
    private int positionAttribute;
    private int texCoordAttribute;
    private int textureUniform;

    public BackgroundRenderer() {
        quadCoords.put(QUAD_COORDS).position(0);
    }

    public int getTextureId() {
        return textureId;
    }

    public void createOnGlThread(Context context) throws IOException {
        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        textureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);

        int vertex = ShaderUtil.loadShader(TAG, context, GLES20.GL_VERTEX_SHADER,
                "shaders/camera.vert");
        int fragment = ShaderUtil.loadShader(TAG, context, GLES20.GL_FRAGMENT_SHADER,
                "shaders/camera.frag");
        program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vertex);
        GLES20.glAttachShader(program, fragment);
        GLES20.glLinkProgram(program);

        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] == 0) {
            throw new IOException("Camera shader link failed: " + GLES20.glGetProgramInfoLog(program));
        }

        positionAttribute = GLES20.glGetAttribLocation(program, "a_Position");
        texCoordAttribute = GLES20.glGetAttribLocation(program, "a_TexCoord");
        textureUniform = GLES20.glGetUniformLocation(program, "sTexture");
        ShaderUtil.checkGlError(TAG, "createOnGlThread");
    }

    public void draw(Frame frame) {
        if (frame.hasDisplayGeometryChanged()) {
            quadCoords.position(0);
            quadTexCoords.position(0);
            frame.transformCoordinates2d(
                    Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,
                    quadCoords,
                    Coordinates2d.TEXTURE_NORMALIZED,
                    quadTexCoords);
        }
        if (frame.getTimestamp() == 0) {
            return;
        }

        quadCoords.position(0);
        quadTexCoords.position(0);
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);
        GLES20.glDepthMask(false);
        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId);
        GLES20.glUniform1i(textureUniform, 0);
        GLES20.glVertexAttribPointer(positionAttribute, 2, GLES20.GL_FLOAT, false, 0, quadCoords);
        GLES20.glVertexAttribPointer(texCoordAttribute, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords);
        GLES20.glEnableVertexAttribArray(positionAttribute);
        GLES20.glEnableVertexAttribArray(texCoordAttribute);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glDisableVertexAttribArray(positionAttribute);
        GLES20.glDisableVertexAttribArray(texCoordAttribute);
        GLES20.glDepthMask(true);
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        ShaderUtil.checkGlError(TAG, "draw");
    }

    private static FloatBuffer directFloatBuffer(int floatCount) {
        return ByteBuffer.allocateDirect(floatCount * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
    }
}
