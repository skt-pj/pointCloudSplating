package com.sktpj.pointcloudsplatting;

import android.content.Context;
import android.opengl.GLES20;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

/** Derived from google-ar/codelab-raw-depth-api ShaderUtil.java (Apache-2.0). */
public final class ShaderUtil {
    private ShaderUtil() {}

    public static int loadShader(String tag, Context context, int type, String assetPath)
            throws IOException {
        String source = readAsset(context, assetPath);
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);

        int[] status = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0);
        if (status[0] == 0) {
            String log = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new IOException("Shader compile failed for " + assetPath + ": " + log);
        }
        return shader;
    }

    public static void checkGlError(String tag, String label) {
        int lastError = GLES20.GL_NO_ERROR;
        int error;
        while ((error = GLES20.glGetError()) != GLES20.GL_NO_ERROR) {
            Log.e(tag, label + ": glError " + error);
            lastError = error;
        }
        if (lastError != GLES20.GL_NO_ERROR) {
            throw new RuntimeException(label + ": glError " + lastError);
        }
    }

    private static String readAsset(Context context, String assetPath) throws IOException {
        try (InputStream inputStream = context.getAssets().open(assetPath);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            StringBuilder source = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                source.append(line).append('\n');
            }
            return source.toString();
        }
    }
}
