package com.sktpj.pointcloudsplatting;

import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Locale;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

interface RendererStatusListener {
    void onStatus(String message);
}

final class GaussianEs3Renderer implements GLSurfaceView.Renderer {
    private static final String TAG = "GaussianViewer";
    private static final float MIN_DISPLAY_SCALE = 0.05f;
    private static final float MAX_DISPLAY_SCALE = 3.0f;
    private static final int INSTANCE_FLOATS = 23;
    private static final int INSTANCE_STRIDE_BYTES = INSTANCE_FLOATS * Float.BYTES;

    private static final float[] QUAD_CORNERS = {
            -1f, -1f,
             1f, -1f,
             1f,  1f,
            -1f, -1f,
             1f,  1f,
            -1f,  1f
    };

    private static final String VERTEX_SHADER =
            "#version 300 es\n"
                    + "precision highp float;\n"
                    + "in vec2 a_Corner;\n"
                    + "in vec3 a_Position;\n"
                    + "in vec4 a_DcAlpha;\n"
                    + "in vec3 a_ShR;\n"
                    + "in vec3 a_ShG;\n"
                    + "in vec3 a_ShB;\n"
                    + "in vec3 a_Scale;\n"
                    + "in vec4 a_Rotation;\n"
                    + "uniform mat4 u_View;\n"
                    + "uniform mat4 u_Projection;\n"
                    + "uniform vec3 u_Camera;\n"
                    + "uniform vec2 u_PixelNdc;\n"
                    + "uniform float u_SizeScale;\n"
                    + "out vec4 v_Color;\n"
                    + "out vec2 v_GaussianCoord;\n"
                    + "mat3 quatRotation(vec4 qIn) {\n"
                    + "  vec4 q = normalize(qIn);\n"
                    + "  float w=q.x, x=q.y, y=q.z, z=q.w;\n"
                    + "  float xx=x*x, yy=y*y, zz=z*z;\n"
                    + "  float xy=x*y, xz=x*z, yz=y*z;\n"
                    + "  float wx=w*x, wy=w*y, wz=w*z;\n"
                    + "  return mat3(\n"
                    + "    vec3(1.0-2.0*(yy+zz), 2.0*(xy+wz), 2.0*(xz-wy)),\n"
                    + "    vec3(2.0*(xy-wz), 1.0-2.0*(xx+zz), 2.0*(yz+wx)),\n"
                    + "    vec3(2.0*(xz+wy), 2.0*(yz-wx), 1.0-2.0*(xx+yy)));\n"
                    + "}\n"
                    + "void main() {\n"
                    + "  vec4 viewCenter = u_View * vec4(a_Position, 1.0);\n"
                    + "  float depth = -viewCenter.z;\n"
                    + "  if (depth <= 0.01) {\n"
                    + "    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n"
                    + "    v_Color = vec4(0.0);\n"
                    + "    v_GaussianCoord = vec2(99.0);\n"
                    + "    return;\n"
                    + "  }\n"
                    + "  mat3 rotation = quatRotation(a_Rotation);\n"
                    + "  mat3 scale = mat3(\n"
                    + "    vec3(a_Scale.x,0.0,0.0),\n"
                    + "    vec3(0.0,a_Scale.y,0.0),\n"
                    + "    vec3(0.0,0.0,a_Scale.z));\n"
                    + "  mat3 m = mat3(u_View) * rotation * scale;\n"
                    + "  mat3 cov = m * transpose(m);\n"
                    + "  float invZ = 1.0 / depth;\n"
                    + "  float invZ2 = invZ * invZ;\n"
                    + "  float fx = u_Projection[0][0];\n"
                    + "  float fy = u_Projection[1][1];\n"
                    + "  vec3 j0 = vec3(fx*invZ, 0.0, fx*viewCenter.x*invZ2);\n"
                    + "  vec3 j1 = vec3(0.0, fy*invZ, fy*viewCenter.y*invZ2);\n"
                    + "  float cxx = dot(j0, cov*j0) + 0.04*u_PixelNdc.x*u_PixelNdc.x;\n"
                    + "  float cxy = dot(j0, cov*j1);\n"
                    + "  float cyy = dot(j1, cov*j1) + 0.04*u_PixelNdc.y*u_PixelNdc.y;\n"
                    + "  float trace = cxx + cyy;\n"
                    + "  float disc = sqrt(max(0.0, (cxx-cyy)*(cxx-cyy) + 4.0*cxy*cxy));\n"
                    + "  float l1 = max(1e-10, 0.5*(trace + disc));\n"
                    + "  float l2 = max(1e-10, 0.5*(trace - disc));\n"
                    + "  vec2 e1;\n"
                    + "  if (abs(cxy) > 1e-10) {\n"
                    + "    e1 = normalize(vec2(cxy, l1-cxx));\n"
                    + "  } else {\n"
                    + "    e1 = cxx >= cyy ? vec2(1.0,0.0) : vec2(0.0,1.0);\n"
                    + "  }\n"
                    + "  vec2 e2 = vec2(-e1.y, e1.x);\n"
                    + "  vec2 offsetNdc = (e1*sqrt(l1)*a_Corner.x + e2*sqrt(l2)*a_Corner.y)\n"
                    + "      * (3.0*u_SizeScale);\n"
                    + "  vec4 clip = u_Projection * viewCenter;\n"
                    + "  clip.xy += offsetNdc * clip.w;\n"
                    + "  gl_Position = clip;\n"
                    + "  vec3 d = normalize(a_Position - u_Camera);\n"
                    + "  const float c0 = 0.2820947918;\n"
                    + "  const float c1 = 0.4886025119;\n"
                    + "  vec3 basis = vec3(-c1*d.y, c1*d.z, -c1*d.x);\n"
                    + "  vec3 rgb = vec3(\n"
                    + "    0.5 + c0*a_DcAlpha.r + dot(a_ShR,basis),\n"
                    + "    0.5 + c0*a_DcAlpha.g + dot(a_ShG,basis),\n"
                    + "    0.5 + c0*a_DcAlpha.b + dot(a_ShB,basis));\n"
                    + "  v_Color = vec4(clamp(rgb,0.0,1.0), clamp(a_DcAlpha.a,0.001,0.999));\n"
                    + "  v_GaussianCoord = a_Corner * 3.0;\n"
                    + "}\n";

    private static final String FRAGMENT_SHADER =
            "#version 300 es\n"
                    + "precision highp float;\n"
                    + "in vec4 v_Color;\n"
                    + "in vec2 v_GaussianCoord;\n"
                    + "out vec4 outColor;\n"
                    + "void main() {\n"
                    + "  float r2 = dot(v_GaussianCoord, v_GaussianCoord);\n"
                    + "  if (r2 > 9.0) discard;\n"
                    + "  float alpha = v_Color.a * exp(-0.5*r2);\n"
                    + "  if (alpha < 0.0039215686) discard;\n"
                    + "  outColor = vec4(v_Color.rgb, alpha);\n"
                    + "}\n";

    private final RendererStatusListener statusListener;
    private volatile GaussianModel model;
    private int program;
    private int cornerVbo;
    private int instanceVbo;
    private int cornerLocation;
    private int positionLocation;
    private int dcAlphaLocation;
    private int shRLocation;
    private int shGLocation;
    private int shBLocation;
    private int scaleLocation;
    private int rotationLocation;
    private int viewLocation;
    private int projectionLocation;
    private int cameraLocation;
    private int pixelNdcLocation;
    private int sizeScaleLocation;
    private int width = 1;
    private int height = 1;
    private float yawDegrees;
    private float pitchDegrees = -10f;
    private float distance = 2.8f;
    private float baseDistance = 2.8f;
    private float displayScale = MIN_DISPLAY_SCALE;
    private boolean firstDrawReported;
    private boolean drawErrorReported;
    private volatile boolean sortedUploadDirty = true;
    private FloatBuffer uploadBuffer;

    GaussianEs3Renderer(RendererStatusListener statusListener) {
        this.statusListener = statusListener;
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        try {
            GLES30.glClearColor(0.04f, 0.04f, 0.04f, 1f);
            int vertex = compileShader(GLES30.GL_VERTEX_SHADER, VERTEX_SHADER);
            int fragment = compileShader(GLES30.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
            program = GLES30.glCreateProgram();
            GLES30.glAttachShader(program, vertex);
            GLES30.glAttachShader(program, fragment);
            GLES30.glLinkProgram(program);
            int[] linked = new int[1];
            GLES30.glGetProgramiv(program, GLES30.GL_LINK_STATUS, linked, 0);
            if (linked[0] == 0) {
                throw new IllegalStateException(
                        "viewer shader link failed: " + GLES30.glGetProgramInfoLog(program));
            }
            GLES30.glDeleteShader(vertex);
            GLES30.glDeleteShader(fragment);

            cornerLocation = requireAttribute("a_Corner");
            positionLocation = requireAttribute("a_Position");
            dcAlphaLocation = requireAttribute("a_DcAlpha");
            shRLocation = requireAttribute("a_ShR");
            shGLocation = requireAttribute("a_ShG");
            shBLocation = requireAttribute("a_ShB");
            scaleLocation = requireAttribute("a_Scale");
            rotationLocation = requireAttribute("a_Rotation");
            viewLocation = requireUniform("u_View");
            projectionLocation = requireUniform("u_Projection");
            cameraLocation = requireUniform("u_Camera");
            pixelNdcLocation = requireUniform("u_PixelNdc");
            sizeScaleLocation = requireUniform("u_SizeScale");

            int[] buffers = new int[2];
            GLES30.glGenBuffers(2, buffers, 0);
            cornerVbo = buffers[0];
            instanceVbo = buffers[1];

            FloatBuffer corners = directFloats(QUAD_CORNERS.length);
            corners.put(QUAD_CORNERS).position(0);
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, cornerVbo);
            GLES30.glBufferData(
                    GLES30.GL_ARRAY_BUFFER,
                    QUAD_CORNERS.length * Float.BYTES,
                    corners,
                    GLES30.GL_STATIC_DRAW);

            GLES30.glEnable(GLES30.GL_BLEND);
            GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA);
            GLES30.glDisable(GLES30.GL_DEPTH_TEST);
            GLES30.glDepthMask(false);
            sortedUploadDirty = true;

            DiagnosticLog.i(
                    TAG,
                    "GL ready renderer=" + GLES30.glGetString(GLES30.GL_RENDERER)
                            + " version=" + GLES30.glGetString(GLES30.GL_VERSION)
                            + " vendor=" + GLES30.glGetString(GLES30.GL_VENDOR)
                            + " viewer=ANISOTROPIC_COVARIANCE_SORTED_ES3"
                            + " appearance=JPEG_SH1");
        } catch (RuntimeException e) {
            program = 0;
            DiagnosticLog.e(TAG, "Viewer GL initialization failed", e);
            statusListener.onStatus(
                    "3D表示を初期化できませんでした\n戻って診断情報をコピーしてください");
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        this.width = Math.max(1, width);
        this.height = Math.max(1, height);
        GLES30.glViewport(0, 0, width, height);
        DiagnosticLog.i(TAG, "Viewport " + this.width + "x" + this.height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);
        GaussianModel current = model;
        if (current == null || current.gaussianCount == 0 || program == 0) {
            return;
        }

        float yaw;
        float pitch;
        float cameraDistance;
        float sizeScale;
        synchronized (this) {
            yaw = yawDegrees;
            pitch = pitchDegrees;
            cameraDistance = distance;
            sizeScale = displayScale;
        }

        float yawRad = (float) Math.toRadians(yaw);
        float pitchRad = (float) Math.toRadians(pitch);
        float cosPitch = (float) Math.cos(pitchRad);
        float eyeX = cameraDistance * cosPitch * (float) Math.sin(yawRad);
        float eyeY = cameraDistance * (float) Math.sin(pitchRad);
        float eyeZ = cameraDistance * cosPitch * (float) Math.cos(yawRad);

        if (sortedUploadDirty) {
            uploadSortedInstances(current, eyeX, eyeY, eyeZ);
            sortedUploadDirty = false;
        }

        float[] view = new float[16];
        float[] projection = new float[16];
        Matrix.setLookAtM(
                view, 0,
                eyeX, eyeY, eyeZ,
                0f, 0f, 0f,
                0f, 1f, 0f);
        Matrix.perspectiveM(
                projection, 0,
                55f,
                (float) width / (float) height,
                0.05f,
                30f);

        GLES30.glUseProgram(program);
        GLES30.glUniformMatrix4fv(viewLocation, 1, false, view, 0);
        GLES30.glUniformMatrix4fv(projectionLocation, 1, false, projection, 0);
        GLES30.glUniform3f(cameraLocation, eyeX, eyeY, eyeZ);
        GLES30.glUniform2f(
                pixelNdcLocation,
                2f / Math.max(1, width),
                2f / Math.max(1, height));
        GLES30.glUniform1f(sizeScaleLocation, sizeScale);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, cornerVbo);
        GLES30.glEnableVertexAttribArray(cornerLocation);
        GLES30.glVertexAttribPointer(
                cornerLocation, 2, GLES30.GL_FLOAT, false, 2 * Float.BYTES, 0);
        GLES30.glVertexAttribDivisor(cornerLocation, 0);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, instanceVbo);
        bindInstance(positionLocation, 3, 0);
        bindInstance(dcAlphaLocation, 4, 3);
        bindInstance(shRLocation, 3, 7);
        bindInstance(shGLocation, 3, 10);
        bindInstance(shBLocation, 3, 13);
        bindInstance(scaleLocation, 3, 16);
        bindInstance(rotationLocation, 4, 19);

        GLES30.glDrawArraysInstanced(
                GLES30.GL_TRIANGLES,
                0,
                6,
                current.gaussianCount);

        disableAttribute(cornerLocation);
        disableInstance(positionLocation);
        disableInstance(dcAlphaLocation);
        disableInstance(shRLocation);
        disableInstance(shGLocation);
        disableInstance(shBLocation);
        disableInstance(scaleLocation);
        disableInstance(rotationLocation);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);

        int error = GLES30.glGetError();
        if (error != GLES30.GL_NO_ERROR) {
            if (!drawErrorReported) {
                drawErrorReported = true;
                DiagnosticLog.e(
                        TAG,
                        "Viewer draw failed glError=0x" + Integer.toHexString(error));
                statusListener.onStatus(
                        "3Dモデルを描画できませんでした\n戻って診断情報をコピーしてください");
            }
            return;
        }

        if (!firstDrawReported) {
            firstDrawReported = true;
            DiagnosticLog.i(
                    TAG,
                    String.format(
                            Locale.US,
                            "First model draw succeeded gaussians=%d sizeScale=%.3f "
                                    + "viewer=ANISOTROPIC_COVARIANCE_SORTED_ES3 "
                                    + "appearance=JPEG_SH1 alpha=BACK_TO_FRONT",
                            current.gaussianCount,
                            sizeScale));
            statusListener.onStatus(
                    "3Dモデルを表示中\nドラッグで回転 / ピンチで拡大・縮小");
        }
    }

    private void bindInstance(int location, int size, int floatOffset) {
        GLES30.glEnableVertexAttribArray(location);
        GLES30.glVertexAttribPointer(
                location,
                size,
                GLES30.GL_FLOAT,
                false,
                INSTANCE_STRIDE_BYTES,
                floatOffset * Float.BYTES);
        GLES30.glVertexAttribDivisor(location, 1);
    }

    private void disableInstance(int location) {
        GLES30.glVertexAttribDivisor(location, 0);
        GLES30.glDisableVertexAttribArray(location);
    }

    private void disableAttribute(int location) {
        GLES30.glDisableVertexAttribArray(location);
    }

    private void uploadSortedInstances(
            GaussianModel current,
            float eyeX,
            float eyeY,
            float eyeZ) {
        int count = current.gaussianCount;
        int neededFloats = count * INSTANCE_FLOATS;
        if (uploadBuffer == null || uploadBuffer.capacity() < neededFloats) {
            uploadBuffer = directFloats(neededFloats);
        }
        uploadBuffer.clear();

        float eyeLen = (float) Math.sqrt(
                eyeX * eyeX + eyeY * eyeY + eyeZ * eyeZ);
        float invEye = eyeLen > 1e-6f ? 1f / eyeLen : 1f;
        float forwardX = -eyeX * invEye;
        float forwardY = -eyeY * invEye;
        float forwardZ = -eyeZ * invEye;

        int[] indices = new int[count];
        float[] depths = new float[count];
        for (int i = 0; i < count; i++) {
            indices[i] = i;
            int p = i * 3;
            float dx = current.positions[p] - eyeX;
            float dy = current.positions[p + 1] - eyeY;
            float dz = current.positions[p + 2] - eyeZ;
            depths[i] = dx * forwardX + dy * forwardY + dz * forwardZ;
        }
        sortByDepthDescending(indices, depths, 0, count - 1);

        for (int sorted = 0; sorted < count; sorted++) {
            int i = indices[sorted];
            int p = i * 3;
            int s = i * 9;
            int r = i * 4;
            uploadBuffer
                    .put(current.positions[p])
                    .put(current.positions[p + 1])
                    .put(current.positions[p + 2])
                    .put(current.dc[p])
                    .put(current.dc[p + 1])
                    .put(current.dc[p + 2])
                    .put(current.alpha[i])
                    .put(current.sh1[s])
                    .put(current.sh1[s + 1])
                    .put(current.sh1[s + 2])
                    .put(current.sh1[s + 3])
                    .put(current.sh1[s + 4])
                    .put(current.sh1[s + 5])
                    .put(current.sh1[s + 6])
                    .put(current.sh1[s + 7])
                    .put(current.sh1[s + 8])
                    .put(current.scales[p])
                    .put(current.scales[p + 1])
                    .put(current.scales[p + 2])
                    .put(current.rotations[r])
                    .put(current.rotations[r + 1])
                    .put(current.rotations[r + 2])
                    .put(current.rotations[r + 3]);
        }
        uploadBuffer.flip();

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, instanceVbo);
        GLES30.glBufferData(
                GLES30.GL_ARRAY_BUFFER,
                uploadBuffer.remaining() * Float.BYTES,
                uploadBuffer,
                GLES30.GL_STREAM_DRAW);
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0);

        DiagnosticLog.i(
                TAG,
                "Sorted/uploaded " + count + " anisotropic Gaussians for current view");
    }

    private static void sortByDepthDescending(
            int[] indices,
            float[] depths,
            int left,
            int right) {
        int i = left;
        int j = right;
        float pivot = depths[indices[(left + right) >>> 1]];
        while (i <= j) {
            while (i <= right && depths[indices[i]] > pivot) {
                i++;
            }
            while (j >= left && depths[indices[j]] < pivot) {
                j--;
            }
            if (i <= j) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
                i++;
                j--;
            }
        }
        if (left < j) {
            sortByDepthDescending(indices, depths, left, j);
        }
        if (i < right) {
            sortByDepthDescending(indices, depths, i, right);
        }
    }

    void setModel(GaussianModel model) {
        this.model = model;
        synchronized (this) {
            baseDistance = 2.8f;
            distance = baseDistance;
            yawDegrees = 0f;
            pitchDegrees = -10f;
            firstDrawReported = false;
            drawErrorReported = false;
            sortedUploadDirty = true;
        }
    }

    synchronized void setDisplaySizeProgress(int progress) {
        float t = Math.max(0f, Math.min(1f, progress / 100f));
        displayScale = MIN_DISPLAY_SCALE
                * (float) Math.pow(MAX_DISPLAY_SCALE / MIN_DISPLAY_SCALE, t);
    }

    synchronized void rotate(float dxDegrees, float dyDegrees) {
        yawDegrees = (yawDegrees - dxDegrees) % 360f;
        pitchDegrees = Math.max(
                -85f,
                Math.min(85f, pitchDegrees - dyDegrees));
        sortedUploadDirty = true;
    }

    synchronized void zoom(float scaleFactor) {
        if (model == null || !Float.isFinite(scaleFactor) || scaleFactor <= 0f) {
            return;
        }
        distance /= scaleFactor;
        distance = Math.max(0.8f, Math.min(12f, distance));
    }

    synchronized void resetCamera() {
        yawDegrees = 0f;
        pitchDegrees = -10f;
        distance = baseDistance;
        sortedUploadDirty = true;
    }

    private int requireAttribute(String name) {
        int location = GLES30.glGetAttribLocation(program, name);
        if (location < 0) {
            throw new IllegalStateException("missing viewer attribute " + name);
        }
        return location;
    }

    private int requireUniform(String name) {
        int location = GLES30.glGetUniformLocation(program, name);
        if (location < 0) {
            throw new IllegalStateException("missing viewer uniform " + name);
        }
        return location;
    }

    private static FloatBuffer directFloats(int count) {
        return ByteBuffer.allocateDirect(count * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
    }

    private static int compileShader(int type, String source) {
        int shader = GLES30.glCreateShader(type);
        GLES30.glShaderSource(shader, source);
        GLES30.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES30.glGetShaderiv(
                shader,
                GLES30.GL_COMPILE_STATUS,
                compiled,
                0);
        if (compiled[0] == 0) {
            String error = GLES30.glGetShaderInfoLog(shader);
            GLES30.glDeleteShader(shader);
            throw new IllegalStateException(
                    "viewer shader compile failed: " + error);
        }
        return shader;
    }
}
