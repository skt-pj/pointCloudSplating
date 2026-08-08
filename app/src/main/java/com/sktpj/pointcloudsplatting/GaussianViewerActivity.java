package com.sktpj.pointcloudsplatting;

import android.app.Activity;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** OpenGL ES viewer for the app's generated splat.ply. */
public final class GaussianViewerActivity extends Activity {
    public static final String EXTRA_DATASET_PATH = "dataset_path";

    private GLSurfaceView glView;
    private GaussianRenderer renderer;
    private TextView statusView;
    private ScaleGestureDetector scaleDetector;
    private float lastX;
    private float lastY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        String datasetPath = getIntent().getStringExtra(EXTRA_DATASET_PATH);
        if (datasetPath == null) {
            Toast.makeText(this, "3Dモデルを開けませんでした。", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        File dataset = new File(datasetPath);
        File splat = new File(dataset, "splat.ply");
        if (!splat.isFile()) {
            Toast.makeText(this, "3Dモデルを開けませんでした。", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        renderer = new GaussianRenderer(message -> runOnUiThread(() -> {
            if (statusView != null) {
                statusView.setText(message);
            }
        }));
        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(2);
        glView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        glView.setPreserveEGLContextOnPause(true);
        glView.setRenderer(renderer);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        scaleDetector = new ScaleGestureDetector(this,
                new ScaleGestureDetector.SimpleOnScaleGestureListener() {
                    @Override
                    public boolean onScale(ScaleGestureDetector detector) {
                        renderer.zoom(detector.getScaleFactor());
                        return true;
                    }
                });
        glView.setOnTouchListener(this::handleTouch);
        glView.setContentDescription("3Dモデル表示。指一本で回転、二本指で拡大縮小できます");

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF101010);
        root.addView(glView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        LinearOverlay overlay = new LinearOverlay(this);
        overlay.button("戻る", "保存したスキャンに戻る", v -> finish());
        statusView = overlay.status();
        statusView.setText("3Dモデルを読み込んでいます…\n少しお待ちください");
        overlay.button("正面に戻す", "3Dモデルの向きと大きさを元に戻す",
                v -> renderer.resetCamera());

        FrameLayout.LayoutParams topParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(64));
        topParams.gravity = Gravity.TOP;
        topParams.leftMargin = dp(6);
        topParams.rightMargin = dp(6);
        topParams.topMargin = dp(6);
        root.addView(overlay.root, topParams);

        TextView title = new TextView(this);
        title.setText("3Dモデル");
        title.setTextColor(0xFFFFFFFF);
        title.setTextSize(14f);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        title.setBackgroundColor(0x66000000);
        FrameLayout.LayoutParams titleParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(32));
        titleParams.gravity = Gravity.BOTTOM;
        root.addView(title, titleParams);

        setContentView(root);
        loadModelAsync(splat);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (glView != null) {
            glView.onResume();
        }
    }

    @Override
    protected void onPause() {
        if (glView != null) {
            glView.onPause();
        }
        super.onPause();
    }

    private void loadModelAsync(File splat) {
        new Thread(() -> {
            try {
                ModelData model = GaussianPlyReader.read(splat);
                renderer.setModel(model);
                runOnUiThread(() -> statusView.setText(
                        "3Dモデルを画面に準備しています…\n少しお待ちください"));
            } catch (Exception e) {
                DiagnosticLog.e("GaussianViewer", "Failed to load splat.ply", e);
                runOnUiThread(() -> {
                    statusView.setText("3Dモデルを表示できませんでした");
                    Toast.makeText(this,
                            "3Dモデルを表示できませんでした。",
                            Toast.LENGTH_LONG).show();
                });
            }
        }, "LoadGaussianPly").start();
    }

    private boolean handleTouch(View view, MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        if (event.getPointerCount() == 1 && !scaleDetector.isInProgress()) {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    lastX = event.getX();
                    lastY = event.getY();
                    return true;
                case MotionEvent.ACTION_MOVE:
                    float x = event.getX();
                    float y = event.getY();
                    float dx = x - lastX;
                    float dy = y - lastY;
                    lastX = x;
                    lastY = y;
                    renderer.rotate(dx * 0.35f, dy * 0.35f);
                    return true;
                default:
                    break;
            }
        }
        return true;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private interface RendererStatusListener {
        void onStatus(String message);
    }

    private static final class LinearOverlay {
        final android.widget.LinearLayout root;
        private final Activity activity;

        LinearOverlay(Activity activity) {
            this.activity = activity;
            root = new android.widget.LinearLayout(activity);
            root.setOrientation(android.widget.LinearLayout.HORIZONTAL);
            root.setGravity(Gravity.CENTER_VERTICAL);
            root.setBackgroundColor(0x77000000);
        }

        Button button(String text, String contentDescription, View.OnClickListener listener) {
            Button button = new Button(activity);
            button.setText(text);
            button.setAllCaps(false);
            button.setContentDescription(contentDescription);
            button.setMinHeight(Math.round(48 * activity.getResources().getDisplayMetrics().density));
            button.setOnClickListener(listener);
            int widthDp = "正面に戻す".equals(text) ? 128 : 86;
            root.addView(button, new android.widget.LinearLayout.LayoutParams(
                    Math.round(widthDp * activity.getResources().getDisplayMetrics().density),
                    android.widget.LinearLayout.LayoutParams.MATCH_PARENT));
            return button;
        }

        TextView status() {
            TextView status = new TextView(activity);
            status.setTextColor(0xFFFFFFFF);
            status.setTextSize(13f);
            status.setGravity(Gravity.CENTER);
            root.addView(status, new android.widget.LinearLayout.LayoutParams(
                    0, android.widget.LinearLayout.LayoutParams.MATCH_PARENT, 1f));
            return status;
        }
    }

    private static final class GaussianRenderer implements GLSurfaceView.Renderer {
        private static final String TAG = "GaussianViewer";
        private static final String VERTEX_SHADER =
                "attribute vec3 a_Position;\n"
                        + "attribute vec4 a_Color;\n"
                        + "attribute float a_Size;\n"
                        + "attribute vec2 a_Corner;\n"
                        + "uniform mat4 u_View;\n"
                        + "uniform mat4 u_Projection;\n"
                        + "uniform vec3 u_Center;\n"
                        + "uniform float u_Radius;\n"
                        + "varying vec4 v_Color;\n"
                        + "varying vec2 v_Corner;\n"
                        + "void main() {\n"
                        + "  float radius = max(u_Radius, 0.0001);\n"
                        + "  vec3 normalizedPosition = (a_Position - u_Center) / radius;\n"
                        + "  vec4 viewCenter = u_View * vec4(normalizedPosition, 1.0);\n"
                        + "  float normalizedSize = clamp((a_Size / radius) * 2.8, 0.006, 0.10);\n"
                        + "  vec4 viewPosition = viewCenter + vec4(a_Corner * normalizedSize, 0.0, 0.0);\n"
                        + "  gl_Position = u_Projection * viewPosition;\n"
                        + "  v_Color = a_Color;\n"
                        + "  v_Corner = a_Corner;\n"
                        + "}\n";

        private static final String FRAGMENT_SHADER =
                "precision mediump float;\n"
                        + "varying vec4 v_Color;\n"
                        + "varying vec2 v_Corner;\n"
                        + "void main() {\n"
                        + "  float r2 = dot(v_Corner, v_Corner);\n"
                        + "  if (r2 > 1.0) discard;\n"
                        + "  float weight = exp(-2.4 * r2);\n"
                        + "  float alpha = max(0.42, v_Color.a) * weight;\n"
                        + "  gl_FragColor = vec4(v_Color.rgb, alpha);\n"
                        + "}\n";

        private final RendererStatusListener statusListener;
        private volatile ModelData model;
        private int program;
        private int positionLocation;
        private int colorLocation;
        private int sizeLocation;
        private int cornerLocation;
        private int viewLocation;
        private int projectionLocation;
        private int centerLocation;
        private int radiusLocation;
        private int width = 1;
        private int height = 1;
        private float yawDegrees;
        private float pitchDegrees = -10f;
        private float distance = 2.8f;
        private float baseDistance = 2.8f;
        private boolean firstDrawReported;
        private boolean drawErrorReported;

        GaussianRenderer(RendererStatusListener statusListener) {
            this.statusListener = statusListener;
        }

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            try {
                GLES20.glClearColor(0.04f, 0.04f, 0.04f, 1f);
                int vertex = compileShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
                int fragment = compileShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
                program = GLES20.glCreateProgram();
                GLES20.glAttachShader(program, vertex);
                GLES20.glAttachShader(program, fragment);
                GLES20.glLinkProgram(program);
                int[] linked = new int[1];
                GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linked, 0);
                if (linked[0] == 0) {
                    throw new IllegalStateException("viewer shader link failed: "
                            + GLES20.glGetProgramInfoLog(program));
                }
                positionLocation = requireAttribute("a_Position");
                colorLocation = requireAttribute("a_Color");
                sizeLocation = requireAttribute("a_Size");
                cornerLocation = requireAttribute("a_Corner");
                viewLocation = requireUniform("u_View");
                projectionLocation = requireUniform("u_Projection");
                centerLocation = requireUniform("u_Center");
                radiusLocation = requireUniform("u_Radius");

                GLES20.glEnable(GLES20.GL_BLEND);
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
                GLES20.glEnable(GLES20.GL_DEPTH_TEST);
                GLES20.glDepthFunc(GLES20.GL_LEQUAL);
                GLES20.glDepthMask(true);
                DiagnosticLog.i(TAG,
                        "GL ready renderer=" + GLES20.glGetString(GLES20.GL_RENDERER)
                                + " version=" + GLES20.glGetString(GLES20.GL_VERSION)
                                + " vendor=" + GLES20.glGetString(GLES20.GL_VENDOR));
            } catch (RuntimeException e) {
                program = 0;
                DiagnosticLog.e(TAG, "Viewer GL initialization failed", e);
                statusListener.onStatus(
                        "3D表示を初期化できませんでした\n戻ってログをコピーしてください");
            }
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            this.width = Math.max(1, width);
            this.height = Math.max(1, height);
            GLES20.glViewport(0, 0, width, height);
            DiagnosticLog.i(TAG, "Viewport " + this.width + "x" + this.height);
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
            ModelData current = model;
            if (current == null || current.vertexCount == 0 || program == 0) {
                return;
            }

            float yaw;
            float pitch;
            float cameraDistance;
            synchronized (this) {
                yaw = yawDegrees;
                pitch = pitchDegrees;
                cameraDistance = distance;
            }

            float yawRad = (float) Math.toRadians(yaw);
            float pitchRad = (float) Math.toRadians(pitch);
            float cosPitch = (float) Math.cos(pitchRad);
            float eyeX = cameraDistance * cosPitch * (float) Math.sin(yawRad);
            float eyeY = cameraDistance * (float) Math.sin(pitchRad);
            float eyeZ = cameraDistance * cosPitch * (float) Math.cos(yawRad);

            float[] view = new float[16];
            float[] projection = new float[16];
            Matrix.setLookAtM(view, 0,
                    eyeX, eyeY, eyeZ,
                    0f, 0f, 0f,
                    0f, 1f, 0f);
            float aspect = (float) width / (float) height;
            Matrix.perspectiveM(projection, 0, 55f, aspect, 0.05f, 30f);

            GLES20.glUseProgram(program);
            GLES20.glUniformMatrix4fv(viewLocation, 1, false, view, 0);
            GLES20.glUniformMatrix4fv(projectionLocation, 1, false, projection, 0);
            GLES20.glUniform3f(centerLocation,
                    current.centerX, current.centerY, current.centerZ);
            GLES20.glUniform1f(radiusLocation, current.radius);

            current.positions.position(0);
            current.colors.position(0);
            current.sizes.position(0);
            current.corners.position(0);
            GLES20.glEnableVertexAttribArray(positionLocation);
            GLES20.glVertexAttribPointer(positionLocation, 3, GLES20.GL_FLOAT,
                    false, 3 * Float.BYTES, current.positions);
            GLES20.glEnableVertexAttribArray(colorLocation);
            GLES20.glVertexAttribPointer(colorLocation, 4, GLES20.GL_FLOAT,
                    false, 4 * Float.BYTES, current.colors);
            GLES20.glEnableVertexAttribArray(sizeLocation);
            GLES20.glVertexAttribPointer(sizeLocation, 1, GLES20.GL_FLOAT,
                    false, Float.BYTES, current.sizes);
            GLES20.glEnableVertexAttribArray(cornerLocation);
            GLES20.glVertexAttribPointer(cornerLocation, 2, GLES20.GL_FLOAT,
                    false, 2 * Float.BYTES, current.corners);

            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, current.vertexCount);

            GLES20.glDisableVertexAttribArray(positionLocation);
            GLES20.glDisableVertexAttribArray(colorLocation);
            GLES20.glDisableVertexAttribArray(sizeLocation);
            GLES20.glDisableVertexAttribArray(cornerLocation);

            int error = GLES20.glGetError();
            if (error != GLES20.GL_NO_ERROR) {
                if (!drawErrorReported) {
                    drawErrorReported = true;
                    DiagnosticLog.e(TAG, "Viewer draw failed glError=0x"
                            + Integer.toHexString(error));
                    statusListener.onStatus(
                            "3Dモデルを描画できませんでした\n戻ってログをコピーしてください");
                }
                return;
            }
            if (!firstDrawReported) {
                firstDrawReported = true;
                DiagnosticLog.i(TAG,
                        "First model draw succeeded gaussians=" + current.gaussianCount
                                + " triangles=" + (current.vertexCount / 3)
                                + " radius=" + current.radius);
                statusListener.onStatus(
                        "3Dモデルを表示中\nドラッグで回転 / ピンチで拡大・縮小");
            }
        }

        void setModel(ModelData model) {
            this.model = model;
            synchronized (this) {
                baseDistance = 2.8f;
                distance = baseDistance;
                yawDegrees = 0f;
                pitchDegrees = -10f;
                firstDrawReported = false;
                drawErrorReported = false;
            }
        }

        synchronized void rotate(float dxDegrees, float dyDegrees) {
            yawDegrees = (yawDegrees - dxDegrees) % 360f;
            pitchDegrees = Math.max(-85f, Math.min(85f, pitchDegrees - dyDegrees));
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
        }

        private int requireAttribute(String name) {
            int location = GLES20.glGetAttribLocation(program, name);
            if (location < 0) {
                throw new IllegalStateException("missing viewer attribute " + name);
            }
            return location;
        }

        private int requireUniform(String name) {
            int location = GLES20.glGetUniformLocation(program, name);
            if (location < 0) {
                throw new IllegalStateException("missing viewer uniform " + name);
            }
            return location;
        }

        private static int compileShader(int type, String source) {
            int shader = GLES20.glCreateShader(type);
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            int[] compiled = new int[1];
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
            if (compiled[0] == 0) {
                String error = GLES20.glGetShaderInfoLog(shader);
                GLES20.glDeleteShader(shader);
                throw new IllegalStateException("viewer shader compile failed: " + error);
            }
            return shader;
        }
    }

    private static final class ModelData {
        final int gaussianCount;
        final int vertexCount;
        final FloatBuffer positions;
        final FloatBuffer colors;
        final FloatBuffer sizes;
        final FloatBuffer corners;
        final float centerX;
        final float centerY;
        final float centerZ;
        final float radius;

        ModelData(
                int gaussianCount,
                int vertexCount,
                FloatBuffer positions,
                FloatBuffer colors,
                FloatBuffer sizes,
                FloatBuffer corners,
                float centerX,
                float centerY,
                float centerZ,
                float radius) {
            this.gaussianCount = gaussianCount;
            this.vertexCount = vertexCount;
            this.positions = positions;
            this.colors = colors;
            this.sizes = sizes;
            this.corners = corners;
            this.centerX = centerX;
            this.centerY = centerY;
            this.centerZ = centerZ;
            this.radius = radius;
        }
    }

    private static final class GaussianPlyReader {
        private static final String TAG = "GaussianViewer";
        private static final float SH_C0 = 0.28209479177387814f;
        private static final int ROBUST_SAMPLE_LIMIT = 8192;
        private static final float[] QUAD_CORNERS = {
                -1f, -1f, 1f, -1f, 1f, 1f,
                -1f, -1f, 1f, 1f, -1f, 1f
        };

        static ModelData read(File file) throws IOException {
            try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
                Header header = readHeader(input);
                if (!header.binaryLittleEndian) {
                    throw new IOException("binary_little_endian PLY only");
                }
                if (header.vertexCount <= 0) {
                    throw new IOException("PLY has no vertices");
                }
                int xIndex = header.indexOf("x");
                int yIndex = header.indexOf("y");
                int zIndex = header.indexOf("z");
                if (xIndex < 0 || yIndex < 0 || zIndex < 0) {
                    throw new IOException("PLY position properties are missing");
                }
                int rIndex = header.indexOf("f_dc_0");
                int gIndex = header.indexOf("f_dc_1");
                int bIndex = header.indexOf("f_dc_2");
                int opacityIndex = header.indexOf("opacity");
                int sxIndex = header.indexOf("scale_0");
                int syIndex = header.indexOf("scale_1");
                int szIndex = header.indexOf("scale_2");

                long bodyBytes = (long) header.vertexCount * header.properties.size() * Float.BYTES;
                long available = file.length() - header.bodyOffset;
                if (available < bodyBytes) {
                    throw new IOException("truncated splat.ply");
                }
                MappedByteBuffer body = input.getChannel().map(
                        FileChannel.MapMode.READ_ONLY, header.bodyOffset, bodyBytes);
                body.order(ByteOrder.LITTLE_ENDIAN);

                float[] xyz = new float[header.vertexCount * 3];
                float[] rgba = new float[header.vertexCount * 4];
                float[] scales = new float[header.vertexCount];
                boolean[] valid = new boolean[header.vertexCount];
                float[] values = new float[header.properties.size()];
                int validCount = 0;
                int invalidColorCount = 0;
                float minScale = Float.POSITIVE_INFINITY;
                float maxScale = 0f;

                for (int i = 0; i < header.vertexCount; i++) {
                    for (int p = 0; p < values.length; p++) {
                        values[p] = body.getFloat();
                    }
                    float x = values[xIndex];
                    float y = values[yIndex];
                    float z = values[zIndex];
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) {
                        continue;
                    }
                    valid[i] = true;
                    validCount++;
                    xyz[i * 3] = x;
                    xyz[i * 3 + 1] = y;
                    xyz[i * 3 + 2] = z;

                    float red = rIndex >= 0 ? 0.5f + SH_C0 * values[rIndex] : 0.75f;
                    float green = gIndex >= 0 ? 0.5f + SH_C0 * values[gIndex] : 0.75f;
                    float blue = bIndex >= 0 ? 0.5f + SH_C0 * values[bIndex] : 0.75f;
                    if (!Float.isFinite(red) || !Float.isFinite(green) || !Float.isFinite(blue)) {
                        invalidColorCount++;
                        red = green = blue = 0.75f;
                    }
                    float opacityValue = opacityIndex >= 0 ? values[opacityIndex] : 6f;
                    float alpha = Float.isFinite(opacityValue) ? sigmoid(opacityValue) : 0.8f;
                    rgba[i * 4] = clamp01(red);
                    rgba[i * 4 + 1] = clamp01(green);
                    rgba[i * 4 + 2] = clamp01(blue);
                    rgba[i * 4 + 3] = Math.max(0.35f, clamp01(alpha));

                    float worldScale = 0.012f;
                    if (sxIndex >= 0 && syIndex >= 0 && szIndex >= 0
                            && Float.isFinite(values[sxIndex])
                            && Float.isFinite(values[syIndex])
                            && Float.isFinite(values[szIndex])) {
                        float sx = safeExp(values[sxIndex]);
                        float sy = safeExp(values[syIndex]);
                        float sz = safeExp(values[szIndex]);
                        float product = sx * sy * sz;
                        if (Float.isFinite(product) && product > 0f) {
                            worldScale = (float) Math.cbrt(product) * 2.4f;
                        }
                    }
                    if (!Float.isFinite(worldScale) || worldScale <= 0f) {
                        worldScale = 0.012f;
                    }
                    worldScale = Math.max(0.0025f, Math.min(0.14f, worldScale));
                    scales[i] = worldScale;
                    minScale = Math.min(minScale, worldScale);
                    maxScale = Math.max(maxScale, worldScale);
                }

                if (validCount == 0) {
                    throw new IOException("PLY has no finite positions");
                }

                int sampleStride = Math.max(1, validCount / ROBUST_SAMPLE_LIMIT);
                int sampleCapacity = Math.min(validCount,
                        (validCount + sampleStride - 1) / sampleStride + 1);
                float[] sampleX = new float[sampleCapacity];
                float[] sampleY = new float[sampleCapacity];
                float[] sampleZ = new float[sampleCapacity];
                int sampleCount = 0;
                int seenValid = 0;
                for (int i = 0; i < header.vertexCount && sampleCount < sampleCapacity; i++) {
                    if (!valid[i]) {
                        continue;
                    }
                    if (seenValid % sampleStride == 0) {
                        sampleX[sampleCount] = xyz[i * 3];
                        sampleY[sampleCount] = xyz[i * 3 + 1];
                        sampleZ[sampleCount] = xyz[i * 3 + 2];
                        sampleCount++;
                    }
                    seenValid++;
                }

                float[] medianX = Arrays.copyOf(sampleX, sampleCount);
                float[] medianY = Arrays.copyOf(sampleY, sampleCount);
                float[] medianZ = Arrays.copyOf(sampleZ, sampleCount);
                Arrays.sort(medianX);
                Arrays.sort(medianY);
                Arrays.sort(medianZ);
                float centerX = medianX[sampleCount / 2];
                float centerY = medianY[sampleCount / 2];
                float centerZ = medianZ[sampleCount / 2];

                float[] distances = new float[sampleCount];
                for (int i = 0; i < sampleCount; i++) {
                    float dx = sampleX[i] - centerX;
                    float dy = sampleY[i] - centerY;
                    float dz = sampleZ[i] - centerZ;
                    distances[i] = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
                }
                Arrays.sort(distances);
                int percentileIndex = Math.min(sampleCount - 1,
                        Math.max(0, Math.round((sampleCount - 1) * 0.99f)));
                float radius = Math.max(0.05f, distances[percentileIndex] * 1.12f);
                if (!Float.isFinite(radius) || radius <= 0f) {
                    radius = 1f;
                }

                int expandedVertexCount = validCount * 6;
                FloatBuffer positions = directFloats(expandedVertexCount * 3);
                FloatBuffer colors = directFloats(expandedVertexCount * 4);
                FloatBuffer sizes = directFloats(expandedVertexCount);
                FloatBuffer corners = directFloats(expandedVertexCount * 2);
                for (int i = 0; i < header.vertexCount; i++) {
                    if (!valid[i]) {
                        continue;
                    }
                    float x = xyz[i * 3];
                    float y = xyz[i * 3 + 1];
                    float z = xyz[i * 3 + 2];
                    float red = rgba[i * 4];
                    float green = rgba[i * 4 + 1];
                    float blue = rgba[i * 4 + 2];
                    float alpha = rgba[i * 4 + 3];
                    for (int v = 0; v < 6; v++) {
                        positions.put(x).put(y).put(z);
                        colors.put(red).put(green).put(blue).put(alpha);
                        sizes.put(scales[i]);
                        corners.put(QUAD_CORNERS[v * 2]).put(QUAD_CORNERS[v * 2 + 1]);
                    }
                }
                positions.position(0);
                colors.position(0);
                sizes.position(0);
                corners.position(0);

                DiagnosticLog.i(TAG,
                        String.format(Locale.US,
                                "Loaded model gaussians=%d/%d robustCenter=(%.3f,%.3f,%.3f) radius99=%.3f scale=%.5f..%.5f invalidColors=%d",
                                validCount,
                                header.vertexCount,
                                centerX,
                                centerY,
                                centerZ,
                                radius,
                                minScale,
                                maxScale,
                                invalidColorCount));
                return new ModelData(
                        validCount,
                        expandedVertexCount,
                        positions,
                        colors,
                        sizes,
                        corners,
                        centerX,
                        centerY,
                        centerZ,
                        radius);
            }
        }

        private static Header readHeader(RandomAccessFile input) throws IOException {
            Header header = new Header();
            String first = input.readLine();
            if (!"ply".equals(first)) {
                throw new IOException("not a PLY file");
            }
            boolean inVertex = false;
            String line;
            while ((line = input.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith("format ")) {
                    header.binaryLittleEndian = trimmed.startsWith("format binary_little_endian");
                } else if (trimmed.startsWith("element ")) {
                    String[] tokens = trimmed.split("\\s+");
                    inVertex = tokens.length >= 3 && "vertex".equals(tokens[1]);
                    if (inVertex) {
                        header.vertexCount = Integer.parseInt(tokens[2]);
                    }
                } else if (inVertex && trimmed.startsWith("property ")) {
                    String[] tokens = trimmed.split("\\s+");
                    if (tokens.length != 3
                            || !("float".equals(tokens[1]) || "float32".equals(tokens[1]))) {
                        throw new IOException("viewer supports float Gaussian properties only");
                    }
                    header.properties.add(tokens[2]);
                } else if ("end_header".equals(trimmed)) {
                    header.bodyOffset = input.getFilePointer();
                    break;
                }
            }
            if (header.bodyOffset <= 0 || header.properties.isEmpty()) {
                throw new IOException("invalid PLY header");
            }
            return header;
        }

        private static FloatBuffer directFloats(int count) {
            return ByteBuffer.allocateDirect(count * Float.BYTES)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
        }

        private static float safeExp(float value) {
            return (float) Math.exp(Math.max(-20f, Math.min(5f, value)));
        }

        private static float sigmoid(float x) {
            if (x >= 0f) {
                float z = (float) Math.exp(-x);
                return 1f / (1f + z);
            }
            float z = (float) Math.exp(x);
            return z / (1f + z);
        }

        private static float clamp01(float value) {
            if (!Float.isFinite(value)) {
                return 0.75f;
            }
            return Math.max(0f, Math.min(1f, value));
        }

        private static final class Header {
            boolean binaryLittleEndian;
            int vertexCount;
            long bodyOffset;
            final List<String> properties = new ArrayList<>();

            int indexOf(String name) {
                return properties.indexOf(name);
            }
        }
    }
}
