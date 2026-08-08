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

/** Lightweight OpenGL ES viewer for the app's generated splat.ply. */
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
            Toast.makeText(this, "dataset pathがありません", Toast.LENGTH_LONG).show();
            finish();
            return;
        }
        File dataset = new File(datasetPath);
        File splat = new File(dataset, "splat.ply");
        if (!splat.isFile()) {
            Toast.makeText(this, "splat.plyがありません", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        renderer = new GaussianRenderer();
        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(2);
        glView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
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

        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(0xFF101010);
        root.addView(glView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        LinearOverlay overlay = new LinearOverlay(this);
        overlay.button("戻る", v -> finish());
        statusView = overlay.status();
        statusView.setText("Gaussianを読み込んでいます...\nドラッグ: 回転 / ピンチ: ズーム");
        overlay.button("リセット", v -> renderer.resetCamera());

        FrameLayout.LayoutParams topParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(60));
        topParams.gravity = Gravity.TOP;
        topParams.leftMargin = dp(6);
        topParams.rightMargin = dp(6);
        topParams.topMargin = dp(6);
        root.addView(overlay.root, topParams);

        TextView title = new TextView(this);
        title.setText(dataset.getName());
        title.setTextColor(0xFFFFFFFF);
        title.setTextSize(14f);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        title.setBackgroundColor(0x55000000);
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
                        String.format(Locale.US,
                                "%,d Gaussians / auto-fit\nドラッグ: 回転 / ピンチ: ズーム",
                                model.count)));
            } catch (Exception e) {
                DiagnosticLog.e("GaussianViewer", "Failed to load splat.ply", e);
                runOnUiThread(() -> {
                    statusView.setText("Gaussian読込失敗: " + e.getMessage());
                    Toast.makeText(this, "Gaussianを表示できません", Toast.LENGTH_LONG).show();
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

    private static final class LinearOverlay {
        final android.widget.LinearLayout root;
        private final Activity activity;

        LinearOverlay(Activity activity) {
            this.activity = activity;
            root = new android.widget.LinearLayout(activity);
            root.setOrientation(android.widget.LinearLayout.HORIZONTAL);
            root.setGravity(Gravity.CENTER_VERTICAL);
            root.setBackgroundColor(0x66000000);
        }

        Button button(String text, View.OnClickListener listener) {
            Button button = new Button(activity);
            button.setText(text);
            button.setOnClickListener(listener);
            int widthDp = activity.getResources().getDisplayMetrics().densityDpi >= 320 ? 108 : 92;
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
        private static final String VERTEX_SHADER =
                "attribute vec3 a_Position;\n"
                        + "attribute vec4 a_Color;\n"
                        + "attribute float a_Size;\n"
                        + "uniform mat4 u_Mvp;\n"
                        + "uniform float u_PointScale;\n"
                        + "varying vec4 v_Color;\n"
                        + "void main() {\n"
                        + "  vec4 clip = u_Mvp * vec4(a_Position, 1.0);\n"
                        + "  gl_Position = clip;\n"
                        + "  gl_PointSize = clamp(a_Size * u_PointScale / max(0.05, clip.w), 2.0, 48.0);\n"
                        + "  v_Color = a_Color;\n"
                        + "}\n";

        private static final String FRAGMENT_SHADER =
                "precision mediump float;\n"
                        + "varying vec4 v_Color;\n"
                        + "void main() {\n"
                        + "  vec2 p = gl_PointCoord * 2.0 - 1.0;\n"
                        + "  float r2 = dot(p, p);\n"
                        + "  if (r2 > 1.0) discard;\n"
                        + "  float gaussian = exp(-2.7 * r2);\n"
                        + "  gl_FragColor = vec4(v_Color.rgb, v_Color.a * gaussian);\n"
                        + "}\n";

        private volatile ModelData model;
        private int program;
        private int positionLocation;
        private int colorLocation;
        private int sizeLocation;
        private int mvpLocation;
        private int pointScaleLocation;
        private int width = 1;
        private int height = 1;
        private float yawDegrees;
        private float pitchDegrees;
        private float distance = 2f;
        private float baseDistance = 2f;

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
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
            positionLocation = GLES20.glGetAttribLocation(program, "a_Position");
            colorLocation = GLES20.glGetAttribLocation(program, "a_Color");
            sizeLocation = GLES20.glGetAttribLocation(program, "a_Size");
            mvpLocation = GLES20.glGetUniformLocation(program, "u_Mvp");
            pointScaleLocation = GLES20.glGetUniformLocation(program, "u_PointScale");
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
            GLES20.glEnable(GLES20.GL_DEPTH_TEST);
            GLES20.glDepthMask(false);
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            this.width = Math.max(1, width);
            this.height = Math.max(1, height);
            GLES20.glViewport(0, 0, width, height);
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
            ModelData current = model;
            if (current == null || current.count == 0 || program == 0) {
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
            float eyeX = current.centerX
                    + cameraDistance * cosPitch * (float) Math.sin(yawRad);
            float eyeY = current.centerY
                    + cameraDistance * (float) Math.sin(pitchRad);
            float eyeZ = current.centerZ
                    + cameraDistance * cosPitch * (float) Math.cos(yawRad);

            float[] view = new float[16];
            float[] projection = new float[16];
            float[] mvp = new float[16];
            Matrix.setLookAtM(view, 0,
                    eyeX, eyeY, eyeZ,
                    current.centerX, current.centerY, current.centerZ,
                    0f, 1f, 0f);
            float aspect = (float) width / (float) height;
            Matrix.perspectiveM(projection, 0, 60f, aspect,
                    Math.max(0.003f, current.radius * 0.01f),
                    Math.max(10f, current.radius * 30f));
            Matrix.multiplyMM(mvp, 0, projection, 0, view, 0);

            GLES20.glUseProgram(program);
            GLES20.glUniformMatrix4fv(mvpLocation, 1, false, mvp, 0);
            float pointScale = (float) (height / (2.0 * Math.tan(Math.toRadians(30.0))));
            GLES20.glUniform1f(pointScaleLocation, pointScale);

            current.positions.position(0);
            current.colors.position(0);
            current.sizes.position(0);
            GLES20.glEnableVertexAttribArray(positionLocation);
            GLES20.glVertexAttribPointer(positionLocation, 3, GLES20.GL_FLOAT,
                    false, 3 * Float.BYTES, current.positions);
            GLES20.glEnableVertexAttribArray(colorLocation);
            GLES20.glVertexAttribPointer(colorLocation, 4, GLES20.GL_FLOAT,
                    false, 4 * Float.BYTES, current.colors);
            GLES20.glEnableVertexAttribArray(sizeLocation);
            GLES20.glVertexAttribPointer(sizeLocation, 1, GLES20.GL_FLOAT,
                    false, Float.BYTES, current.sizes);
            GLES20.glDrawArrays(GLES20.GL_POINTS, 0, current.count);
            GLES20.glDisableVertexAttribArray(positionLocation);
            GLES20.glDisableVertexAttribArray(colorLocation);
            GLES20.glDisableVertexAttribArray(sizeLocation);
        }

        void setModel(ModelData model) {
            this.model = model;
            synchronized (this) {
                baseDistance = Math.max(model.radius * 2.35f, 0.18f);
                distance = baseDistance;
                yawDegrees = 0f;
                pitchDegrees = 0f;
            }
        }

        synchronized void rotate(float dxDegrees, float dyDegrees) {
            yawDegrees = (yawDegrees - dxDegrees) % 360f;
            pitchDegrees = Math.max(-85f, Math.min(85f, pitchDegrees - dyDegrees));
        }

        synchronized void zoom(float scaleFactor) {
            ModelData current = model;
            if (current == null || !Float.isFinite(scaleFactor) || scaleFactor <= 0f) {
                return;
            }
            distance /= scaleFactor;
            distance = Math.max(current.radius * 0.20f,
                    Math.min(current.radius * 18f, distance));
        }

        synchronized void resetCamera() {
            yawDegrees = 0f;
            pitchDegrees = 0f;
            distance = baseDistance;
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
        final int count;
        final FloatBuffer positions;
        final FloatBuffer colors;
        final FloatBuffer sizes;
        final float centerX;
        final float centerY;
        final float centerZ;
        final float radius;

        ModelData(
                int count,
                FloatBuffer positions,
                FloatBuffer colors,
                FloatBuffer sizes,
                float centerX,
                float centerY,
                float centerZ,
                float radius) {
            this.count = count;
            this.positions = positions;
            this.colors = colors;
            this.sizes = sizes;
            this.centerX = centerX;
            this.centerY = centerY;
            this.centerZ = centerZ;
            this.radius = radius;
        }
    }

    private static final class GaussianPlyReader {
        private static final float SH_C0 = 0.28209479177387814f;
        private static final int ROBUST_SAMPLE_LIMIT = 8192;

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

                FloatBuffer positions = directFloats(header.vertexCount * 3);
                FloatBuffer colors = directFloats(header.vertexCount * 4);
                FloatBuffer sizes = directFloats(header.vertexCount);

                int sampleStride = Math.max(1, header.vertexCount / ROBUST_SAMPLE_LIMIT);
                int sampleCapacity = Math.min(header.vertexCount,
                        (header.vertexCount + sampleStride - 1) / sampleStride);
                float[] sampleX = new float[sampleCapacity];
                float[] sampleY = new float[sampleCapacity];
                float[] sampleZ = new float[sampleCapacity];
                int sampleCount = 0;
                float[] values = new float[header.properties.size()];

                for (int i = 0; i < header.vertexCount; i++) {
                    for (int p = 0; p < values.length; p++) {
                        values[p] = body.getFloat();
                    }
                    float x = values[xIndex];
                    float y = values[yIndex];
                    float z = values[zIndex];
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) {
                        x = y = z = 0f;
                    }
                    positions.put(x).put(y).put(z);
                    if (i % sampleStride == 0 && sampleCount < sampleCapacity) {
                        sampleX[sampleCount] = x;
                        sampleY[sampleCount] = y;
                        sampleZ[sampleCount] = z;
                        sampleCount++;
                    }

                    float red = rIndex >= 0 ? 0.5f + SH_C0 * values[rIndex] : 0.75f;
                    float green = gIndex >= 0 ? 0.5f + SH_C0 * values[gIndex] : 0.75f;
                    float blue = bIndex >= 0 ? 0.5f + SH_C0 * values[bIndex] : 0.75f;
                    float alpha = opacityIndex >= 0 ? sigmoid(values[opacityIndex]) : 1f;
                    colors.put(clamp01(red)).put(clamp01(green)).put(clamp01(blue))
                            .put(Math.max(0.08f, clamp01(alpha)));

                    float worldScale = 0.012f;
                    if (sxIndex >= 0 && syIndex >= 0 && szIndex >= 0) {
                        float sx = (float) Math.exp(values[sxIndex]);
                        float sy = (float) Math.exp(values[syIndex]);
                        float sz = (float) Math.exp(values[szIndex]);
                        worldScale = (float) Math.cbrt(Math.max(1e-12f, sx * sy * sz)) * 2.4f;
                    }
                    sizes.put(Math.max(0.0025f, Math.min(0.14f, worldScale)));
                }
                positions.position(0);
                colors.position(0);
                sizes.position(0);

                if (sampleCount == 0) {
                    throw new IOException("PLY has no finite sample positions");
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
                DiagnosticLog.i("GaussianViewer",
                        String.format(Locale.US,
                                "Loaded model count=%d robustCenter=(%.3f,%.3f,%.3f) radius99=%.3f",
                                header.vertexCount, centerX, centerY, centerZ, radius));
                return new ModelData(header.vertexCount, positions, colors, sizes,
                        centerX, centerY, centerZ, radius);
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

        private static float sigmoid(float x) {
            if (x >= 0f) {
                float z = (float) Math.exp(-x);
                return 1f / (1f + z);
            }
            float z = (float) Math.exp(x);
            return z / (1f + z);
        }

        private static float clamp01(float value) {
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
