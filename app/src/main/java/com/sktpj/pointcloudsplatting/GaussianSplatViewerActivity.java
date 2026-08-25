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
import android.widget.LinearLayout;
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

/** Robust phone-side viewer for the app-generated splat.ply. */
public final class GaussianSplatViewerActivity extends Activity {
    public static final String EXTRA_DATASET_PATH = "dataset_path";
    private static final String TAG = "GaussianSplatViewer";

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

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setBackgroundColor(0x88000000);

        Button back = new Button(this);
        back.setText("戻る");
        back.setOnClickListener(v -> finish());
        top.addView(back, new LinearLayout.LayoutParams(dp(88), dp(56)));

        statusView = new TextView(this);
        statusView.setTextColor(0xFFFFFFFF);
        statusView.setTextSize(13f);
        statusView.setGravity(Gravity.CENTER);
        statusView.setText("3DGSを読み込んでいます...\nドラッグ: 回転 / ピンチ: ズーム");
        top.addView(statusView, new LinearLayout.LayoutParams(0, dp(56), 1f));

        Button reset = new Button(this);
        reset.setText("リセット");
        reset.setOnClickListener(v -> renderer.resetCamera());
        top.addView(reset, new LinearLayout.LayoutParams(dp(96), dp(56)));

        FrameLayout.LayoutParams topParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(56));
        topParams.gravity = Gravity.TOP;
        root.addView(top, topParams);

        TextView title = new TextView(this);
        title.setText(dataset.getName());
        title.setTextColor(0xFFFFFFFF);
        title.setTextSize(13f);
        title.setGravity(Gravity.CENTER);
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
                DiagnosticLog.i(TAG,
                        String.format(Locale.US,
                                "Loaded splat gaussians=%d fitRadius=%.4fm center=(%.3f,%.3f,%.3f) outliers=%d",
                                model.count,
                                model.sourceRadius,
                                model.sourceCenterX,
                                model.sourceCenterY,
                                model.sourceCenterZ,
                                model.outlierCount));
                runOnUiThread(() -> statusView.setText(
                        String.format(Locale.US,
                                "%,d Gaussians / fit %.2fm\nドラッグ: 回転 / ピンチ: ズーム",
                                model.count, model.sourceRadius)));
            } catch (Exception e) {
                DiagnosticLog.e(TAG, "Failed to load splat.ply", e);
                runOnUiThread(() -> {
                    statusView.setText("3DGS読込失敗: " + e.getMessage());
                    Toast.makeText(this,
                            "3DGSを表示できません: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }, "LoadGaussianSplat").start();
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
                    renderer.rotate((x - lastX) * 0.30f, (y - lastY) * 0.30f);
                    lastX = x;
                    lastY = y;
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

    private static final class GaussianRenderer implements GLSurfaceView.Renderer {
        private static final String VERTEX_SHADER =
                "attribute vec3 a_Position;\n"
                        + "attribute vec4 a_Color;\n"
                        + "attribute float a_Size;\n"
                        + "uniform mat4 u_Mvp;\n"
                        + "uniform float u_PointScale;\n"
                        + "varying vec4 v_Color;\n"
                        + "void main() {\n"
                        + "  gl_Position = u_Mvp * vec4(a_Position, 1.0);\n"
                        + "  gl_PointSize = clamp(max(3.0, a_Size * u_PointScale), 3.0, 52.0);\n"
                        + "  v_Color = v_Color = a_Color;\n"
                        + "}\n";

        private static final String FRAGMENT_SHADER =
                "precision mediump float;\n"
                        + "varying vec4 v_Color;\n"
                        + "void main() {\n"
                        + "  vec2 p = gl_PointCoord * 2.0 - 1.0;\n"
                        + "  float r2 = dot(p, p);\n"
                        + "  if (r2 > 1.0) discard;\n"
                        + "  float g = exp(-2.5 * r2);\n"
                        + "  gl_FragColor = vec4(v_Color.rgb, max(0.18, v_Color.a * g));\n"
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
        private float yawDegrees = -20f;
        private float pitchDegrees = 12f;
        private float viewScale = 1f;
        private boolean loggedFirstDraw;

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            GLES20.glClearColor(0.035f, 0.035f, 0.040f, 1f);
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

            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glDepthMask(false);
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
            DiagnosticLog.i(TAG,
                    "GL ready renderer=" + GLES20.glGetString(GLES20.GL_RENDERER)
                            + " version=" + GLES20.glGetString(GLES20.GL_VERSION));
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            this.width = Math.max(1, width);
            this.height = Math.max(1, height);
            GLES20.glViewport(0, 0, width, height);
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
            ModelData current = model;
            if (current == null || current.count == 0 || program == 0) {
                return;
            }

            float yaw;
            float pitch;
            float scale;
            synchronized (this) {
                yaw = yawDegrees;
                pitch = pitchDegrees;
                scale = viewScale;
            }

            float aspect = (float) width / (float) height;
            float extent = 1.15f * scale;
            float halfWidth;
            float halfHeight;
            if (aspect >= 1f) {
                halfHeight = extent;
                halfWidth = extent * aspect;
            } else {
                halfWidth = extent;
                halfHeight = extent / Math.max(0.01f, aspect);
            }

            float[] modelMatrix = new float[16];
            float[] view = new float[16];
            float[] projection = new float[16];
            float[] viewModel = new float[16];
            float[] mvp = new float[16];
            Matrix.setIdentityM(modelMatrix, 0);
            Matrix.rotateM(modelMatrix, 0, pitch, 1f, 0f, 0f);
            Matrix.rotateM(modelMatrix, 0, yaw, 0f, 1f, 0f);
            Matrix.setLookAtM(view, 0,
                    0f, 0f, 3f,
                    0f, 0f, 0f,
                    0f, 1f, 0f);
            Matrix.orthoM(projection, 0,
                    -halfWidth, halfWidth, -halfHeight, halfHeight,
                    0.1f, 20f);
            Matrix.multiplyMM(viewModel, 0, view, 0, modelMatrix, 0);
            Matrix.multiplyMM(mvp, 0, projection, 0, viewModel, 0);

            GLES20.glUseProgram(program);
            GLES20.glUniformMatrix4fv(mvpLocation, 1, false, mvp, 0);
            float pointScale = Math.min(width, height) * 2.2f / Math.max(0.15f, scale);
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
            int error = GLES20.glGetError();
            if (!loggedFirstDraw) {
                loggedFirstDraw = true;
                DiagnosticLog.i(TAG,
                        "First draw count=" + current.count
                                + " viewport=" + width + "x" + height
                                + " glError=0x" + Integer.toHexString(error));
            } else if (error != GLES20.GL_NO_ERROR) {
                DiagnosticLog.w(TAG,
                        "GL draw error=0x" + Integer.toHexString(error));
            }
            GLES20.glDisableVertexAttribArray(positionLocation);
            GLES20.glDisableVertexAttribArray(colorLocation);
            GLES20.glDisableVertexAttribArray(sizeLocation);
        }

        void setModel(ModelData model) {
            this.model = model;
            synchronized (this) {
                yawDegrees = -20f;
                pitchDegrees = 12f;
                viewScale = 1f;
                loggedFirstDraw = false;
            }
        }

        synchronized void rotate(float dxDegrees, float dyDegrees) {
            yawDegrees = (yawDegrees - dxDegrees) % 360f;
            pitchDegrees = Math.max(-85f, Math.min(85f, pitchDegrees - dyDegrees));
        }

        synchronized void zoom(float gestureScale) {
            if (!Float.isFinite(gestureScale) || gestureScale <= 0f) {
                return;
            }
            viewScale /= gestureScale;
            viewScale = Math.max(0.15f, Math.min(8f, viewScale));
        }

        synchronized void resetCamera() {
            yawDegrees = -20f;
            pitchDegrees = 12f;
            viewScale = 1f;
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
        final float sourceCenterX;
        final float sourceCenterY;
        final float sourceCenterZ;
        final float sourceRadius;
        final int outlierCount;

        ModelData(
                int count,
                FloatBuffer positions,
                FloatBuffer colors,
                FloatBuffer sizes,
                float sourceCenterX,
                float sourceCenterY,
                float sourceCenterZ,
                float sourceRadius,
                int outlierCount) {
            this.count = count;
            this.positions = positions;
            this.colors = colors;
            this.sizes = sizes;
            this.sourceCenterX = sourceCenterX;
            this.sourceCenterY = sourceCenterY;
            this.sourceCenterZ = sourceCenterZ;
            this.sourceRadius = sourceRadius;
            this.outlierCount = outlierCount;
        }
    }

    private static final class GaussianPlyReader {
        private static final float SH_C0 = 0.28209479177387814f;

        static ModelData read(File file) throws IOException {
            try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
                Header header = readHeader(input);
                if (!header.binaryLittleEndian) {
                    throw new IOException("binary_little_endian PLY only");
                }
                if (header.vertexCount <= 0) {
                    throw new IOException("PLY has no vertices");
                }
                int xIndex = required(header, "x");
                int yIndex = required(header, "y");
                int zIndex = required(header, "z");
                int rIndex = header.indexOf("f_dc_0");
                int gIndex = header.indexOf("f_dc_1");
                int bIndex = header.indexOf("f_dc_2");
                int opacityIndex = header.indexOf("opacity");
                int sxIndex = header.indexOf("scale_0");
                int syIndex = header.indexOf("scale_1");
                int szIndex = header.indexOf("scale_2");

                int propertyCount = header.properties.size();
                long bodyBytes = (long) header.vertexCount * propertyCount * Float.BYTES;
                long available = file.length() - header.bodyOffset;
                if (available < bodyBytes) {
                    throw new IOException("truncated splat.ply");
                }
                MappedByteBuffer body = input.getChannel().map(
                        FileChannel.MapMode.READ_ONLY, header.bodyOffset, bodyBytes);
                body.order(ByteOrder.LITTLE_ENDIAN);

                int count = header.vertexCount;
                float[] xyz = new float[count * 3];
                float[] rgba = new float[count * 4];
                float[] worldSizes = new float[count];
                float[] values = new float[propertyCount];
                double sumX = 0.0;
                double sumY = 0.0;
                double sumZ = 0.0;
                int finiteCount = 0;

                for (int i = 0; i < count; i++) {
                    for (int p = 0; p < propertyCount; p++) {
                        values[p] = body.getFloat();
                    }
                    float x = values[xIndex];
                    float y = values[yIndex];
                    float z = values[zIndex];
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) {
                        x = y = z = 0f;
                    } else {
                        sumX += x;
                        sumY += y;
                        sumZ += z;
                        finiteCount++;
                    }
                    int pi = i * 3;
                    xyz[pi] = x;
                    xyz[pi + 1] = y;
                    xyz[pi + 2] = z;

                    float red = rIndex >= 0 ? 0.5f + SH_C0 * values[rIndex] : 0.8f;
                    float green = gIndex >= 0 ? 0.5f + SH_C0 * values[gIndex] : 0.8f;
                    float blue = bIndex >= 0 ? 0.5f + SH_C0 * values[bIndex] : 0.8f;
                    float alpha = opacityIndex >= 0 ? sigmoid(values[opacityIndex]) : 1f;
                    int ci = i * 4;
                    rgba[ci] = clamp01(red);
                    rgba[ci + 1] = clamp01(green);
                    rgba[ci + 2] = clamp01(blue);
                    rgba[ci + 3] = Math.max(0.35f, clamp01(alpha));

                    float worldScale = 0.012f;
                    if (sxIndex >= 0 && syIndex >= 0 && szIndex >= 0) {
                        float averageLogScale =
                                (values[sxIndex] + values[syIndex] + values[szIndex]) / 3f;
                        if (Float.isFinite(averageLogScale)) {
                            worldScale = (float) Math.exp(averageLogScale) * 2.2f;
                        }
                    }
                    worldSizes[i] = Math.max(0.002f, Math.min(0.12f, worldScale));
                }

                if (finiteCount == 0) {
                    throw new IOException("PLY has no finite positions");
                }
                float centerX = (float) (sumX / finiteCount);
                float centerY = (float) (sumY / finiteCount);
                float centerZ = (float) (sumZ / finiteCount);

                float[] distances = new float[count];
                for (int i = 0; i < count; i++) {
                    int pi = i * 3;
                    float dx = xyz[pi] - centerX;
                    float dy = xyz[pi + 1] - centerY;
                    float dz = xyz[pi + 2] - centerZ;
                    distances[i] = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
                }
                float[] sortedDistances = distances.clone();
                Arrays.sort(sortedDistances);
                int percentileIndex = Math.min(count - 1,
                        Math.max(0, (int) Math.floor((count - 1) * 0.99)));
                float fitRadius = Math.max(0.05f, sortedDistances[percentileIndex]);
                if (!Float.isFinite(fitRadius)) {
                    throw new IOException("invalid Gaussian bounds");
                }

                FloatBuffer positions = directFloats(count * 3);
                FloatBuffer colors = directFloats(count * 4);
                FloatBuffer sizes = directFloats(count);
                int outliers = 0;
                for (int i = 0; i < count; i++) {
                    int pi = i * 3;
                    float nx = (xyz[pi] - centerX) / fitRadius;
                    float ny = (xyz[pi + 1] - centerY) / fitRadius;
                    float nz = (xyz[pi + 2] - centerZ) / fitRadius;
                    float length = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
                    if (length > 1.25f) {
                        outliers++;
                        float inv = 1.25f / Math.max(length, 1e-6f);
                        nx *= inv;
                        ny *= inv;
                        nz *= inv;
                    }
                    positions.put(nx).put(ny).put(nz);
                    int ci = i * 4;
                    colors.put(rgba[ci]).put(rgba[ci + 1]).put(rgba[ci + 2]).put(rgba[ci + 3]);
                    float normalizedSize = worldSizes[i] / fitRadius;
                    sizes.put(Math.max(0.0025f, Math.min(0.08f, normalizedSize)));
                }
                positions.position(0);
                colors.position(0);
                sizes.position(0);
                return new ModelData(count, positions, colors, sizes,
                        centerX, centerY, centerZ, fitRadius, outliers);
            }
        }

        private static int required(Header header, String name) throws IOException {
            int index = header.indexOf(name);
            if (index < 0) {
                throw new IOException("PLY property missing: " + name);
            }
            return index;
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
