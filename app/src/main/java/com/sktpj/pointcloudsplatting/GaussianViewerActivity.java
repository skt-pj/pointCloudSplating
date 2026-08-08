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
import android.widget.SeekBar;
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
    private TextView sizeLabel;
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
        File splat = new File(new File(datasetPath), "splat.ply");
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

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setBackgroundColor(0x77000000);
        Button back = makeButton("戻る", "保存したスキャンに戻る", v -> finish(), 86);
        top.addView(back);
        statusView = new TextView(this);
        statusView.setTextColor(0xFFFFFFFF);
        statusView.setTextSize(13f);
        statusView.setGravity(Gravity.CENTER);
        statusView.setText("3Dモデルを読み込んでいます…\n少しお待ちください");
        top.addView(statusView, new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.MATCH_PARENT, 1f));
        Button reset = makeButton("正面に戻す", "3Dモデルの向きと大きさを元に戻す",
                v -> renderer.resetCamera(), 128);
        top.addView(reset);
        FrameLayout.LayoutParams topParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(64));
        topParams.gravity = Gravity.TOP;
        topParams.leftMargin = dp(6);
        topParams.rightMargin = dp(6);
        topParams.topMargin = dp(6);
        root.addView(top, topParams);

        LinearLayout sizePanel = new LinearLayout(this);
        sizePanel.setOrientation(LinearLayout.VERTICAL);
        sizePanel.setGravity(Gravity.CENTER_VERTICAL);
        sizePanel.setPadding(dp(16), dp(4), dp(16), dp(4));
        sizePanel.setBackgroundColor(0x99000000);
        sizeLabel = new TextView(this);
        sizeLabel.setTextColor(0xFFFFFFFF);
        sizeLabel.setTextSize(14f);
        sizeLabel.setText("表示サイズ: 最小");
        sizePanel.addView(sizeLabel, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(24)));
        SeekBar sizeSlider = new SeekBar(this);
        sizeSlider.setMax(100);
        sizeSlider.setProgress(0);
        sizeSlider.setMinHeight(dp(48));
        sizeSlider.setContentDescription("3Dモデルの表示サイズ");
        sizeSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                renderer.setDisplaySizeProgress(progress);
                if (progress == 0) {
                    sizeLabel.setText("表示サイズ: 最小");
                } else if (progress == 100) {
                    sizeLabel.setText("表示サイズ: 最大");
                } else {
                    sizeLabel.setText("表示サイズ: " + progress + "%");
                }
            }

            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        sizePanel.addView(sizeSlider, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));
        FrameLayout.LayoutParams sizeParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, dp(80));
        sizeParams.gravity = Gravity.BOTTOM;
        sizeParams.leftMargin = dp(8);
        sizeParams.rightMargin = dp(8);
        sizeParams.bottomMargin = dp(8);
        root.addView(sizePanel, sizeParams);

        setContentView(root);
        loadModelAsync(splat);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (glView != null) glView.onResume();
    }

    @Override
    protected void onPause() {
        if (glView != null) glView.onPause();
        super.onPause();
    }

    private Button makeButton(String text, String description, View.OnClickListener listener, int widthDp) {
        Button button = new Button(this);
        button.setText(text);
        button.setAllCaps(false);
        button.setContentDescription(description);
        button.setMinHeight(dp(48));
        button.setOnClickListener(listener);
        button.setLayoutParams(new LinearLayout.LayoutParams(dp(widthDp),
                LinearLayout.LayoutParams.MATCH_PARENT));
        return button;
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
                    Toast.makeText(this, "3Dモデルを表示できませんでした。", Toast.LENGTH_LONG).show();
                });
            }
        }, "LoadGaussianPly").start();
    }

    private boolean handleTouch(View view, MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        if (event.getPointerCount() == 1 && !scaleDetector.isInProgress()) {
            if (event.getActionMasked() == MotionEvent.ACTION_DOWN) {
                lastX = event.getX();
                lastY = event.getY();
            } else if (event.getActionMasked() == MotionEvent.ACTION_MOVE) {
                float x = event.getX();
                float y = event.getY();
                renderer.rotate((x - lastX) * 0.35f, (y - lastY) * 0.35f);
                lastX = x;
                lastY = y;
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

    private static final class GaussianRenderer implements GLSurfaceView.Renderer {
        private static final String TAG = "GaussianViewer";
        private static final float MIN_DISPLAY_SCALE = 0.08f;
        private static final float MAX_DISPLAY_SCALE = 4.0f;
        private static final String VERTEX_SHADER =
                "attribute vec3 a_Position;\n"
                        + "attribute vec4 a_DcAlpha;\n"
                        + "attribute vec3 a_ShR;\n"
                        + "attribute vec3 a_ShG;\n"
                        + "attribute vec3 a_ShB;\n"
                        + "attribute float a_Size;\n"
                        + "attribute vec2 a_Corner;\n"
                        + "uniform mat4 u_View;\n"
                        + "uniform mat4 u_Projection;\n"
                        + "uniform vec3 u_Center;\n"
                        + "uniform float u_Radius;\n"
                        + "uniform vec3 u_Camera;\n"
                        + "uniform float u_SizeScale;\n"
                        + "varying vec4 v_Color;\n"
                        + "varying vec2 v_Corner;\n"
                        + "void main() {\n"
                        + "  float radius = max(u_Radius, 0.0001);\n"
                        + "  vec3 p = (a_Position - u_Center) / radius;\n"
                        + "  vec3 d = normalize(p - u_Camera);\n"
                        + "  float c0 = 0.2820947918;\n"
                        + "  float c1 = 0.4886025119;\n"
                        + "  vec3 basis = vec3(-c1*d.y, c1*d.z, -c1*d.x);\n"
                        + "  vec3 rgb = vec3(\n"
                        + "    0.5 + c0*a_DcAlpha.r + dot(a_ShR, basis),\n"
                        + "    0.5 + c0*a_DcAlpha.g + dot(a_ShG, basis),\n"
                        + "    0.5 + c0*a_DcAlpha.b + dot(a_ShB, basis));\n"
                        + "  rgb = clamp(rgb, 0.0, 1.0);\n"
                        + "  vec4 viewCenter = u_View * vec4(p, 1.0);\n"
                        + "  float size = clamp((a_Size / radius) * 2.0 * u_SizeScale, 0.0008, 0.08);\n"
                        + "  vec4 viewPosition = viewCenter + vec4(a_Corner * size, 0.0, 0.0);\n"
                        + "  gl_Position = u_Projection * viewPosition;\n"
                        + "  v_Color = vec4(rgb, clamp(a_DcAlpha.a, 0.02, 0.98));\n"
                        + "  v_Corner = a_Corner;\n"
                        + "}\n";

        private static final String FRAGMENT_SHADER =
                "precision mediump float;\n"
                        + "varying vec4 v_Color;\n"
                        + "varying vec2 v_Corner;\n"
                        + "void main() {\n"
                        + "  float r2 = dot(v_Corner, v_Corner);\n"
                        + "  if (r2 > 1.0) discard;\n"
                        + "  float weight = exp(-3.0 * r2);\n"
                        + "  float alpha = v_Color.a * weight;\n"
                        + "  if (alpha < 0.004) discard;\n"
                        + "  gl_FragColor = vec4(v_Color.rgb, alpha);\n"
                        + "}\n";

        private final RendererStatusListener statusListener;
        private volatile ModelData model;
        private int program;
        private int positionLocation;
        private int dcAlphaLocation;
        private int shRLocation;
        private int shGLocation;
        private int shBLocation;
        private int sizeLocation;
        private int cornerLocation;
        private int viewLocation;
        private int projectionLocation;
        private int centerLocation;
        private int radiusLocation;
        private int cameraLocation;
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
                dcAlphaLocation = requireAttribute("a_DcAlpha");
                shRLocation = requireAttribute("a_ShR");
                shGLocation = requireAttribute("a_ShG");
                shBLocation = requireAttribute("a_ShB");
                sizeLocation = requireAttribute("a_Size");
                cornerLocation = requireAttribute("a_Corner");
                viewLocation = requireUniform("u_View");
                projectionLocation = requireUniform("u_Projection");
                centerLocation = requireUniform("u_Center");
                radiusLocation = requireUniform("u_Radius");
                cameraLocation = requireUniform("u_Camera");
                sizeScaleLocation = requireUniform("u_SizeScale");
                GLES20.glEnable(GLES20.GL_BLEND);
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
                GLES20.glEnable(GLES20.GL_DEPTH_TEST);
                GLES20.glDepthFunc(GLES20.GL_LEQUAL);
                GLES20.glDepthMask(true);
                DiagnosticLog.i(TAG,
                        "GL ready renderer=" + GLES20.glGetString(GLES20.GL_RENDERER)
                                + " version=" + GLES20.glGetString(GLES20.GL_VERSION)
                                + " vendor=" + GLES20.glGetString(GLES20.GL_VENDOR)
                                + " viewerAppearance=JPEG_SH1");
            } catch (RuntimeException e) {
                program = 0;
                DiagnosticLog.e(TAG, "Viewer GL initialization failed", e);
                statusListener.onStatus("3D表示を初期化できませんでした\n戻ってログをコピーしてください");
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
            if (current == null || current.vertexCount == 0 || program == 0) return;

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

            float[] view = new float[16];
            float[] projection = new float[16];
            Matrix.setLookAtM(view, 0, eyeX, eyeY, eyeZ, 0f, 0f, 0f, 0f, 1f, 0f);
            Matrix.perspectiveM(projection, 0, 55f, (float) width / (float) height, 0.05f, 30f);

            GLES20.glUseProgram(program);
            GLES20.glUniformMatrix4fv(viewLocation, 1, false, view, 0);
            GLES20.glUniformMatrix4fv(projectionLocation, 1, false, projection, 0);
            GLES20.glUniform3f(centerLocation, current.centerX, current.centerY, current.centerZ);
            GLES20.glUniform1f(radiusLocation, current.radius);
            GLES20.glUniform3f(cameraLocation, eyeX, eyeY, eyeZ);
            GLES20.glUniform1f(sizeScaleLocation, sizeScale);

            current.positions.position(0);
            current.dcAlpha.position(0);
            current.shR.position(0);
            current.shG.position(0);
            current.shB.position(0);
            current.sizes.position(0);
            current.corners.position(0);
            bind(positionLocation, 3, current.positions);
            bind(dcAlphaLocation, 4, current.dcAlpha);
            bind(shRLocation, 3, current.shR);
            bind(shGLocation, 3, current.shG);
            bind(shBLocation, 3, current.shB);
            bind(sizeLocation, 1, current.sizes);
            bind(cornerLocation, 2, current.corners);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, current.vertexCount);
            GLES20.glDisableVertexAttribArray(positionLocation);
            GLES20.glDisableVertexAttribArray(dcAlphaLocation);
            GLES20.glDisableVertexAttribArray(shRLocation);
            GLES20.glDisableVertexAttribArray(shGLocation);
            GLES20.glDisableVertexAttribArray(shBLocation);
            GLES20.glDisableVertexAttribArray(sizeLocation);
            GLES20.glDisableVertexAttribArray(cornerLocation);

            int error = GLES20.glGetError();
            if (error != GLES20.GL_NO_ERROR) {
                if (!drawErrorReported) {
                    drawErrorReported = true;
                    DiagnosticLog.e(TAG, "Viewer draw failed glError=0x" + Integer.toHexString(error));
                    statusListener.onStatus("3Dモデルを描画できませんでした\n戻ってログをコピーしてください");
                }
                return;
            }
            if (!firstDrawReported) {
                firstDrawReported = true;
                DiagnosticLog.i(TAG,
                        String.format(Locale.US,
                                "First model draw succeeded gaussians=%d triangles=%d radius=%.4f sizeScale=%.3f appearance=JPEG_SH1",
                                current.gaussianCount, current.vertexCount / 3, current.radius, sizeScale));
                statusListener.onStatus("3Dモデルを表示中\nドラッグで回転 / ピンチで拡大・縮小");
            }
        }

        private void bind(int location, int size, FloatBuffer buffer) {
            GLES20.glEnableVertexAttribArray(location);
            GLES20.glVertexAttribPointer(location, size, GLES20.GL_FLOAT,
                    false, size * Float.BYTES, buffer);
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

        synchronized void setDisplaySizeProgress(int progress) {
            float t = Math.max(0f, Math.min(1f, progress / 100f));
            displayScale = MIN_DISPLAY_SCALE
                    * (float) Math.pow(MAX_DISPLAY_SCALE / MIN_DISPLAY_SCALE, t);
        }

        synchronized void rotate(float dxDegrees, float dyDegrees) {
            yawDegrees = (yawDegrees - dxDegrees) % 360f;
            pitchDegrees = Math.max(-85f, Math.min(85f, pitchDegrees - dyDegrees));
        }

        synchronized void zoom(float scaleFactor) {
            if (model == null || !Float.isFinite(scaleFactor) || scaleFactor <= 0f) return;
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
            if (location < 0) throw new IllegalStateException("missing viewer attribute " + name);
            return location;
        }

        private int requireUniform(String name) {
            int location = GLES20.glGetUniformLocation(program, name);
            if (location < 0) throw new IllegalStateException("missing viewer uniform " + name);
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
        final FloatBuffer dcAlpha;
        final FloatBuffer shR;
        final FloatBuffer shG;
        final FloatBuffer shB;
        final FloatBuffer sizes;
        final FloatBuffer corners;
        final float centerX;
        final float centerY;
        final float centerZ;
        final float radius;

        ModelData(int gaussianCount, int vertexCount,
                  FloatBuffer positions, FloatBuffer dcAlpha,
                  FloatBuffer shR, FloatBuffer shG, FloatBuffer shB,
                  FloatBuffer sizes, FloatBuffer corners,
                  float centerX, float centerY, float centerZ, float radius) {
            this.gaussianCount = gaussianCount;
            this.vertexCount = vertexCount;
            this.positions = positions;
            this.dcAlpha = dcAlpha;
            this.shR = shR;
            this.shG = shG;
            this.shB = shB;
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
        private static final int ROBUST_SAMPLE_LIMIT = 8192;
        private static final float[] QUAD_CORNERS = {
                -1f, -1f, 1f, -1f, 1f, 1f,
                -1f, -1f, 1f, 1f, -1f, 1f
        };

        static ModelData read(File file) throws IOException {
            try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
                Header header = readHeader(input);
                if (!header.binaryLittleEndian || header.vertexCount <= 0) {
                    throw new IOException("invalid Gaussian PLY");
                }
                int xIndex = require(header, "x");
                int yIndex = require(header, "y");
                int zIndex = require(header, "z");
                int dcR = require(header, "f_dc_0");
                int dcG = require(header, "f_dc_1");
                int dcB = require(header, "f_dc_2");
                int opacityIndex = require(header, "opacity");
                int sxIndex = require(header, "scale_0");
                int syIndex = require(header, "scale_1");
                int szIndex = require(header, "scale_2");
                int r1 = header.indexOf("f_rest_0");
                int r2 = header.indexOf("f_rest_1");
                int r3 = header.indexOf("f_rest_2");
                int g1 = header.indexOf("f_rest_15");
                int g2 = header.indexOf("f_rest_16");
                int g3 = header.indexOf("f_rest_17");
                int b1 = header.indexOf("f_rest_30");
                int b2 = header.indexOf("f_rest_31");
                int b3 = header.indexOf("f_rest_32");

                long bodyBytes = (long) header.vertexCount * header.properties.size() * Float.BYTES;
                if (file.length() - header.bodyOffset < bodyBytes) {
                    throw new IOException("truncated splat.ply");
                }
                MappedByteBuffer body = input.getChannel().map(
                        FileChannel.MapMode.READ_ONLY, header.bodyOffset, bodyBytes);
                body.order(ByteOrder.LITTLE_ENDIAN);

                float[] xyz = new float[header.vertexCount * 3];
                float[] dc = new float[header.vertexCount * 3];
                float[] sh1 = new float[header.vertexCount * 9];
                float[] alpha = new float[header.vertexCount];
                float[] scale = new float[header.vertexCount];
                boolean[] valid = new boolean[header.vertexCount];
                float[] values = new float[header.properties.size()];
                int validCount = 0;
                int sh1Count = 0;
                for (int i = 0; i < header.vertexCount; i++) {
                    for (int p = 0; p < values.length; p++) values[p] = body.getFloat();
                    float x = values[xIndex];
                    float y = values[yIndex];
                    float z = values[zIndex];
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)) continue;
                    valid[i] = true;
                    validCount++;
                    xyz[i * 3] = x;
                    xyz[i * 3 + 1] = y;
                    xyz[i * 3 + 2] = z;
                    dc[i * 3] = finiteOrZero(values[dcR]);
                    dc[i * 3 + 1] = finiteOrZero(values[dcG]);
                    dc[i * 3 + 2] = finiteOrZero(values[dcB]);
                    int s = i * 9;
                    sh1[s] = valueAt(values, r1);
                    sh1[s + 1] = valueAt(values, r2);
                    sh1[s + 2] = valueAt(values, r3);
                    sh1[s + 3] = valueAt(values, g1);
                    sh1[s + 4] = valueAt(values, g2);
                    sh1[s + 5] = valueAt(values, g3);
                    sh1[s + 6] = valueAt(values, b1);
                    sh1[s + 7] = valueAt(values, b2);
                    sh1[s + 8] = valueAt(values, b3);
                    if (Math.abs(sh1[s]) + Math.abs(sh1[s + 1]) + Math.abs(sh1[s + 2])
                            + Math.abs(sh1[s + 3]) + Math.abs(sh1[s + 4]) + Math.abs(sh1[s + 5])
                            + Math.abs(sh1[s + 6]) + Math.abs(sh1[s + 7]) + Math.abs(sh1[s + 8]) > 1e-7f) {
                        sh1Count++;
                    }
                    alpha[i] = sigmoid(values[opacityIndex]);
                    float sx = safeExp(values[sxIndex]);
                    float sy = safeExp(values[syIndex]);
                    float sz = safeExp(values[szIndex]);
                    float product = sx * sy * sz;
                    float worldScale = Float.isFinite(product) && product > 0f
                            ? (float) Math.cbrt(product) : 0.008f;
                    scale[i] = Math.max(0.001f, Math.min(0.06f, worldScale));
                }
                if (validCount == 0) throw new IOException("PLY has no finite positions");

                float[] centerRadius = robustCenterRadius(xyz, valid, validCount);
                float centerX = centerRadius[0];
                float centerY = centerRadius[1];
                float centerZ = centerRadius[2];
                float radius = centerRadius[3];
                int vertexCount = validCount * 6;
                FloatBuffer positions = directFloats(vertexCount * 3);
                FloatBuffer dcAlpha = directFloats(vertexCount * 4);
                FloatBuffer shR = directFloats(vertexCount * 3);
                FloatBuffer shG = directFloats(vertexCount * 3);
                FloatBuffer shB = directFloats(vertexCount * 3);
                FloatBuffer sizes = directFloats(vertexCount);
                FloatBuffer corners = directFloats(vertexCount * 2);
                for (int i = 0; i < header.vertexCount; i++) {
                    if (!valid[i]) continue;
                    int p = i * 3;
                    int s = i * 9;
                    for (int v = 0; v < 6; v++) {
                        positions.put(xyz[p]).put(xyz[p + 1]).put(xyz[p + 2]);
                        dcAlpha.put(dc[p]).put(dc[p + 1]).put(dc[p + 2])
                                .put(Math.max(0.02f, Math.min(0.98f, alpha[i])));
                        shR.put(sh1[s]).put(sh1[s + 1]).put(sh1[s + 2]);
                        shG.put(sh1[s + 3]).put(sh1[s + 4]).put(sh1[s + 5]);
                        shB.put(sh1[s + 6]).put(sh1[s + 7]).put(sh1[s + 8]);
                        sizes.put(scale[i]);
                        corners.put(QUAD_CORNERS[v * 2]).put(QUAD_CORNERS[v * 2 + 1]);
                    }
                }
                positions.position(0);
                dcAlpha.position(0);
                shR.position(0);
                shG.position(0);
                shB.position(0);
                sizes.position(0);
                corners.position(0);
                DiagnosticLog.i(TAG,
                        String.format(Locale.US,
                                "Loaded model gaussians=%d/%d robustCenter=(%.3f,%.3f,%.3f) radius99=%.3f sh1=%d appearance=JPEG_SH1",
                                validCount, header.vertexCount, centerX, centerY, centerZ, radius, sh1Count));
                return new ModelData(validCount, vertexCount,
                        positions, dcAlpha, shR, shG, shB, sizes, corners,
                        centerX, centerY, centerZ, radius);
            }
        }

        private static float[] robustCenterRadius(float[] xyz, boolean[] valid, int validCount) {
            int stride = Math.max(1, validCount / ROBUST_SAMPLE_LIMIT);
            int capacity = Math.min(validCount, (validCount + stride - 1) / stride + 1);
            float[] xs = new float[capacity];
            float[] ys = new float[capacity];
            float[] zs = new float[capacity];
            int count = 0;
            int seen = 0;
            for (int i = 0; i < valid.length && count < capacity; i++) {
                if (!valid[i]) continue;
                if (seen % stride == 0) {
                    xs[count] = xyz[i * 3];
                    ys[count] = xyz[i * 3 + 1];
                    zs[count] = xyz[i * 3 + 2];
                    count++;
                }
                seen++;
            }
            float[] sx = Arrays.copyOf(xs, count);
            float[] sy = Arrays.copyOf(ys, count);
            float[] sz = Arrays.copyOf(zs, count);
            Arrays.sort(sx);
            Arrays.sort(sy);
            Arrays.sort(sz);
            float cx = sx[count / 2];
            float cy = sy[count / 2];
            float cz = sz[count / 2];
            float[] distances = new float[count];
            for (int i = 0; i < count; i++) {
                float dx = xs[i] - cx;
                float dy = ys[i] - cy;
                float dz = zs[i] - cz;
                distances[i] = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
            }
            Arrays.sort(distances);
            int index = Math.min(count - 1, Math.max(0, Math.round((count - 1) * 0.99f)));
            float radius = Math.max(0.05f, distances[index] * 1.12f);
            if (!Float.isFinite(radius) || radius <= 0f) radius = 1f;
            return new float[] {cx, cy, cz, radius};
        }

        private static Header readHeader(RandomAccessFile input) throws IOException {
            Header header = new Header();
            if (!"ply".equals(input.readLine())) throw new IOException("not a PLY file");
            boolean inVertex = false;
            String line;
            while ((line = input.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith("format ")) {
                    header.binaryLittleEndian = trimmed.startsWith("format binary_little_endian");
                } else if (trimmed.startsWith("element ")) {
                    String[] tokens = trimmed.split("\\s+");
                    inVertex = tokens.length >= 3 && "vertex".equals(tokens[1]);
                    if (inVertex) header.vertexCount = Integer.parseInt(tokens[2]);
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

        private static int require(Header header, String name) throws IOException {
            int index = header.indexOf(name);
            if (index < 0) throw new IOException("missing PLY property " + name);
            return index;
        }

        private static float valueAt(float[] values, int index) {
            return index >= 0 ? finiteOrZero(values[index]) : 0f;
        }

        private static float finiteOrZero(float value) {
            return Float.isFinite(value) ? value : 0f;
        }

        private static FloatBuffer directFloats(int count) {
            return ByteBuffer.allocateDirect(count * Float.BYTES)
                    .order(ByteOrder.nativeOrder()).asFloatBuffer();
        }

        private static float safeExp(float value) {
            if (!Float.isFinite(value)) return 0.008f;
            return (float) Math.exp(Math.max(-20f, Math.min(5f, value)));
        }

        private static float sigmoid(float x) {
            if (!Float.isFinite(x)) return 0.5f;
            if (x >= 0f) {
                float z = (float) Math.exp(-x);
                return 1f / (1f + z);
            }
            float z = (float) Math.exp(x);
            return z / (1f + z);
        }

        private static final class Header {
            boolean binaryLittleEndian;
            int vertexCount;
            long bodyOffset;
            final List<String> properties = new ArrayList<>();
            int indexOf(String name) { return properties.indexOf(name); }
        }
    }
}
