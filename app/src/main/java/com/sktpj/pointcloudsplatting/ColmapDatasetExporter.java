package com.sktpj.pointcloudsplatting;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapRegionDecoder;
import android.graphics.Rect;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Converts the app's saved camera dataset into the COLMAP camera format consumed by VkSplat. */
public final class ColmapDatasetExporter {
    private static final String TAG = "ColmapExport";
    private static final int MAX_TRAIN_LONG_EDGE = 1000;
    private static final int PHASE3_LOW_LONG_EDGE = 720;
    private static final int PHASE3_HIGH_PATCH_WIDTH = 1280;
    private static final int PHASE3_HIGH_PATCH_HEIGHT = 960;
    private static final String PHASE3_STATE_FILE = "phase3_training_state.json";
    private static final float VOXEL_METERS = 0.008f;
    private static final int MAX_INITIAL_POINTS = 220_000;
    private static final int VOXEL_BITS = 21;
    private static final int VOXEL_OFFSET = 1 << 20;
    private static final long VOXEL_MASK = (1L << VOXEL_BITS) - 1L;

    private ColmapDatasetExporter() {}

    public interface ProgressListener {
        void onProgress(int percent, String message);
    }

    public static final class Result {
        public final boolean success;
        public final String message;
        public final File root;
        public final File imageDir;
        public final File sparseDir;
        public final int frameCount;
        public final int initialPointCount;

        private Result(boolean success, String message, File root, File imageDir, File sparseDir,
                int frameCount, int initialPointCount) {
            this.success = success;
            this.message = message;
            this.root = root;
            this.imageDir = imageDir;
            this.sparseDir = sparseDir;
            this.frameCount = frameCount;
            this.initialPointCount = initialPointCount;
        }

        static Result fail(String message) {
            return new Result(false, message, null, null, null, 0, 0);
        }
    }

    public static Result prepare(File dataset, ProgressListener listener) {
        if (dataset == null || !dataset.isDirectory()) return Result.fail("dataset unavailable");
        File transformsFile = new File(dataset, "transforms.json");
        if (!transformsFile.isFile()) return Result.fail("transforms.json missing");

        try {
            JSONObject transforms = new JSONObject(readText(transformsFile));
            JSONArray frames = transforms.getJSONArray("frames");
            if (frames.length() < 2) return Result.fail("3DGS training needs at least two saved views");

            int phase3Stage = readPhase3Stage(dataset);
            File root = new File(dataset, "vksplat_data");
            File imageDir = new File(root, "images_4");
            File sparseDir = new File(root, "sparse/0");
            recreateDirectory(imageDir);
            recreateDirectory(sparseDir);

            notifyProgress(listener, 3, "学習用の写真を準備しています…");
            File camerasFile = new File(sparseDir, "cameras.txt");
            File imagesFile = new File(sparseDir, "images.txt");
            try (BufferedWriter cameras = new BufferedWriter(new FileWriter(camerasFile));
                 BufferedWriter images = new BufferedWriter(new FileWriter(imagesFile))) {
                cameras.write("# pointCloudSplating saved ARCore CPU-frame intrinsics\n");
                images.write("# Saved ARCore camera poses converted to COLMAP world-to-camera\n");
                for (int i = 0; i < frames.length(); i++) {
                    JSONObject frame = frames.getJSONObject(i);
                    File source = new File(dataset, frame.getString("file_path"));
                    if (!source.isFile()) throw new IOException("missing RGB image " + source.getName());

                    int sourceW = frame.getInt("w");
                    int sourceH = frame.getInt("h");
                    File target = new File(imageDir, source.getName());
                    int targetW;
                    int targetH;
                    double fx;
                    double fy;
                    double cx;
                    double cy;
                    if (phase3Stage >= 3) {
                        targetW = Math.min(sourceW, PHASE3_HIGH_PATCH_WIDTH);
                        targetH = Math.min(sourceH, PHASE3_HIGH_PATCH_HEIGHT);
                        int left = clampInt(
                                (int) Math.round(frame.getDouble("cx") - targetW * 0.5),
                                0, Math.max(0, sourceW - targetW));
                        int top = clampInt(
                                (int) Math.round(frame.getDouble("cy") - targetH * 0.5),
                                0, Math.max(0, sourceH - targetH));
                        cropJpeg(source, target, new Rect(left, top, left + targetW, top + targetH));
                        fx = frame.getDouble("fl_x");
                        fy = frame.getDouble("fl_y");
                        cx = frame.getDouble("cx") - left;
                        cy = frame.getDouble("cy") - top;
                    } else {
                        int longEdge = phase3Stage == 1 ? PHASE3_LOW_LONG_EDGE : MAX_TRAIN_LONG_EDGE;
                        double trainingScale = Math.min(
                                1.0, (double) longEdge / Math.max(sourceW, sourceH));
                        targetW = Math.max(1, (int) Math.round(sourceW * trainingScale));
                        targetH = Math.max(1, (int) Math.round(sourceH * trainingScale));
                        resizeJpeg(source, target, targetW, targetH);
                        double sx = (double) targetW / sourceW;
                        double sy = (double) targetH / sourceH;
                        fx = frame.getDouble("fl_x") * sx;
                        fy = frame.getDouble("fl_y") * sy;
                        cx = frame.getDouble("cx") * sx;
                        cy = frame.getDouble("cy") * sy;
                    }
                    int id = i + 1;
                    cameras.write(String.format(Locale.US,
                            "%d PINHOLE %d %d %.10f %.10f %.10f %.10f\n",
                            id, targetW, targetH, fx, fy, cx, cy));

                    double[][] c2wOpenGl = parseMatrix(frame.getJSONArray("transform_matrix"));
                    CameraExtrinsics extrinsics = toColmapWorldToCamera(c2wOpenGl);
                    images.write(String.format(Locale.US,
                            "%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %d %s\n\n",
                            id,
                            extrinsics.qw, extrinsics.qx, extrinsics.qy, extrinsics.qz,
                            extrinsics.tx, extrinsics.ty, extrinsics.tz,
                            id, target.getName()));
                    notifyProgress(listener,
                            3 + Math.round(34f * (i + 1) / frames.length()),
                            "学習用の写真を準備しています… " + (i + 1) + "/" + frames.length());
                }
            }

            notifyProgress(listener, 39, "3Dの初期位置をまとめています…");
            File phase2Geometry = new File(dataset, "phase2_geometry_prior.ply");
            if (!Phase2DatasetEvaluator.hasStoredPass(dataset)
                    || !phase2Geometry.isFile() || phase2Geometry.length() <= 0L) {
                return Result.fail("Phase 2の3D形状検証が完了していません");
            }
            File[] depthFiles = {phase2Geometry};
            DiagnosticLog.i(TAG, "Phase 3 geometry source=phase2_geometry_prior.ply");

            Map<Long, PointAccumulator> points = collectDepthPoints(depthFiles, 0.05f);
            if (points.size() < 256) {
                DiagnosticLog.w(TAG, "Low-confidence depth fallback: accepted=" + points.size());
                points = collectDepthPoints(depthFiles, 0.0f);
            }
            if (points.size() < 64) {
                return Result.fail("3Dの初期位置を作れるDepthデータが不足しています");
            }

            File pointsFile = new File(sparseDir, "points3D.txt");
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(pointsFile))) {
                writer.write("# Dense ARCore Raw Depth prior. RGB will be optimized from saved photographs.\n");
                long id = 1;
                for (PointAccumulator point : points.values()) {
                    double inv = 1.0 / Math.max(1e-9, point.weight);
                    int r = clamp255((int) Math.round(point.r * inv));
                    int g = clamp255((int) Math.round(point.g * inv));
                    int b = clamp255((int) Math.round(point.b * inv));
                    writer.write(String.format(Locale.US,
                            "%d %.7f %.7f %.7f %d %d %d 0.0\n",
                            id++, point.x * inv, point.y * inv, point.z * inv, r, g, b));
                }
            }

            JSONObject meta = new JSONObject();
            meta.put("format_version", 2);
            meta.put("source", "pointCloudSplating continuous ARCore CPU RGB + ARCore Raw Depth");
            meta.put("frame_count", frames.length());
            meta.put("initial_point_count", points.size());
            meta.put("phase3_stage", phase3Stage);
            meta.put("geometry_source", "phase2_geometry_prior.ply");
            meta.put("training_image_policy", phase3Stage >= 3
                    ? "native-resolution principal-point crop up to 1280x960; never upscale"
                    : "preserve aspect ratio; max long edge "
                            + (phase3Stage == 1 ? PHASE3_LOW_LONG_EDGE : MAX_TRAIN_LONG_EDGE)
                            + " px; never upscale");
            meta.put("training_image_dir", "images_4");
            meta.put("coordinate_system", "datasetRootAnchor; OpenGL c2w converted to COLMAP w2c");
            writeText(new File(root, "export.json"), meta.toString(2));

            DiagnosticLog.i(TAG, "VkSplat dataset ready frames=" + frames.length()
                    + " initialPoints=" + points.size() + " images=" + imageDir.getAbsolutePath());
            notifyProgress(listener, 45, "3DGS学習データを準備しました");
            return new Result(true, "ready", root, imageDir, sparseDir,
                    frames.length(), points.size());
        } catch (Exception e) {
            DiagnosticLog.e(TAG, "VkSplat dataset export failed", e);
            return Result.fail("3DGS学習データの準備に失敗しました: " + e.getClass().getSimpleName());
        }
    }

    private static Map<Long, PointAccumulator> collectDepthPoints(File[] files, float minConfidence)
            throws IOException {
        Map<Long, PointAccumulator> out = new HashMap<>(Math.min(MAX_INITIAL_POINTS * 2, 400_000));
        for (File file : files) {
            PlyHeader h = readPlyHeader(file);
            if (h.xOffset < 0 || h.yOffset < 0 || h.zOffset < 0) continue;
            try (FileInputStream input = new FileInputStream(file);
                 FileChannel channel = input.getChannel()) {
                long bodyBytes = (long) h.vertexCount * h.recordBytes;
                if (bodyBytes <= 0 || file.length() < h.headerBytes + bodyBytes) continue;
                MappedByteBuffer body = channel.map(FileChannel.MapMode.READ_ONLY, h.headerBytes, bodyBytes);
                body.order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < h.vertexCount; i++) {
                    int base = i * h.recordBytes;
                    float x = body.getFloat(base + h.xOffset);
                    float y = body.getFloat(base + h.yOffset);
                    float z = body.getFloat(base + h.zOffset);
                    float confidence = h.confidenceOffset >= 0
                            ? body.getFloat(base + h.confidenceOffset) : 1f;
                    if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z)
                            || !Float.isFinite(confidence) || confidence < minConfidence) continue;
                    float d2 = x * x + y * y + z * z;
                    if (d2 < 0.0025f || d2 > 400f) continue;

                    int qx = fastFloor(x / VOXEL_METERS);
                    int qy = fastFloor(y / VOXEL_METERS);
                    int qz = fastFloor(z / VOXEL_METERS);
                    if (Math.abs(qx) >= VOXEL_OFFSET || Math.abs(qy) >= VOXEL_OFFSET
                            || Math.abs(qz) >= VOXEL_OFFSET) continue;
                    long key = pack(qx, qy, qz);
                    PointAccumulator p = out.get(key);
                    if (p == null) {
                        if (out.size() >= MAX_INITIAL_POINTS) continue;
                        p = new PointAccumulator();
                        out.put(key, p);
                    }
                    int r = h.redOffset >= 0 ? body.get(base + h.redOffset) & 0xff : 128;
                    int g = h.greenOffset >= 0 ? body.get(base + h.greenOffset) & 0xff : 128;
                    int b = h.blueOffset >= 0 ? body.get(base + h.blueOffset) & 0xff : 128;
                    double w = Math.max(0.05, confidence);
                    p.x += x * w; p.y += y * w; p.z += z * w;
                    p.r += r * w; p.g += g * w; p.b += b * w; p.weight += w;
                }
            }
        }
        return out;
    }

    private static PlyHeader readPlyHeader(File file) throws IOException {
        try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
            ByteArrayOutputStream raw = new ByteArrayOutputStream();
            List<String> lines = new ArrayList<>();
            StringBuilder line = new StringBuilder();
            int c;
            while ((c = input.read()) != -1) {
                raw.write(c);
                if (c == '\n') {
                    String s = line.toString().trim();
                    lines.add(s);
                    line.setLength(0);
                    if ("end_header".equals(s)) break;
                } else if (c != '\r') line.append((char) c);
                if (raw.size() > 64 * 1024) throw new IOException("PLY header too large");
            }
            int vertices = -1;
            boolean binary = false;
            boolean inVertex = false;
            int offset = 0;
            int x=-1,y=-1,z=-1,r=-1,g=-1,b=-1,conf=-1;
            for (String s : lines) {
                if ("format binary_little_endian 1.0".equals(s)) binary = true;
                else if (s.startsWith("element ")) {
                    String[] p = s.split("\\s+");
                    inVertex = p.length == 3 && "vertex".equals(p[1]);
                    if (inVertex) { vertices = Integer.parseInt(p[2]); offset = 0; }
                } else if (inVertex && s.startsWith("property ")) {
                    String[] p = s.split("\\s+");
                    if (p.length != 3 || "list".equals(p[1])) throw new IOException("unsupported PLY property");
                    int size = typeSize(p[1]);
                    switch (p[2]) {
                        case "x": x=offset; break; case "y": y=offset; break; case "z": z=offset; break;
                        case "red": r=offset; break; case "green": g=offset; break; case "blue": b=offset; break;
                        case "confidence": conf=offset; break;
                    }
                    offset += size;
                }
            }
            if (!binary || vertices < 0 || offset <= 0) throw new IOException("unsupported PLY " + file.getName());
            return new PlyHeader(raw.size(), vertices, offset, x,y,z,r,g,b,conf);
        }
    }

    private static int typeSize(String type) throws IOException {
        switch (type) {
            case "char": case "uchar": case "int8": case "uint8": return 1;
            case "short": case "ushort": case "int16": case "uint16": return 2;
            case "int": case "uint": case "float": case "int32": case "uint32": case "float32": return 4;
            case "double": case "float64": case "int64": case "uint64": return 8;
            default: throw new IOException("unsupported PLY type " + type);
        }
    }

    private static int readPhase3Stage(File dataset) {
        File state = new File(dataset, PHASE3_STATE_FILE);
        if (!state.isFile()) return 0;
        try {
            return new JSONObject(readText(state)).optInt("current_stage", 0);
        } catch (Exception ignored) {
            return 0;
        }
    }

    private static int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static void cropJpeg(File source, File target, Rect region) throws IOException {
        BitmapRegionDecoder decoder = null;
        Bitmap cropped = null;
        try {
            decoder = BitmapRegionDecoder.newInstance(source.getAbsolutePath(), false);
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            options.inScaled = false;
            cropped = decoder.decodeRegion(region, options);
            if (cropped == null) throw new IOException("region decode failed " + source.getName());
            try (FileOutputStream output = new FileOutputStream(target)) {
                if (!cropped.compress(Bitmap.CompressFormat.JPEG, 96, output)) {
                    throw new IOException("crop JPEG compression failed " + source.getName());
                }
            }
        } finally {
            if (cropped != null) cropped.recycle();
            if (decoder != null) decoder.recycle();
        }
    }

    private static void resizeJpeg(File source, File target, int width, int height) throws IOException {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(source.getAbsolutePath(), bounds);
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("decode bounds failed " + source.getName());
        }

        int sample = 1;
        while (bounds.outWidth / (sample * 2) >= width
                && bounds.outHeight / (sample * 2) >= height) {
            sample *= 2;
        }

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inScaled = false;
        options.inSampleSize = Math.max(1, sample);
        Bitmap decoded = BitmapFactory.decodeFile(source.getAbsolutePath(), options);
        if (decoded == null) throw new IOException("decode failed " + source.getName());
        Bitmap scaled = decoded;
        try {
            if (decoded.getWidth() != width || decoded.getHeight() != height) {
                scaled = Bitmap.createScaledBitmap(decoded, width, height, true);
            }
            try (FileOutputStream output = new FileOutputStream(target)) {
                if (!scaled.compress(Bitmap.CompressFormat.JPEG, 95, output)) {
                    throw new IOException("JPEG encode failed " + target.getName());
                }
            }
        } finally {
            if (scaled != decoded) scaled.recycle();
            decoded.recycle();
        }
        DiagnosticLog.i(TAG, "Prepared training JPEG " + source.getName()
                + " source=" + bounds.outWidth + "x" + bounds.outHeight
                + " sample=" + sample + " output=" + width + "x" + height);
    }

    private static double[][] parseMatrix(JSONArray rows) throws Exception {
        if (rows.length() != 4) throw new IOException("camera matrix is not 4x4");
        double[][] m = new double[4][4];
        for (int r=0;r<4;r++) {
            JSONArray row = rows.getJSONArray(r);
            if (row.length()!=4) throw new IOException("camera matrix is not 4x4");
            for (int c=0;c<4;c++) m[r][c]=row.getDouble(c);
        }
        return m;
    }

    private static CameraExtrinsics toColmapWorldToCamera(double[][] c2w) {
        // Saved poses use OpenGL camera axes (+X right, +Y up, -Z forward). COLMAP uses
        // +X right, +Y down, +Z forward. Convert camera basis by negating columns Y and Z.
        double[][] rc = new double[3][3];
        for (int r=0;r<3;r++) {
            rc[r][0] = c2w[r][0];
            rc[r][1] = -c2w[r][1];
            rc[r][2] = -c2w[r][2];
        }
        double[][] rw = transpose3(rc);
        double cx=c2w[0][3], cy=c2w[1][3], cz=c2w[2][3];
        double tx = -(rw[0][0]*cx + rw[0][1]*cy + rw[0][2]*cz);
        double ty = -(rw[1][0]*cx + rw[1][1]*cy + rw[1][2]*cz);
        double tz = -(rw[2][0]*cx + rw[2][1]*cy + rw[2][2]*cz);
        double[] q = quaternionFromRotation(rw);
        return new CameraExtrinsics(q[0],q[1],q[2],q[3],tx,ty,tz);
    }

    private static double[][] transpose3(double[][] a) {
        double[][] t = new double[3][3];
        for(int r=0;r<3;r++) for(int c=0;c<3;c++) t[r][c]=a[c][r];
        return t;
    }

    private static double[] quaternionFromRotation(double[][] m) {
        double qw,qx,qy,qz;
        double trace=m[0][0]+m[1][1]+m[2][2];
        if(trace>0){double s=Math.sqrt(trace+1.0)*2;qw=0.25*s;qx=(m[2][1]-m[1][2])/s;qy=(m[0][2]-m[2][0])/s;qz=(m[1][0]-m[0][1])/s;}
        else if(m[0][0]>m[1][1]&&m[0][0]>m[2][2]){double s=Math.sqrt(1.0+m[0][0]-m[1][1]-m[2][2])*2;qw=(m[2][1]-m[1][2])/s;qx=0.25*s;qy=(m[0][1]+m[1][0])/s;qz=(m[0][2]+m[2][0])/s;}
        else if(m[1][1]>m[2][2]){double s=Math.sqrt(1.0+m[1][1]-m[0][0]-m[2][2])*2;qw=(m[0][2]-m[2][0])/s;qx=(m[0][1]+m[1][0])/s;qy=0.25*s;qz=(m[1][2]+m[2][1])/s;}
        else{double s=Math.sqrt(1.0+m[2][2]-m[0][0]-m[1][1])*2;qw=(m[1][0]-m[0][1])/s;qx=(m[0][2]+m[2][0])/s;qy=(m[1][2]+m[2][1])/s;qz=0.25*s;}
        double n=Math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz);if(n>0){qw/=n;qx/=n;qy/=n;qz/=n;}
        return new double[]{qw,qx,qy,qz};
    }

    private static long pack(int x,int y,int z){return ((x+VOXEL_OFFSET)&VOXEL_MASK)|(((long)(y+VOXEL_OFFSET)&VOXEL_MASK)<<VOXEL_BITS)|(((long)(z+VOXEL_OFFSET)&VOXEL_MASK)<<(2*VOXEL_BITS));}
    private static int fastFloor(float v){int i=(int)v;return v<i?i-1:i;}
    private static int clamp255(int v){return Math.max(0,Math.min(255,v));}
    private static void notifyProgress(ProgressListener l,int p,String m){if(l!=null)l.onProgress(Math.max(0,Math.min(100,p)),m);}

    private static void recreateDirectory(File dir) throws IOException {
        if (dir.exists()) deleteChildren(dir);
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("cannot create " + dir);
    }
    private static void deleteChildren(File dir) throws IOException {
        File[] children=dir.listFiles(); if(children==null)return;
        for(File child:children){if(child.isDirectory())deleteChildren(child);if(!child.delete())throw new IOException("cannot delete "+child);}
    }
    private static String readText(File file)throws IOException{try(FileInputStream in=new FileInputStream(file)){byte[]buf=new byte[(int)file.length()];int off=0,n;while(off<buf.length&&(n=in.read(buf,off,buf.length-off))>0)off+=n;return new String(buf,0,off,StandardCharsets.UTF_8);}}
    private static void writeText(File file,String text)throws IOException{try(FileOutputStream out=new FileOutputStream(file)){out.write(text.getBytes(StandardCharsets.UTF_8));}}

    private static final class CameraExtrinsics { final double qw,qx,qy,qz,tx,ty,tz; CameraExtrinsics(double a,double b,double c,double d,double e,double f,double g){qw=a;qx=b;qy=c;qz=d;tx=e;ty=f;tz=g;} }
    private static final class PointAccumulator { double x,y,z,r,g,b,weight; }
    private static final class PlyHeader {
        final long headerBytes; final int vertexCount,recordBytes,xOffset,yOffset,zOffset,redOffset,greenOffset,blueOffset,confidenceOffset;
        PlyHeader(long h,int v,int rb,int x,int y,int z,int r,int g,int b,int c){headerBytes=h;vertexCount=v;recordBytes=rb;xOffset=x;yOffset=y;zOffset=z;redOffset=r;greenOffset=g;blueOffset=b;confidenceOffset=c;}
    }
}
