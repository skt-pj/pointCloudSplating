package com.sktpj.pointcloudsplatting;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Finalizes RGB/Pose observations and optional Raw Depth priors. */
public final class DatasetFinalizer {
    private DatasetFinalizer() {}

    public static final class Result {
        public final boolean success;
        public final File directory;
        public final int frameCount;
        public final String message;
        private Result(boolean success,File directory,int frameCount,String message){this.success=success;this.directory=directory;this.frameCount=frameCount;this.message=message;}
        static Result ok(File d,int n){return new Result(true,d,n,"saved "+n+" keyframes");}
        static Result fail(File d,String m){return new Result(false,d,0,m);}
    }

    public static Result finalizeDataset(File workingDirectory) {
        if (workingDirectory == null || !workingDirectory.isDirectory()) return Result.fail(workingDirectory,"capture directory unavailable");
        try {
            List<JSONObject> sourceFrames=readFrameMetadata(workingDirectory);
            if(sourceFrames.isEmpty()) return Result.fail(workingDirectory,"no captured keyframes");
            Camera2Calibration calibration=readCamera2Calibration(workingDirectory);

            JSONObject transforms=new JSONObject();
            transforms.put("camera_model","OPENCV");
            transforms.put("k1",0.0);transforms.put("k2",0.0);transforms.put("p1",0.0);transforms.put("p2",0.0);
            transforms.put("coordinate_system","ARCore root-anchor local; OpenGL camera convention (+X right,+Y up,-Z forward)");
            transforms.put("source","pointCloudSplating ARCore SharedCamera continuous CPU frames");
            transforms.put("intrinsics_policy","Continuous ARCore CPU frames use the exact ARCore image intrinsics for that stream; legacy Camera2 stills retain Camera2 calibration mapping when present");

            JSONArray frames=new JSONArray();int depthFrames=0;int camera2CalibratedFrames=0;
            for(JSONObject source:sourceFrames){JSONObject frame=toNerfstudioFrame(source,calibration);frames.put(frame);if(frame.has("depth_point_cloud_path"))depthFrames++;if("camera2_lens_intrinsic_calibration".equals(frame.optString("intrinsics_source","")))camera2CalibratedFrames++;}
            transforms.put("frames",frames);transforms.put("rgb_frame_count",sourceFrames.size());transforms.put("raw_depth_prior_frame_count",depthFrames);transforms.put("camera2_calibrated_rgb_frame_count",camera2CalibratedFrames);
            writeJson(new File(workingDirectory,"transforms.json"),transforms);

            JSONObject manifest=new JSONObject();manifest.put("format_version",4);manifest.put("state","saved");manifest.put("frame_count",sourceFrames.size());manifest.put("rgb_frame_count",sourceFrames.size());manifest.put("raw_depth_prior_frame_count",depthFrames);manifest.put("camera2_calibrated_rgb_frame_count",camera2CalibratedFrames);manifest.put("transforms","transforms.json");manifest.put("image_pattern","frame_*.jpg");manifest.put("per_frame_metadata_pattern","frame_*.json");manifest.put("raw_depth_point_cloud_pattern","frame_*.ply (optional per RGB frame)");manifest.put("capture_strategy","continuous ARCore CPU-frame sampling; automatically keep the sharpest frame from each moving viewpoint window");manifest.put("observation_priority","Automatically selected RGB + synchronized camera pose are primary 3DGS observations; Raw Depth PLY is an optional geometry prior and does not gate a valid RGB frame.");manifest.put("intrinsics_priority","Exact ARCore image intrinsics for continuous CPU frames; Camera2 physical-camera calibration only for legacy Camera2 still captures.");
            writeJson(new File(workingDirectory,"dataset_manifest.json"),manifest);
            try(FileOutputStream out=new FileOutputStream(new File(workingDirectory,".saved"))){out.write("saved\n".getBytes(StandardCharsets.UTF_8));}
            DiagnosticLog.i("DatasetFinalizer","Finalized continuous RGB="+sourceFrames.size()+" depthPrior="+depthFrames+" legacyCamera2Intrinsics="+camera2CalibratedFrames);
            File finalDirectory=renameAsSavedDataset(workingDirectory);return Result.ok(finalDirectory,sourceFrames.size());
        } catch(IOException|JSONException|RuntimeException e){DiagnosticLog.e("DatasetFinalizer","Failed to finalize dataset",e);return Result.fail(workingDirectory,"finalize failed: "+e.getClass().getSimpleName());}
    }

    private static List<JSONObject> readFrameMetadata(File directory)throws IOException,JSONException{
        File[]files=directory.listFiles((dir,name)->name.startsWith("frame_")&&name.endsWith(".json"));List<JSONObject>frames=new ArrayList<>();if(files==null)return frames;for(File file:files)frames.add(new JSONObject(readText(file)));frames.sort(Comparator.comparingInt(o->o.optInt("capture_index",Integer.MAX_VALUE)));return frames;
    }

    private static Camera2Calibration readCamera2Calibration(File directory){
        File file=new File(directory,"session_camera.json");if(!file.isFile())return null;
        try{
            JSONObject json=new JSONObject(readText(file));JSONArray intr=json.optJSONArray("lens_intrinsic_calibration");JSONObject active=json.optJSONObject("active_array");
            if(intr==null||intr.length()<5||active==null)return null;
            double fx=intr.getDouble(0),fy=intr.getDouble(1),cx=intr.getDouble(2),cy=intr.getDouble(3),skew=intr.getDouble(4);
            int left=active.getInt("left"),top=active.getInt("top"),right=active.getInt("right"),bottom=active.getInt("bottom");
            if(!finitePositive(fx)||!finitePositive(fy)||right<=left||bottom<=top)return null;
            return new Camera2Calibration(fx,fy,cx,cy,skew,left,top,right,bottom);
        }catch(Exception e){DiagnosticLog.w("DatasetFinalizer","Camera2 calibration unavailable: "+e.getMessage());return null;}
    }

    private static JSONObject toNerfstudioFrame(JSONObject source,Camera2Calibration calibration)throws JSONException{
        int jpegWidth=source.getInt("jpeg_width"),jpegHeight=source.getInt("jpeg_height");
        double fx,fy,cx,cy,skew=0.0;String intrinsicsSource;
        Camera2MappedIntrinsics mapped=mapCamera2ToJpeg(source,calibration,jpegWidth,jpegHeight);
        if(mapped!=null){fx=mapped.fx;fy=mapped.fy;cx=mapped.cx;cy=mapped.cy;skew=mapped.skew;intrinsicsSource="camera2_lens_intrinsic_calibration";}
        else{
            JSONObject intrinsics=source.getJSONObject("arcore_image_intrinsics");JSONArray focal=intrinsics.getJSONArray("focal_length_px");JSONArray principal=intrinsics.getJSONArray("principal_point_px");JSONArray dimensions=intrinsics.getJSONArray("image_dimensions");double sourceWidth=dimensions.getDouble(0),sourceHeight=dimensions.getDouble(1);double scaleX=jpegWidth/sourceWidth,scaleY=jpegHeight/sourceHeight;fx=focal.getDouble(0)*scaleX;fy=focal.getDouble(1)*scaleY;cx=principal.getDouble(0)*scaleX;cy=principal.getDouble(1)*scaleY;intrinsicsSource="arcore_image_intrinsics";
        }

        JSONObject out=new JSONObject();out.put("file_path",source.getString("image"));
        if(!source.isNull("point_cloud")){String pointCloud=source.optString("point_cloud","");if(!pointCloud.isEmpty()&&!"null".equals(pointCloud))out.put("depth_point_cloud_path",pointCloud);}
        out.put("w",jpegWidth);out.put("h",jpegHeight);out.put("fl_x",fx);out.put("fl_y",fy);out.put("cx",cx);out.put("cy",cy);out.put("camera2_skew",skew);out.put("intrinsics_source",intrinsicsSource);out.put("capture_index",source.getInt("capture_index"));out.put("timestamp_ns",source.getLong("jpeg_sensor_timestamp_ns"));out.put("has_raw_depth_prior",source.optBoolean("has_raw_depth_prior",false));out.put("transform_matrix",columnMajorToRows(source.getJSONArray("world_from_camera_column_major")));return out;
    }

    private static Camera2MappedIntrinsics mapCamera2ToJpeg(JSONObject source,Camera2Calibration c,int w,int h){
        if(c==null)return null;
        try{
            // Continuous ARCore CPU images are already paired with exact ARCore image intrinsics.
            // Never reinterpret those pixels through the Camera2 active array/crop model.
            JSONObject capture=source.optJSONObject("camera2_capture");
            if(capture==null)return null;
            int left=c.activeLeft,top=c.activeTop,right=c.activeRight,bottom=c.activeBottom;
            JSONObject crop=capture.optJSONObject("crop_region");
            if(crop!=null){left=crop.getInt("left");top=crop.getInt("top");right=crop.getInt("right");bottom=crop.getInt("bottom");}
            double cropW=right-left,cropH=bottom-top;if(cropW<=0||cropH<=0)return null;
            double sx=w/cropW,sy=h/cropH;double fx=c.fx*sx,fy=c.fy*sy,cx=(c.cx-left)*sx,cy=(c.cy-top)*sy,skew=c.skew*sx;
            if(!finitePositive(fx)||!finitePositive(fy)||!Double.isFinite(cx)||!Double.isFinite(cy))return null;
            if(cx<-w*.25||cx>w*1.25||cy<-h*.25||cy>h*1.25)return null;
            return new Camera2MappedIntrinsics(fx,fy,cx,cy,skew);
        }catch(Exception e){return null;}
    }

    private static boolean finitePositive(double v){return Double.isFinite(v)&&v>0.0;}

    private static JSONArray columnMajorToRows(JSONArray columnMajor)throws JSONException{if(columnMajor.length()!=16)throw new JSONException("Expected 16-value camera transform");JSONArray rows=new JSONArray();for(int row=0;row<4;row++){JSONArray values=new JSONArray();for(int col=0;col<4;col++)values.put(columnMajor.getDouble(col*4+row));rows.put(values);}return rows;}
    private static File renameAsSavedDataset(File workingDirectory){String name=workingDirectory.getName();if(!name.startsWith("capture_tmp_"))return workingDirectory;File parent=workingDirectory.getParentFile();if(parent==null)return workingDirectory;String suffix=name.substring("capture_tmp_".length());File target=new File(parent,"dataset_"+suffix);if(target.exists())target=new File(parent,"dataset_"+suffix+"_saved");return workingDirectory.renameTo(target)?target:workingDirectory;}
    private static String readText(File file)throws IOException{StringBuilder out=new StringBuilder();try(BufferedReader reader=new BufferedReader(new InputStreamReader(new FileInputStream(file),StandardCharsets.UTF_8))){String line;while((line=reader.readLine())!=null)out.append(line).append('\n');}return out.toString();}
    private static void writeJson(File file,JSONObject json)throws IOException,JSONException{try(FileOutputStream out=new FileOutputStream(file)){out.write(json.toString(2).getBytes(StandardCharsets.UTF_8));}}

    private static final class Camera2Calibration{final double fx,fy,cx,cy,skew;final int activeLeft,activeTop,activeRight,activeBottom;Camera2Calibration(double fx,double fy,double cx,double cy,double skew,int l,int t,int r,int b){this.fx=fx;this.fy=fy;this.cx=cx;this.cy=cy;this.skew=skew;activeLeft=l;activeTop=t;activeRight=r;activeBottom=b;}}
    private static final class Camera2MappedIntrinsics{final double fx,fy,cx,cy,skew;Camera2MappedIntrinsics(double fx,double fy,double cx,double cy,double skew){this.fx=fx;this.fy=fy;this.cx=cx;this.cy=cy;this.skew=skew;}}
}
