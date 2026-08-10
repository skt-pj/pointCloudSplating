package com.sktpj.pointcloudsplatting;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Environment;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/** Thumbnail library for saved reconstruction datasets and their real training state. */
public final class LibraryActivity extends Activity {
    private static final String TAG = "PointCloudLibrary";
    private static final int REQUEST_SAVE_LOG = 1201;
    private static final int MENU_SAVE_LOG = 1;
    private static final int MENU_CHANGE_LOG_DESTINATION = 2;
    private static final int CARD_THUMBNAIL_HEIGHT_DP = 150;
    private static final String FINAL_SPLAT = "splat.ply";
    private static final String PREVIEW_SPLAT = "preview_splat.ply";
    private static final String DEPTH_PRIOR = "depth_prior.ply";
    private static final int[] TRAINING_PRESETS = {300, 1_000, 3_000, 10_000};

    private GridLayout grid;
    private TextView emptyView;
    private volatile boolean generationInProgress;

    @Override protected void onCreate(Bundle savedInstanceState){super.onCreate(savedInstanceState);setContentView(buildContentView());}
    @Override protected void onResume(){super.onResume();reloadLibrary();DriveDiagnosticLogStore.requestOverwrite();}

    private View buildContentView(){
        LinearLayout root=new LinearLayout(this);root.setOrientation(LinearLayout.VERTICAL);root.setBackgroundColor(0xFF101010);
        LinearLayout header=new LinearLayout(this);header.setOrientation(LinearLayout.HORIZONTAL);header.setGravity(Gravity.CENTER_VERTICAL);header.setPadding(dp(12),dp(8),dp(12),dp(8));header.setBackgroundColor(0xFF202020);
        Button back=new Button(this);back.setText("戻る");back.setAllCaps(false);back.setContentDescription("撮影画面に戻る");back.setMinHeight(dp(48));back.setOnClickListener(v->finish());header.addView(back,new LinearLayout.LayoutParams(dp(84),dp(52)));
        TextView title=new TextView(this);title.setText("ライブラリ");title.setTextColor(0xFFFFFFFF);title.setTextSize(20f);title.setGravity(Gravity.CENTER);header.addView(title,new LinearLayout.LayoutParams(0,dp(52),1f));
        Button reload=new Button(this);reload.setText("更新");reload.setAllCaps(false);reload.setContentDescription("ライブラリの一覧を更新する");reload.setMinHeight(dp(48));reload.setOnClickListener(v->reloadLibrary());header.addView(reload,new LinearLayout.LayoutParams(dp(84),dp(52)));
        Button menu=new Button(this);menu.setText("⋮");menu.setTextSize(24f);menu.setAllCaps(false);menu.setContentDescription("ライブラリメニュー");menu.setMinWidth(0);menu.setMinHeight(dp(48));menu.setPadding(0,0,0,0);menu.setOnClickListener(v->showLibraryMenu(menu));header.addView(menu,new LinearLayout.LayoutParams(dp(52),dp(52)));root.addView(header);
        FrameLayout content=new FrameLayout(this);root.addView(content,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,0,1f));
        ScrollView scroll=new ScrollView(this);grid=new GridLayout(this);grid.setColumnCount(2);grid.setUseDefaultMargins(false);grid.setPadding(dp(6),dp(6),dp(6),dp(24));scroll.addView(grid,new ScrollView.LayoutParams(ScrollView.LayoutParams.MATCH_PARENT,ScrollView.LayoutParams.WRAP_CONTENT));content.addView(scroll,new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,FrameLayout.LayoutParams.MATCH_PARENT));
        emptyView=new TextView(this);emptyView.setText("ライブラリにはまだ撮影がありません\n\n撮影画面で対象の周りを撮影し、\n「撮影を保存」を押してください。");emptyView.setTextColor(0xFFCCCCCC);emptyView.setTextSize(17f);emptyView.setGravity(Gravity.CENTER);emptyView.setPadding(dp(24),dp(24),dp(24),dp(24));content.addView(emptyView,new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,FrameLayout.LayoutParams.MATCH_PARENT));
        root.setOnApplyWindowInsetsListener((view,insets)->{int top=insets.getSystemWindowInsetTop(),bottom=insets.getSystemWindowInsetBottom(),left=insets.getSystemWindowInsetLeft(),right=insets.getSystemWindowInsetRight();if(android.os.Build.VERSION.SDK_INT>=28&&insets.getDisplayCutout()!=null){android.view.DisplayCutout c=insets.getDisplayCutout();top=Math.max(top,c.getSafeInsetTop());bottom=Math.max(bottom,c.getSafeInsetBottom());left=Math.max(left,c.getSafeInsetLeft());right=Math.max(right,c.getSafeInsetRight());}root.setPadding(left,top+dp(12),right,bottom+dp(8));return insets;});root.post(root::requestApplyInsets);return root;
    }

    private void showLibraryMenu(View anchor){
        PopupMenu popup=new PopupMenu(this,anchor);
        boolean configured=DriveDiagnosticLogStore.hasDestination(this);
        popup.getMenu().add(0,MENU_SAVE_LOG,0,configured?"Driveログを更新":"Driveログ保存先を設定");
        if(configured)popup.getMenu().add(0,MENU_CHANGE_LOG_DESTINATION,1,"Driveログ保存先を変更");
        popup.setOnMenuItemClickListener(item->{
            if(item.getItemId()==MENU_SAVE_LOG){saveDiagnosticLog();return true;}
            if(item.getItemId()==MENU_CHANGE_LOG_DESTINATION){chooseDriveDiagnosticDestination();return true;}
            return false;
        });
        popup.show();
    }

    private void saveDiagnosticLog(){
        if(!DriveDiagnosticLogStore.hasDestination(this)){chooseDriveDiagnosticDestination();return;}
        Toast.makeText(this,"Driveのログを更新しています…",Toast.LENGTH_SHORT).show();
        DriveDiagnosticLogStore.overwriteNow((success,message)->runOnUiThread(()->
                Toast.makeText(this,message,success?Toast.LENGTH_SHORT:Toast.LENGTH_LONG).show()));
    }

    private void chooseDriveDiagnosticDestination(){
        Toast.makeText(this,"Google Driveで保存先を選択してください",Toast.LENGTH_SHORT).show();
        startActivityForResult(DriveDiagnosticLogStore.createDestinationIntent(),REQUEST_SAVE_LOG);
    }

    @Override protected void onActivityResult(int requestCode,int resultCode,Intent data){
        super.onActivityResult(requestCode,resultCode,data);
        if(requestCode!=REQUEST_SAVE_LOG)return;
        if(resultCode!=RESULT_OK||data==null||data.getData()==null)return;
        if(!DriveDiagnosticLogStore.registerDestination(this,data)){
            Toast.makeText(this,"Driveの保存先を設定できませんでした",Toast.LENGTH_LONG).show();
            return;
        }
        Toast.makeText(this,"Driveログを設定しました。以後は同じファイルを上書きします",Toast.LENGTH_SHORT).show();
        DriveDiagnosticLogStore.overwriteNow((success,message)->runOnUiThread(()->
                Toast.makeText(this,message,success?Toast.LENGTH_SHORT:Toast.LENGTH_LONG).show()));
    }

    private void reloadLibrary(){File pictures=getExternalFilesDir(Environment.DIRECTORY_PICTURES);List<File>datasets=findSavedDatasets(pictures);grid.removeAllViews();emptyView.setVisibility(datasets.isEmpty()?View.VISIBLE:View.GONE);for(File dataset:datasets){migrateLegacyArtifacts(dataset);grid.addView(createDatasetCard(dataset));}}

    private View createDatasetCard(File dataset){
        LinearLayout card=new LinearLayout(this);card.setOrientation(LinearLayout.VERTICAL);card.setPadding(dp(8),dp(8),dp(8),dp(10));card.setMinimumHeight(dp(220));GradientDrawable bg=new GradientDrawable();bg.setColor(0xFF282828);bg.setCornerRadius(dp(12));card.setBackground(bg);
        GridLayout.LayoutParams params=new GridLayout.LayoutParams();params.width=0;params.height=GridLayout.LayoutParams.WRAP_CONTENT;params.columnSpec=GridLayout.spec(GridLayout.UNDEFINED,1f);params.setMargins(dp(5),dp(5),dp(5),dp(5));card.setLayoutParams(params);
        ImageView thumbnail=new ImageView(this);thumbnail.setScaleType(ImageView.ScaleType.CENTER_CROP);thumbnail.setBackgroundColor(0xFF151515);thumbnail.setContentDescription("ライブラリの撮影写真");Bitmap bitmap=decodeThumbnail(findFirstJpeg(dataset),480,360);if(bitmap!=null)thumbnail.setImageBitmap(bitmap);card.addView(thumbnail,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,dp(CARD_THUMBNAIL_HEIGHT_DP)));
        TextView name=new TextView(this);name.setText(formatDatasetName(dataset.getName()));name.setTextColor(0xFFFFFFFF);name.setTextSize(15f);name.setMaxLines(1);name.setPadding(dp(4),dp(8),dp(4),0);card.addView(name);
        TextView status=new TextView(this);status.setText(buildDatasetStatus(dataset));status.setTextColor(0xFFDDDDDD);status.setTextSize(14f);status.setPadding(dp(4),dp(4),dp(4),dp(4));card.addView(status);
        if(isPhotometricComplete(dataset)&&ModelProcessingCoordinator.isPhase3ProcessingEnabled()){
            Button more=new Button(this);more.setText("追加学習");more.setAllCaps(false);more.setMinHeight(dp(44));more.setContentDescription("完成した3DGSモデルを追加学習する");more.setOnClickListener(v->showTrainingOptions(dataset,status,true));card.addView(more,new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT,dp(48)));
        }
        card.setContentDescription(formatDatasetName(dataset.getName())+"。"+buildDatasetStatus(dataset).replace('\n',' ')+"。タップして3Dモデルを開く、または作成する。");card.setClickable(true);card.setFocusable(true);card.setOnClickListener(v->openOrTrain(dataset,status));return card;
    }

    private void openOrTrain(File dataset,TextView status){migrateLegacyArtifacts(dataset);if(isPhotometricComplete(dataset)){openViewer(dataset);return;}if(!ModelProcessingCoordinator.isPhase3ProcessingEnabled()){Toast.makeText(this,"撮影データは保存済みです。3Dモデル作成機能は現在準備中です。",Toast.LENGTH_LONG).show();return;}showTrainingOptions(dataset,status,false);}

    private void showTrainingOptions(File dataset,TextView status,boolean continuation){
        if(generationInProgress){Toast.makeText(this,"3Dモデルを学習しています。このままお待ちください。",Toast.LENGTH_SHORT).show();return;}
        if(continuation&&!GaussianSplatJob.canContinueTraining(dataset)){
            new AlertDialog.Builder(this).setTitle("このモデルは追加学習できません")
                    .setMessage("このモデルは旧バージョンで作成され、Gaussian本体以外のoptimizer状態が保存されていません。\n\nPLYだけを読み直す方法は同じ学習の続きにはならないため、自動では行いません。新版で作成したモデルからは何度でも追加学習できます。")
                    .setPositiveButton("OK",null).show();return;
        }
        String[] labels={"300 step","1,000 step","3,000 step","10,000 step","任意の回数…"};
        String title=continuation?"追加する学習回数":"最初の学習回数";
        new AlertDialog.Builder(this).setTitle(title).setItems(labels,(dialog,which)->{
            if(which<TRAINING_PRESETS.length)startTraining(dataset,status,continuation,TRAINING_PRESETS[which]);else showCustomTrainingInput(dataset,status,continuation);
        }).setNegativeButton("キャンセル",null).show();
    }

    private void showCustomTrainingInput(File dataset,TextView status,boolean continuation){
        EditText input=new EditText(this);input.setInputType(InputType.TYPE_CLASS_NUMBER);input.setHint("例: 5000");input.setSingleLine(true);input.setPadding(dp(20),dp(10),dp(20),dp(10));
        FrameLayout container=new FrameLayout(this);container.setPadding(dp(20),0,dp(20),0);container.addView(input,new FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT,FrameLayout.LayoutParams.WRAP_CONTENT));
        AlertDialog dialog=new AlertDialog.Builder(this).setTitle(continuation?"追加学習回数":"学習回数").setView(container).setPositiveButton("開始",null).setNegativeButton("キャンセル",null).create();
        dialog.setOnShowListener(ignored->dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v->{String value=input.getText().toString().trim();try{long parsed=Long.parseLong(value);if(parsed<1||parsed>Integer.MAX_VALUE)throw new NumberFormatException();dialog.dismiss();startTraining(dataset,status,continuation,(int)parsed);}catch(NumberFormatException e){input.setError("1〜"+Integer.MAX_VALUE+" の整数を入力してください");}}));dialog.show();
    }

    private void startTraining(File dataset,TextView status,boolean continuation,int steps){
        if(generationInProgress)return;generationInProgress=true;status.setText((continuation?"3Dモデルを追加学習中":"3Dモデルを学習中")+"\n準備しています…");
        String threadName=continuation?"LibraryContinue3DGS":"LibraryTrainFull3DGS";
        new Thread(()->{
            GaussianSplatJob.Result result=continuation
                    ?GaussianSplatJob.continueTraining(dataset,steps,(percent,message)->runOnUiThread(()->status.setText("追加学習中 "+percent+"%\n"+message)))
                    :GaussianSplatJob.prepare(this,dataset,steps,(percent,message)->runOnUiThread(()->status.setText("3Dモデルを学習中 "+percent+"%\n"+message)));
            runOnUiThread(()->{generationInProgress=false;if(result.success){reloadLibrary();openViewer(dataset);}else{status.setText(buildDatasetStatus(dataset));Toast.makeText(this,result.message,Toast.LENGTH_LONG).show();}});
        },threadName).start();
    }

    private void openViewer(File dataset){if(!isPhotometricComplete(dataset)){Toast.makeText(this,"3DGS学習が完了したモデルはまだありません。",Toast.LENGTH_LONG).show();return;}Intent intent=new Intent(this,GaussianViewerActivity.class);intent.putExtra(GaussianViewerActivity.EXTRA_DATASET_PATH,dataset.getAbsolutePath());startActivity(intent);}

    private String buildDatasetStatus(File dataset){int frames=readFrameCount(dataset);String photos=frames+"枚の写真";if(isPhotometricComplete(dataset)){JSONObject result=readResult(dataset);int steps=result==null?0:result.optInt("training_steps",0);String trained=steps>0?"3Dモデル完成・"+steps+" step":"3Dモデル完成";return photos+"\n"+trained+(GaussianSplatJob.canContinueTraining(dataset)?"・追加学習可":"");}if(!ModelProcessingCoordinator.isPhase3ProcessingEnabled())return photos+"\n撮影データ保存済み";if(isHqPreview(dataset))return photos+"\n旧プレビューあり・タップして学習";if(new File(dataset,DEPTH_PRIOR).isFile())return photos+"\nタップして3Dモデルを作成";return photos+"\nタップして3Dモデルを作成";}

    private static boolean isPhotometricComplete(File dataset){File splat=new File(dataset,FINAL_SPLAT);if(!splat.isFile())return false;JSONObject result=readResult(dataset);return result!=null&&result.optBoolean("photometric_optimization",false)&&result.optBoolean("rasterized_image_loss",false)&&result.optBoolean("l1_ssim_backward",false)&&result.optBoolean("density_control",false)&&result.optBoolean("final_3dgs",false)&&"COMPLETE".equals(result.optString("status",""));}
    private static boolean isHqPreview(File dataset){File preview=new File(dataset,PREVIEW_SPLAT);if(!preview.isFile())return false;JSONObject result=readResult(dataset);return result!=null&&"HQ_RGB_REFINED".equals(result.optString("status",""))&&result.optBoolean("appearance_refinement",false)&&!result.optBoolean("final_3dgs",false);}

    private static void migrateLegacyArtifacts(File dataset){
        File finalSplat=new File(dataset,FINAL_SPLAT);if(!finalSplat.isFile())return;JSONObject result=readResult(dataset);if(result==null)return;
        try{
            if("HQ_RGB_REFINED".equals(result.optString("status",""))&&!result.optBoolean("final_3dgs",false)){
                File preview=new File(dataset,PREVIEW_SPLAT);if(!preview.isFile()&&finalSplat.renameTo(preview)){result.put("output",PREVIEW_SPLAT);result.put("preview_output",PREVIEW_SPLAT);result.put("appearance_refinement",true);result.put("photometric_optimization",false);result.put("final_3dgs",false);result.put("note","Migrated pre-v0.7 appearance-refined preview. Not final 3DGS.");writeText(new File(dataset,"3dgs_result.json"),result.toString(2));}return;
            }
            if(!result.optBoolean("photometric_optimization",false)){
                File prior=new File(dataset,DEPTH_PRIOR);if(!prior.isFile()&&finalSplat.renameTo(prior)){result.put("status","DEPTH_PRIOR_READY");result.put("output",DEPTH_PRIOR);result.put("appearance_refinement",false);result.put("photometric_optimization",false);result.put("final_3dgs",false);result.put("note","Migrated legacy depth-only Gaussian output; not a completed 3DGS model.");writeText(new File(dataset,"3dgs_result.json"),result.toString(2));}
            }
        }catch(Exception ignored){}
    }

    private static List<File> findSavedDatasets(File pictures){List<File>out=new ArrayList<>();if(pictures==null||!pictures.isDirectory())return out;File[]dirs=pictures.listFiles(File::isDirectory);if(dirs==null)return out;for(File dir:dirs)if(new File(dir,".saved").isFile()||(dir.getName().startsWith("dataset_")&&new File(dir,"transforms.json").isFile()))out.add(dir);out.sort(Comparator.comparingLong(File::lastModified).reversed());return out;}
    private static File findFirstJpeg(File dataset){File[]images=dataset.listFiles((dir,name)->name.startsWith("frame_")&&name.toLowerCase(Locale.US).endsWith(".jpg"));if(images==null||images.length==0)return null;Arrays.sort(images,Comparator.comparing(File::getName));return images[0];}
    private static Bitmap decodeThumbnail(File file,int targetWidth,int targetHeight){if(file==null||!file.isFile())return null;BitmapFactory.Options bounds=new BitmapFactory.Options();bounds.inJustDecodeBounds=true;BitmapFactory.decodeFile(file.getAbsolutePath(),bounds);if(bounds.outWidth<=0||bounds.outHeight<=0)return null;int sample=1;while(bounds.outWidth/(sample*2)>=targetWidth&&bounds.outHeight/(sample*2)>=targetHeight)sample*=2;BitmapFactory.Options options=new BitmapFactory.Options();options.inSampleSize=Math.max(1,sample);options.inPreferredConfig=Bitmap.Config.RGB_565;return BitmapFactory.decodeFile(file.getAbsolutePath(),options);}
    private static int readFrameCount(File dataset){File manifest=new File(dataset,"dataset_manifest.json");try{if(manifest.isFile()){JSONObject json=new JSONObject(readText(manifest));return json.optInt("frame_count",countJpegs(dataset));}}catch(Exception ignored){}return countJpegs(dataset);}
    private static JSONObject readResult(File dataset){File result=new File(dataset,"3dgs_result.json");try{if(result.isFile())return new JSONObject(readText(result));}catch(Exception ignored){}return null;}
    private static int countJpegs(File dataset){File[]files=dataset.listFiles((dir,name)->name.startsWith("frame_")&&name.toLowerCase(Locale.US).endsWith(".jpg"));return files==null?0:files.length;}
    private static String readText(File file)throws Exception{StringBuilder out=new StringBuilder();try(BufferedReader reader=new BufferedReader(new InputStreamReader(new FileInputStream(file),StandardCharsets.UTF_8))){String line;while((line=reader.readLine())!=null)out.append(line).append('\n');}return out.toString();}
    private static void writeText(File file,String text)throws Exception{try(FileOutputStream out=new FileOutputStream(file)){out.write(text.getBytes(StandardCharsets.UTF_8));}}
    private static String formatDatasetName(String name){String value=name;if(value.startsWith("dataset_"))value=value.substring("dataset_".length());try{Date date=new SimpleDateFormat("yyyyMMdd_HHmmss",Locale.US).parse(value);if(date!=null)return new SimpleDateFormat("yyyy/MM/dd HH:mm:ss",Locale.getDefault()).format(date);}catch(ParseException ignored){}return name;}
    private int dp(int value){return Math.round(value*getResources().getDisplayMetrics().density);}
}
