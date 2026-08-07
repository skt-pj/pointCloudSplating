# pointCloudSplating

Minimal Android app based on Google's current `arcore-android-sdk` Raw Depth Java sample.

Included:
- ARCore Raw Depth (`acquireRawDepthImage16Bits`)
- Raw Depth confidence (`acquireRawDepthConfidenceImage`)
- Depth pixel to camera-local XYZ conversion
- RGB color sampling from the camera image
- Anchor-based placement of each depth frame
- Accumulation of up to 60 Raw Depth frames
- OpenGL RGB point-cloud rendering

Not included:
- plane filtering
- point clustering
- bounding boxes
- 3DGS / photogrammetry reconstruction

## Requirements

- ARCore-compatible Android device
- ARCore Raw Depth API support
- Google Play Services for AR
- Android 7.0 (API 24) or newer

## Build

The repository includes a GitHub Actions workflow that builds `app-debug.apk` and writes the latest APK to `dist/pointCloudSplating-debug.apk`.

## Source

The Raw Depth conversion and rendering are adapted from:
`google-ar/arcore-android-sdk/samples/raw_depth_java` (Apache License 2.0).

ARCore dependency: `com.google.ar:core:1.54.0`.
