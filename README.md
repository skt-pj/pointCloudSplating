# pointCloudSplating

Minimal Android app that extracts the Raw Depth point-cloud acquisition path from Google's `codelab-raw-depth-api` sample.

Included only:
- ARCore camera preview
- Raw Depth (`acquireRawDepthImage16Bits`)
- Raw Depth confidence (`acquireRawDepthConfidenceImage`)
- Depth pixel to world-space XYZ conversion
- OpenGL point-cloud rendering

Not included:
- plane filtering
- point clustering
- bounding boxes
- 3DGS / photogrammetry reconstruction

## Requirements

- ARCore-compatible Android device
- ARCore Depth API support
- Google Play Services for AR
- Android 7.0 (API 24) or newer

## Build

The repository includes a GitHub Actions workflow that builds `app-debug.apk` and writes the latest APK to `dist/pointCloudSplating-debug.apk`.

## Source

The point-cloud conversion and rendering are derived from:
`google-ar/codelab-raw-depth-api` (Apache License 2.0).

ARCore dependency: `com.google.ar:core:1.54.0`.
