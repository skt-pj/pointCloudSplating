# pointCloudSplating

Android Raw Depth point-cloud scanner based on Google's current `arcore-android-sdk` Raw Depth Java sample.

Current app version: `0.2.0` (`versionCode 2`).

Included:
- ARCore Raw Depth (`acquireRawDepthImage16Bits`)
- Raw Depth confidence (`acquireRawDepthConfidenceImage`)
- Depth pixel to camera-local XYZ conversion
- RGB color sampling from the camera image
- Anchor-based placement of each depth frame
- Accumulation of up to 60 Raw Depth frames
- OpenGL RGB point-cloud rendering
- Periodic photogrammetry texture-frame capture
- Per-photo ARCore camera pose, intrinsics and Camera2 metadata JSON
- Sharp-frame selection using exposure time, AF/lens state and device motion
- Pixel 10a capture profile

## Pixel 10a camera policy

The app selects the largest ARCore CPU image stream available from 30 fps camera configs, enables `Config.FocusMode.AUTO`, and explicitly keeps ARCore EIS off so saved image geometry remains suitable for photogrammetry.

The Pixel 10a texture capture gate currently prefers:
- exposure <= 10 ms (~1/100 s)
- linear phone speed <= 0.08 m/s
- angular phone speed <= 6 degrees/s
- AF inactive/fixed-focus, passive-focused, or focused-locked state
- stationary lens state
- at least ~3.5 cm translation or 3 degrees rotation between normal captures

Texture images are JPEG quality 95 and are stored under the app-specific Pictures directory in a `photogrammetry_YYYYMMDD_HHMMSS` session folder. Each JPEG has a matching JSON file containing camera-to-world/world-to-camera transforms, intrinsics, exposure, ISO, aperture, focal length, focus state/range, OIS/EIS metadata, rolling-shutter skew and timestamps.

The app currently uses the regular ARCore camera session rather than `SharedCamera`. This keeps Raw Depth and pose synchronization simple and reliable. Full-resolution Pixel Camera2 still capture can be added later with ARCore `SharedCamera` if the ARCore CPU stream resolution is not sufficient.

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

## Versioning and APK updates

The app keeps the same `applicationId` and uses an incrementing `versionCode`. GitHub Actions restores a stable development signing key before building, so APKs from version `0.2.0` onward can update each other in place.

The older `0.1.0` APK was built with an ephemeral GitHub Actions debug key, so it must be uninstalled once before installing `0.2.0`. After that, future APKs using the stable development key can be installed as updates.

GitHub Actions publishes both:
- `dist/pointCloudSplating-debug.apk`
- `dist/pointCloudSplating-v<versionName>-debug.apk`

## Source

The Raw Depth conversion and rendering are adapted from:
`google-ar/arcore-android-sdk/samples/raw_depth_java` (Apache License 2.0).

ARCore dependency: `com.google.ar:core:1.54.0`.
