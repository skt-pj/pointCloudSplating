/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from google-ar/arcore-android-sdk samples/raw_depth_java.
 */

precision mediump float;
varying vec4 v_Color;

void main() {
    vec2 centered = gl_PointCoord - vec2(0.5);
    float radius = length(centered);
    if (radius > 0.5) {
        discard;
    }

    // Preserve the sampled RGB in the center, but draw a dark ring so points remain visible when
    // they are geometrically aligned with the identical live camera pixels behind them.
    if (radius > 0.32) {
        gl_FragColor = vec4(0.02, 0.02, 0.02, 1.0);
    } else {
        gl_FragColor = vec4(min(v_Color.rgb * 1.15 + vec3(0.05), vec3(1.0)), 1.0);
    }
}
