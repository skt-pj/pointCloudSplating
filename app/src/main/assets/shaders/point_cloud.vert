uniform mat4 u_ModelViewProjection;
uniform float u_PointSize;
attribute vec4 a_Position;
varying vec4 v_Color;

vec3 polynomialColor(float x) {
    x = clamp(x * 0.9 + 0.03, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(
        dot(v4, vec4(0.55305649, 3.00913185, -5.46192616, -11.11819092))
            + dot(v2, vec2(27.81927491, -14.87899417)),
        dot(v4, vec4(0.16207513, 0.17712472, 15.24091500, -36.50657960))
            + dot(v2, vec2(25.95549545, -5.02738237)),
        dot(v4, vec4(-0.05195877, 5.18000081, -30.94853351, 81.96403246))
            + dot(v2, vec2(-86.53476570, 30.23299484))
    );
}

void main() {
    float normalizedHeight = clamp((a_Position.y + 2.0) / 4.0, 0.0, 1.0);
    v_Color = vec4(polynomialColor(normalizedHeight), 1.0);
    gl_Position = u_ModelViewProjection * vec4(a_Position.xyz, 1.0);
    gl_PointSize = u_PointSize;
}
