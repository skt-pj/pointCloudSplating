#extension GL_OES_EGL_image_external : require
precision mediump float;
uniform samplerExternalOES sTexture;
varying vec2 v_TexCoord;

void main() {
    gl_FragColor = texture2D(sTexture, v_TexCoord);
}
