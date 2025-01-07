// Adapted from https://invent.kde.org/-/snippets/1736

#version 120


// Zoom factor
uniform float zoom;

uniform sampler2D curr;
uniform sampler2D next;

uniform vec2 size;

float lanczos2_window_sinc = 0.4;
float lanczos2_sinc = 1.0;
float lanczos2_ar_strength = 0.65;
float lanczos2_res = 0.98;


//#define wa 0.4
//#define wb 0.8

//#define wa (lanczos2_window_sinc * pi)
//#define wb (lanczos2_sinc * pi)

#define wa (0.8)
#define wb (halfpi)

const float halfpi = 1.5707963267948966192313216916398;
const float pi = 3.1415926535897932384626433832795;
const vec3 dtt = vec3(65536.0, 255.0, 1.0);

vec2 zoomed(vec2 coord)
{
    return (coord / zoom) + 0.5 - (0.5 / zoom);
}

vec4 reduce4(vec3 A, vec3 B, vec3 C, vec3 D)
{
    return dtt * mat4x3(A, B, C, D);
}

// Calculates the distance between two points
float d(vec2 pt1, vec2 pt2)
{
    vec2 v = pt2 - pt1;
    return sqrt(dot(v, v));
}

vec3 min4(vec3 a, vec3 b, vec3 c, vec3 d)
{
    return min(a, min(b, min(c, d)));
}

vec3 max4(vec3 a, vec3 b, vec3 c, vec3 d)
{
    return max(a, max(b, max(c, d)));
}

vec4 resampler(vec4 x)
{
    return (x == vec4(0.0)) ?  vec4(wa * wb) : sin(x * wa) * sin(x * wb) / (x * x);
}


vec4 gaussianAt(sampler2D sampler, vec2 coord) {
    // Gaussian kernel 5x5
    float kernel[25];
    kernel[0] = 1.0; kernel[1] = 4.0; kernel[2] = 7.0; kernel[3] = 4.0; kernel[4] = 1.0;
    kernel[5] = 4.0; kernel[6] = 16.0; kernel[7] = 26.0; kernel[8] = 16.0; kernel[9] = 4.0;
    kernel[10] = 7.0; kernel[11] = 26.0; kernel[12] = 41.0; kernel[13] = 26.0; kernel[14] = 7.0;
    kernel[15] = 4.0; kernel[16] = 16.0; kernel[17] = 26.0; kernel[18] = 16.0; kernel[19] = 4.0;
    kernel[20] = 1.0; kernel[21] = 4.0; kernel[22] = 7.0; kernel[23] = 4.0; kernel[24] = 1.0;

    const float kernelSum = 273.0; 

    float offset = 1.0 / 512.0; 

    vec4 result = vec4(0.0);
    int index = 0;
    
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 sampleCoord = coord + vec2(i, j) * 1.0/size;
            result += texture2D(sampler, sampleCoord) * kernel[index];
            index++;
        }
    }

    return result / kernelSum;
}


vec4 lanczos(sampler2D sampler, vec2 coord)
{  
    vec3 color;
    vec3 E;

    vec2 qt_TexCoord0 = coord;

    vec2 dx = vec2(1.0, 0.0);
    vec2 dy = vec2(0.0, 1.0);

    vec2 pc = qt_TexCoord0 * size / lanczos2_res;
    vec2 tex = (floor(pc) + vec2(0.5, 0.5)) * lanczos2_res / size;

    vec2 tc = (floor(pc-vec2(0.5,0.5))+vec2(0.5,0.5));

    mat4 weights;
    weights[0] = resampler(vec4(d(pc, tc    -dx    -dy), d(pc, tc         -dy), d(pc, tc    +dx    -dy), d(pc, tc+2.0*dx    -dy)));
    weights[1] = resampler(vec4(d(pc, tc    -dx     ), d(pc, tc          ), d(pc, tc    +dx     ), d(pc, tc+2.0*dx     )));
    weights[2] = resampler(vec4(d(pc, tc    -dx    +dy), d(pc, tc         +dy), d(pc, tc    +dx    +dy), d(pc, tc+2.0*dx    +dy)));
    weights[3] = resampler(vec4(d(pc, tc    -dx+2.0*dy), d(pc, tc     +2.0*dy), d(pc, tc    +dx+2.0*dy), d(pc, tc+2.0*dx+2.0*dy)));

    dx = dx * lanczos2_res / size;
    dy = dy * lanczos2_res / size;
    tc = tc * lanczos2_res / size;

    // reading the texels
    vec3 c00 = texture2D(sampler, tc    -dx    -dy).xyz;
    vec3 c10 = texture2D(sampler, tc         -dy).xyz;
    vec3 c20 = texture2D(sampler, tc    +dx    -dy).xyz;
    vec3 c30 = texture2D(sampler, tc+2.0*dx    -dy).xyz;
    vec3 c01 = texture2D(sampler, tc    -dx     ).xyz;
    vec3 c11 = texture2D(sampler, tc          ).xyz;
    vec3 c21 = texture2D(sampler, tc    +dx     ).xyz;
    vec3 c31 = texture2D(sampler, tc+2.0*dx     ).xyz;
    vec3 c02 = texture2D(sampler, tc    -dx    +dy).xyz;
    vec3 c12 = texture2D(sampler, tc         +dy).xyz;
    vec3 c22 = texture2D(sampler, tc    +dx    +dy).xyz;
    vec3 c32 = texture2D(sampler, tc+2.0*dx    +dy).xyz;
    vec3 c03 = texture2D(sampler, tc    -dx+2.0*dy).xyz;
    vec3 c13 = texture2D(sampler, tc     +2.0*dy).xyz;
    vec3 c23 = texture2D(sampler, tc    +dx+2.0*dy).xyz;
    vec3 c33 = texture2D(sampler, tc+2.0*dx+2.0*dy).xyz;

    color = E = texture2D(sampler, qt_TexCoord0).xyz;

    vec3 F6 = texture2D(sampler, tex +dx+0.25*dx+0.25*dy).xyz;
    vec3 F7 = texture2D(sampler, tex +dx+0.25*dx-0.25*dy).xyz;
    vec3 F8 = texture2D(sampler, tex +dx-0.25*dx-0.25*dy).xyz;
    vec3 F9 = texture2D(sampler, tex +dx-0.25*dx+0.25*dy).xyz;

    vec3 H6 = texture2D(sampler, tex +0.25*dx+0.25*dy+dy).xyz;
    vec3 H7 = texture2D(sampler, tex +0.25*dx-0.25*dy+dy).xyz;
    vec3 H8 = texture2D(sampler, tex -0.25*dx-0.25*dy+dy).xyz;
    vec3 H9 = texture2D(sampler, tex -0.25*dx+0.25*dy+dy).xyz;

    vec4 f0 = reduce4(F6, F7, F8, F9);
    vec4 h0 = reduce4(H6, H7, H8, H9);

    //  Get min/max samples
    vec3 min_sample = min4(c11, c21, c12, c22);
    vec3 max_sample = max4(c11, c21, c12, c22);

    color = weights[0] * transpose(mat4x3(c00, c10, c20, c30));
    color += weights[1] * transpose(mat4x3(c01, c11, c21, c31));
    color += weights[2] * transpose(mat4x3(c02, c12, c22, c32));
    color += weights[3] * transpose(mat4x3(c03, c13, c23, c33));
    color = color / dot(vec4(1.0) * weights, vec4(1.0));

    // Anti-ringing
    vec3 aux = color;
    color = clamp(color, min_sample, max_sample);

    color = mix(aux, color, lanczos2_ar_strength);

    float alpha = texture2D(sampler, qt_TexCoord0).a;
    return vec4(color.xyz, alpha);
}

void main()
{
    // read texel from the current texture
    vec2 coord = gl_TexCoord[0].xy;
    coord.y = 1.0 - coord.y;
    coord = zoomed(coord);

    vec4 color1 = lanczos(curr, coord);
    vec4 gaussian1 = gaussianAt(curr, coord);

    float amount = 0.8;

    if (zoom > 1.0) {

        // check if a corresponding texel exists in the next texture
        vec2 coord2 = 2.0 * coord - vec2(0.5,0.5);
        if (coord2.x >= 0.0 && coord2.x <= 1.0 && coord2.y >= 0.0 && coord2.y <= 1.0) {

            // read the corresponding texel from the next texture
            vec4 color2 = lanczos(next, coord2);
            vec4 gaussian2 = gaussianAt(curr, coord);


            vec4 finalColor2 = (color2 * (1.0-amount)) + (gaussian2 * (1.0 * amount));

            // interpolate between both texels
            color1 = mix(color1, color2, zoom - 1.0);
        }
    }

    vec4 finalColor = (color1 * (1.0-amount)) + (gaussian1 * (1.0 * amount));
    gl_FragColor = gl_Color * finalColor;
}
