// -----------------------------------------------------------------------------
// Fragmentshader for scaling with Lanczos kernel.
//
// Parameter to tweak:
// A: Size of the Lanczos window and effects the dampening at the edge
// B: Multiplier to change frequency of sinc
// With A=2.0 and B=1.5 result looks simelar to bilinear scalilng but sharper
// -----------------------------------------------------------------------------

uniform sampler2D curr;
uniform sampler2D next;
uniform vec2 size;
uniform float zoom;

#define PI (3.1415926535897932384626433)
// Size of Lanczos window damping factor 
#define A (2.0)
// Multiplier in sinc for higher frequency
#define B (1.5)
const int Aint = int(A);

vec2 zoomed(vec2 coord) {
    return (coord / zoom) + 0.5 - (0.5 / zoom);
}

float sinc(float x) {
    if (x == 0.0) return 1.0;
    x = B * PI * x;
    return sin(x)/(x);
}

float lanczosAt(float x) {
    if (x == 0.0) return 1.0;
    if (abs(x) >= A) return 0.0;
    return (sinc(x) * sinc(x/A));
}

vec4 lanczos(sampler2D sampler, vec2 coord) {
    vec4 color = vec4(0.0);
    float weightSum = 0.0;

    vec2 texCoord = coord * size;
    vec2 texBase = floor(texCoord - 0.5) + 0.5;
    
    // window of size [-A;A] x [-A;A]
    for (int j = -Aint + 1; j <= Aint; ++j) {
        for (int i = -Aint + 1; i <=Aint; ++i) {
            vec2 offset = vec2(float(i), float(j));
            // sample position in pixel space
            vec2 tap = texBase + offset;
            vec2 sincDist = (tap - texCoord) / A;

            // 2D lanczos: L(x) * L(y) 
            float weight = lanczosAt(sincDist.x) * lanczosAt(sincDist.y);
            
            color += texture2D(sampler, tap / size) * weight;
            weightSum += weight;
        }
    }
    
    return color / weightSum;
}

void main() {
    // Read texel from the current texture
    vec2 coord = gl_TexCoord[0].xy;
    coord.y = 1.0 - coord.y;
    coord = zoomed(coord);

    vec4 color1 = lanczos(curr, coord);

    if (zoom > 1.0) {

        // Check if a corresponding texel exists in the next texture
        vec2 coord2 = 2.0 * coord - vec2(0.5,0.5);
        if (coord2.x >= 0.0 && coord2.x <= 1.0 && coord2.y >= 0.0 && coord2.y <= 1.0) {

            // Read the corresponding texel from the next texture
            vec4 color2 = lanczos(next, coord2);

            // Interpolate between both texels (linear)
            color1 = mix(color1, color2, zoom - 1.0);
        }
    }

    gl_FragColor = gl_Color * color1;
}
