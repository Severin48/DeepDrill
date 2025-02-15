// -----------------------------------------------------------------------------
// Fragmentshader for scaling linear and sharpen step (USM).
//
// Parameter to tweak:
// AMOUNT: used to determine how much the sharped Image is added to to the
// original 
// SD: standard deviation in gaussian blur
// With AMOUNT=0.5 and SD=1.0 result looks little sharper than vanilla
// bilinear interpolation
// -----------------------------------------------------------------------------

uniform sampler2D curr;
uniform sampler2D next;
uniform vec2 size;
uniform float zoom;

#define PI (3.1415926535897932384626433)
#define AMOUNT (0.5)
#define SD (1.0)

vec2 zoomed(vec2 coord) {
    return (coord / zoom) + 0.5 - (0.5 / zoom);
}

float gauss(float x, float sx){
    float arg = x;
    arg = -1./2.*arg*arg/sx;
    
    float a = 1./(pow(2.*3.1415*sx, 0.5));
    
    return a*exp(arg);
}

vec4 soft(sampler2D sampler, vec2 coord) {
    vec4 color = vec4(0.0);
    float weightSum = 0.0;
    int Aint = 8;
    float scale = size.x/size.y;

    vec2 texCoord = coord * size;
    vec2 texBase = floor(texCoord - 0.5) + 0.5;
    for (int j = -Aint + 1; j <= Aint; ++j) {
        for (int i = -Aint + 1; i <=Aint; ++i) {
            vec2 offset = vec2(float(i), float(j));
            vec2 tap = texBase + offset;
            vec2 sincDist = (tap - texCoord);

            float weight = gauss(sincDist.x, SD) * gauss(sincDist.y, SD);
            
            color += texture2D(sampler, tap / size) * weight;
            weightSum += weight;
        }
    }
    
    return color / weightSum;
}

vec4 bilinear(sampler2D sampler, vec2 coord)
{
    float dx = 1.0 / size.x;
    float dy = 1.0 / size.y;

    vec4 p0q0 = texture2D(sampler, coord);
    vec4 p1q0 = texture2D(sampler, coord + vec2(dx, 0));
    vec4 p0q1 = texture2D(sampler, coord + vec2(0, dy));
    vec4 p1q1 = texture2D(sampler, coord + vec2(dx,dy));

    float a = fract(coord.x * size.x);
    vec4 pInterp_q0 = mix(p0q0, p1q0, a);
    vec4 pInterp_q1 = mix(p0q1, p1q1, a);

    float b = fract(coord.y * size.y);
    vec4 pInterp = mix(pInterp_q0, pInterp_q1, b);

    return pInterp;
}

vec4 USM(sampler2D sampler, vec2 coord) {
    vec3 original = bilinear(sampler,coord).xyz;
    vec3 soft = soft(sampler,coord).xyz;
    vec3 result = original + ((original-soft)*AMOUNT);
    return vec4(result, 1.0);
}

void main() {
    // Read texel from the current texture
    vec2 coord = gl_TexCoord[0].xy;
    coord.y = 1.0 - coord.y;
    coord = zoomed(coord);

    vec4 color1 = USM(curr, coord);

    if (zoom > 1.0) {

        // Check if a corresponding texel exists in the next texture
        vec2 coord2 = 2.0 * coord - vec2(0.5,0.5);
        if (coord2.x >= 0.0 && coord2.x <= 1.0 && coord2.y >= 0.0 && coord2.y <= 1.0) {

            // Read the corresponding texel from the next texture
            vec4 color2 = USM(next, coord2);

            // Interpolate between both texels (linear)
            color1 = mix(color1, color2, zoom - 1.0);
        }
    }

    gl_FragColor = gl_Color * color1;
}
