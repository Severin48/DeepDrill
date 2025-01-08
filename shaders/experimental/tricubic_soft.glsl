// Sampler for the current texture
uniform sampler2D curr;

// Sampler for the next texture
uniform sampler2D next;

// Texture size
uniform vec2 size;

// Zoom factor
uniform float zoom;

vec2 zoomed(vec2 coord)
{
    return (coord / zoom) + 0.5 - (0.5 / zoom);
}

//
// From https://stackoverflow.com/questions/13501081/efficient-bicubic-filtering-code-in-glsl
//

vec4 cubic(float v)
{
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}

vec4 bicubic(sampler2D sampler, vec2 texCoords)
{
    vec2 texSize = size;
    vec2 invTexSize = 1.0 / texSize;

    texCoords = texCoords * texSize - 0.5;


    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture2D(sampler, offset.xz);
    vec4 sample1 = texture2D(sampler, offset.yz);
    vec4 sample2 = texture2D(sampler, offset.xw);
    vec4 sample3 = texture2D(sampler, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
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

void main()
{
    // Read texel from the current texture
    vec2 coord = gl_TexCoord[0].xy;
    coord.y = 1.0 - coord.y;
    coord = zoomed(coord);

    float amount = 0.75;

    vec4 color1 = bicubic(curr, coord);
    vec4 gaussian1 = gaussianAt(curr, coord);
    vec4 finalColor1 = mix(color1, gaussian1, amount);



    // Check if a corresponding texel exists in the next texture
    vec2 coord2 = 2.0 * coord - vec2(0.5,0.5);
    if (coord2.x >= 0.0 && coord2.x <= 1.0 && coord2.y >= 0.0 && coord2.y <= 1.0) {

        // Read the corresponding texel from the next texture
        vec4 color2 = bicubic(next, coord2);
        vec4 gaussian2 = gaussianAt(curr, coord);
        // mix 
        //vec4 finalColor2 = (color2 * (1.0-amount)) + (gaussian2 * (1.0 * amount));
        vec4 finalColor2 = mix(color2, gaussian2, amount);

        // Interpolate between both texels
        finalColor1 = mix(finalColor1, finalColor2, zoom - 1.0);
    }

    gl_FragColor = gl_Color * finalColor1;
}