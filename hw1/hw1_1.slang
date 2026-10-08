[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // TODO: your code here
    
    float2 p = float2(tid.x + 0.5, (height - 1 - tid.y) + 0.5);
    if (distance(p, center) <= radius) {
        image[tid] = float4(color, 1.0);
    } else {
        image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    }
}
