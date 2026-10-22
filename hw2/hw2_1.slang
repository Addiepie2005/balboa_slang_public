[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    float2 e01 = sp1 - sp0;
    float2 e12 = sp2 - sp1;
    float2 e20 = sp0 - sp2;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float2 p = ipixel + 0.5;

    float m01 = dot((p - sp0), n01);
    float m12 = dot((p - sp1), n12);
    float m20 = dot((p - sp2), n20);

    if(m01 < 0 && m12 < 0 && m20 < 0 || (m01 >= 0 && m12 >= 0 && m20 >= 0)){
        image[ipixel] = float4(color, 1.0);
    }
}
