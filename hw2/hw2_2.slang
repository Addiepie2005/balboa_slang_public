float x_screen_to_camera(float x, float s, float a, float width){
    return ((x * 2 * s * a) / width) + (s * a);
}

float y_screen_to_camera(float y, float s, float a, float height){
    return height - 1 - (((y * 2 * s) / height) + s);
}


[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
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

    // If we are NOT in the triangle, return
    if(!(m01 < 0 && m12 < 0 && m20 < 0 || (m01 >= 0 && m12 >= 0 && m20 >= 0))){
        return;
    }

    // projected camera space
    float3 sp0_camera = float3(x_screen_to_camera(sp0.x, s, a, width), y_screen_to_camera(sp0.y, s, a, height), -1);
    float3 sp1_camera = float3(x_screen_to_camera(sp1.x, s, a, width), y_screen_to_camera(sp1.y, s, a, height), -1);
    float3 sp2_camera = float3(x_screen_to_camera(sp2.x, s, a, width), y_screen_to_camera(sp2.y, s, a, height), -1);

    float3 p_camera = float3(x_screen_to_camera(p.x, s, a, width), y_screen_to_camera(p.y, s, a, height), -1);

    float camera_whole_area = length(cross((sp1_camera - sp0_camera), (sp2_camera - sp1_camera))) / 2;
    float camera_b0_area = length(cross((sp1_camera - p_camera), (sp2_camera - sp1_camera))) / 2;
    float camera_b1_area = length(cross((p_camera - sp0_camera), (sp2_camera - p_camera))) / 2;
    float camera_b2_area = length(cross((sp1_camera - sp0_camera), (p_camera - sp1_camera))) / 2;
    
    float b0_camera = camera_b0_area/camera_whole_area;
    float b1_camera = camera_b1_area/camera_whole_area;
    float b2_camera = camera_b2_area/camera_whole_area;

    float b0_original = (b0_camera * inv_depth0) / ((b0_camera * inv_depth0) + (b1_camera * inv_depth1) + (b2_camera * inv_depth2));
    float b1_original = (b1_camera * inv_depth1) / ((b0_camera * inv_depth0) + (b1_camera * inv_depth1) + (b2_camera * inv_depth2));
    float b2_original = (b2_camera * inv_depth2) / ((b0_camera * inv_depth0) + (b1_camera * inv_depth1) + (b2_camera * inv_depth2));

    float z_original = -((b0_original * 1/inv_depth0) + (b1_original * 1/inv_depth1) + (b2_original * 1/inv_depth2));

    float3 c = (b0_original * c0) + (b1_original * c1) + (b2_original * c2);

    if(z_original < depth_image[ipixel]){
        depth_image[ipixel] = z_original;
        color_image[ipixel] = float4(c, 1.0);
    }
}
