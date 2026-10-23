float x_screen_to_camera(float x, float s, float a, float width){
    return ((x * 2 * s * a) / width) + (s * a);
}

float y_screen_to_camera(float y, float s, float a, float height){
    return height - 1 - (((y * 2 * s) / height) + s);
}

float3 compute_barycentric(float2 p, float3 sp0, float3 sp1, float3 sp2, float s, float a, float width, float height){
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

    float b0_original = (b0_camera * sp0.z) / ((b0_camera * sp0.z) + (b1_camera * sp1.z) + (b2_camera * sp2.z));
    float b1_original = (b1_camera * sp1.z) / ((b0_camera * sp0.z) + (b1_camera * sp1.z) + (b2_camera * sp2.z));
    float b2_original = (b2_camera * sp2.z) / ((b0_camera * sp0.z) + (b1_camera * sp1.z) + (b2_camera * sp2.z));

    return float3(b0_original, b1_original, b2_original);
}

bool in_triangle(float3 sp0, float3 sp1, float3 sp2, float2 p){
    float2 e01 = sp1.xy - sp0.xy;
    float2 e12 = sp2.xy - sp1.xy;
    float2 e20 = sp0.xy - sp2.xy;

    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);

    float m01 = dot((p - sp0.xy), n01);
    float m12 = dot((p - sp1.xy), n12);
    float m20 = dot((p - sp2.xy), n20);

    // return true if we are in the triangle
    return (m01 < 0 && m12 < 0 && m20 < 0 || (m01 >= 0 && m12 >= 0 && m20 >= 0));
}

bool is_degenerate(float3 sp0, float3 sp1, float3 sp2){
    float3 e01 = sp1 - sp0;
    float3 e02 = sp2 - sp0;
    float area = length(cross(e01, e02)) / 2;
    return !(area > 1e-8);   // also catches NaN
}


[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    // HW 2.5 TODO

    if (tid >= num_triangles) return;

    float3 sp0 = projected_vertices[faces[tid][0]];
    float3 sp1 = projected_vertices[faces[tid][1]];
    float3 sp2 = projected_vertices[faces[tid][2]];
    if (is_degenerate(sp0, sp1, sp2)){
        return;
    } 
    
    int x_min = max(0, (int)floor(fmin3(sp0.x, sp1.x, sp2.x)));
    int y_min = max(0, (int)floor(fmin3(sp0.y, sp1.y, sp2.y)));
    int x_max = min(width - 1, (int)ceil(fmax3(sp0.x, sp1.x, sp2.x)));
    int y_max = min(height - 1, (int)ceil(fmax3(sp0.y, sp1.y, sp2.y)));

    for (int x = x_min; x <= x_max; x++){
        for (int y = y_min; y <= y_max; y++){
            float2 p = float2(x + 0.5, y + 0.5);
            if(in_triangle(sp0, sp1, sp2, p)){
                float3 barycentric = compute_barycentric(p, sp0, sp1, sp2, s, a, width, height);
                float depth = (barycentric[0] * 1/sp0.z) + (barycentric[1] * 1/sp1.z) + (barycentric[2] * 1/sp2.z);
                uint depth_bits = asuint(depth);
                InterlockedMin(depth_buffer[y * width + x], depth_bits);
                
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    // HW 2.5 TODO

    if (tid >= num_triangles) return;

    float3 sp0 = projected_vertices[faces[tid][0]];
    float3 sp1 = projected_vertices[faces[tid][1]];
    float3 sp2 = projected_vertices[faces[tid][2]];
    if (is_degenerate(sp0, sp1, sp2)){
        return;
    } 
    
    int x_min = max(0, (int)floor(fmin3(sp0.x, sp1.x, sp2.x)));
    int y_min = max(0, (int)floor(fmin3(sp0.y, sp1.y, sp2.y)));
    int x_max = min(width - 1, (int)ceil(fmax3(sp0.x, sp1.x, sp2.x)));
    int y_max = min(height - 1, (int)ceil(fmax3(sp0.y, sp1.y, sp2.y)));

    for (int x = x_min; x <= x_max; x++){
        for (int y = y_min; y <= y_max; y++){
            float2 p = float2(x + 0.5, y + 0.5);
            if(in_triangle(sp0, sp1, sp2, p)){
                float3 barycentric = compute_barycentric(p, sp0, sp1, sp2, s, a, width, height);
                float depth = (barycentric[0] * 1/sp0.z) + (barycentric[1] * 1/sp1.z) + (barycentric[2] * 1/sp2.z);
                if(asuint(depth) == depth_buffer[y * width + x]){
                    InterlockedMin(triangle_id_buffer[y * width + x], tid);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // HW 2.5 TODO

    uint face_id = triangle_id_buffer[tid.y * width + tid.x];

    if(face_id != 0xffffffff){
        
        float3 sp0 = projected_vertices[faces[face_id][0]];
        float3 sp1 = projected_vertices[faces[face_id][1]];
        float3 sp2 = projected_vertices[faces[face_id][2]];
        
        float2 p = float2(tid.x + 0.5, tid.y + 0.5);

        float3 barycentric = compute_barycentric(p, sp0, sp1, sp2, s, a, width, height);

        float3 c = (barycentric[0] * vertex_colors[faces[face_id][0]]) + (barycentric[1] * vertex_colors[faces[face_id][1]]) + (barycentric[2] * vertex_colors[faces[face_id][2]]);

        image[tid] = float4(c, 1.0);
    }

}
