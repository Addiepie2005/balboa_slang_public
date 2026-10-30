import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);   

    //Shoot a ray from the camera origin towards pixel on the image plane
    float a_ratio = float(width) / float(height);
    float px = (2.0 * (tid.x + 0.5) / width - 1.0) * a_ratio;
    float py = 1.0 - 2.0 * (tid.y + 0.5) / height;
    float3 ray = float3(px, py, -1.0);
    
    Sphere closest_sphere_so_far;
    float closest_depth_so_far = 1e30;
    bool hit = false;

    for (int i = 0; i < num_spheres; i ++){
        //intersect with the scene geometry
        Sphere sphere = spheres[i];
        
        float3 origin_to_center = float3(0,0,0) - sphere.center;
        float a = dot(ray, ray);
        float b = 2 * dot(origin_to_center, ray);
        float c = dot(origin_to_center, origin_to_center) - (sphere.radius * sphere.radius);

        float2 t;
        //Skip if there are no roots
        if (!solve_quadratic(a, b, c, t)) {
            continue;
        }

        //choose closest intersection that's not < 0
        float t0 = min(t.x, t.y);
        float t1 = max(t.x, t.y);
        float closest_intersect = (t0 > 0) ? t0 : t1;
        if (closest_intersect <= 0) {
            continue; 
        }

        //find closest hit among many spheres
        if(closest_intersect < closest_depth_so_far){
            closest_sphere_so_far = sphere;
            closest_depth_so_far = closest_intersect;
            hit = true;
        }
    }

    if (!hit) return;

    //visualize the color using the sphere’s normal
    float3 hit_point = float3(0,0,0) + closest_depth_so_far * ray;
    float3 normal = normalize(hit_point - closest_sphere_so_far.center);

    float3 color = float3((normal + 1)/2);
    
    image[tid] = float4(color, 1.0);   
}
