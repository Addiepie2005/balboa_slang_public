bool intersect_sphere(float3 origin, Sphere sphere, float3 ray, out float2 t){
    float3 center_to_origin = origin - sphere.center;
    float a = dot(ray, ray);
    float b = 2 * dot(center_to_origin, ray);
    float c = dot(center_to_origin, center_to_origin) - (sphere.radius * sphere.radius);

    return solve_quadratic(a, b, c, t);
}


import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
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
    float3 ray = normalize(float3(px, py, -1.0));
    
    Shape closest_shape;
    Sphere closest_sphere;
    float closest_depth = 1e30;
    bool hit = false;

    for (int i = 0; i < num_shapes; i++){
        //intersect with the scene geometry
        Shape shape = shapes[i];
        Sphere sphere = spheres[shape.index];
        
        float2 t;
        //Skip if there are no roots
        if (!intersect_sphere(float3(0.0), sphere, ray, t)) {
            continue;
        }

        //choose closest intersection that's not < EPSILON
        float t0 = min(t.x, t.y);
        float t1 = max(t.x, t.y);
        float closest_intersect = (t0 > EPSILON) ? t0 : t1;
        if (closest_intersect <= EPSILON) {
            continue; 
        }

        //find closest hit among many spheres
        if(closest_intersect < closest_depth){
            closest_shape = shape;
            closest_sphere = sphere;
            closest_depth = closest_intersect;
            hit = true;
        }
    }

    if (!hit) return;

    float3 hit_point = float3(0,0,0) + closest_depth * ray;
    float3 Kd = closest_shape.Kd;
    float3 color = float3(0.0);

    // handle shadows
    for(int i = 0; i < num_lights; i++){
        PointLight light = lights[i];

        float3 shadow_ray = normalize(light.position - hit_point);
        float d = length(light.position - hit_point);
        bool shadow = false;

        for(int j = 0; j < num_shapes; j++){
            Shape shape = shapes[j];

            Sphere sphere = spheres[shape.index];

            float2 t;
            //Skip if there are no roots
            if (!intersect_sphere(hit_point, sphere, shadow_ray, t)) {
                continue;
            }

            // if the furthest intersection point is further than epsilon, it casts shadow
            float t0 = min(t.x, t.y);
            float t1 = max(t.x, t.y);
            float intersect = (t0 > EPSILON) ? t0 : t1;
            if (intersect >= EPSILON && intersect <= d) {
                shadow = true;
                break;
            }
        }

        // If this point light is not blocked, add its light contribution 
        if(!shadow){
            float3 n = normalize(hit_point - closest_sphere.center);
            float3 l = shadow_ray;
            color += (light.intensity / (d * d)) * (Kd * max(dot(n, l), 0));
        }
    }

    
    image[tid] = float4(color, 1.0); 
}
