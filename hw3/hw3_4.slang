bool intersect_sphere(float3 origin, Sphere sphere, float3 dir, out float2 t){
    float3 center_to_origin = origin - sphere.center;
    float a = dot(dir, dir);
    float b = 2 * dot(center_to_origin, dir);
    float c = dot(center_to_origin, center_to_origin) - (sphere.radius * sphere.radius);

    return solve_quadratic(a, b, c, t);
}

import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // find the width and height of the image plane
    float a_ratio = float(width) / float(height);
    float F = radians(camera.fov_x);
    float image_width = 2 * tan(F / 2.0);
    float image_height = image_width / a_ratio;

    // Convert to image plane / camera space 
    float px = (2.0 * (tid.x + 0.5) / width - 1.0) * image_width / 2;
    float py = (1.0 - 2.0 * (tid.y + 0.5) / height) * image_height / 2;

    // these two components make up our ray
    float3 origin = float3(0.0);
    float3 dir = float3(px, py, -1.0);


    // apply transformations
    // transform camera ray to world space
    origin = xform_pt(camera.cam_to_world, origin);
    dir = xform_vec(camera.cam_to_world, dir);

    
    float3 color = float3(0.0);
    float3 throughput = float3(1.0);

    for (int b = 0; b < max_depth; b++) {    
        Shape closest_shape;
        Sphere closest_sphere;
        float closest_depth = 1e30;
        bool hit = false;

        //Shoot a ray from the camera origin towards pixel on the image plane
        for (int i = 0; i < num_shapes; i++){
            //intersect with the scene geometry
            Shape shape = shapes[i];
            Sphere sphere = spheres[shape.index];

            
            // transform the ray to object space
            float3 obj_origin = xform_pt(shape.world_to_obj, origin);
            float3 obj_dir = xform_vec(shape.world_to_obj, dir);
            
            float2 t;
            //Skip if there are no roots
            if (!intersect_sphere(obj_origin, sphere, obj_dir, t)) {
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

        if (!hit){
            color += throughput * background;
            break;
        }

        float3 hit_point = origin + closest_depth * dir;
        float3 Kd = closest_shape.Kd;
        float3 Kr = closest_shape.Kr;
        // transform hit_point into object space to calculate normal
        float3 obj_origin = xform_pt(closest_shape.world_to_obj, origin);
        float3 obj_dir = xform_vec(closest_shape.world_to_obj, dir);
        float3 obj_hit_point = obj_origin + closest_depth * obj_dir;

        // transform normal with inverse transpose of obj_to_world, which is the transpose of world_to_object
        float3 n = normalize(xform_vec(transpose(closest_shape.world_to_obj), obj_hit_point - closest_sphere.center));

        // handle shadows
        // for each point light
        for(int i = 0; i < num_lights; i++){
            PointLight light = lights[i];

            float3 shadow_origin = hit_point;
            float3 shadow_dir = normalize(light.position - shadow_origin);

            float d = length(light.position - shadow_origin);
            bool shadow = false;

            // for each shape that could possibly cast a shadow
            for(int j = 0; j < num_shapes; j++){
                Shape shape = shapes[j];

                Sphere sphere = spheres[shape.index];
                
                // transform the ray to object space
                float3 obj_shadow_origin = xform_pt(shape.world_to_obj, shadow_origin);
                float3 obj_shadow_dir = xform_vec(shape.world_to_obj, shadow_dir);

                float2 t;
                //Skip if there are no roots
                if (!intersect_sphere(obj_shadow_origin, sphere, obj_shadow_dir, t)) {
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
                float3 l = shadow_dir; 
                color += throughput * (light.intensity / (d * d)) * (Kd * max(dot(n, l), 0));
            }
        }

        throughput *= Kr;
        origin = hit_point;
        dir = reflect(dir, n);
        
    }

    image[tid] = float4(color, 1.0); 
}
