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
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
                  StructuredBuffer<BVHNode> bvh_nodes,
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
        float closest_depth = 1e30;
        float3 obj_closest_n = float3(0.0);
        bool hit = false;

        //Shoot a ray from the camera origin towards pixel on the image plane
        for (int i = 0; i < num_shapes; i++){
            //intersect with the scene geometry
            Shape shape = shapes[i];

            float closest_intersect;      
            float3 obj_origin = xform_pt(shape.world_to_obj, origin);
            float3 obj_dir = xform_vec(shape.world_to_obj, dir);

            switch(shape.type){
                case ShapeType.Sphere:
                    
                    Sphere sphere = spheres[shape.index];
                    // transform the ray to object space

                    float2 t;
                    //Skip if there are no roots
                    if (!intersect_sphere(obj_origin, sphere, obj_dir, t)) {
                        continue;
                    }

                    //choose closest intersection that's not < EPSILON
                    float t0 = min(t.x, t.y);
                    float t1 = max(t.x, t.y);
                    closest_intersect = (t0 > EPSILON) ? t0 : t1;
                    if (closest_intersect <= EPSILON) {
                        continue; 
                    }

                    //find closest hit among many shapes
                    if(closest_intersect < closest_depth){
                        closest_shape = shape;
                        closest_depth = closest_intersect;
                        obj_closest_n = (obj_origin + closest_depth * obj_dir) - sphere.center;
                        hit = true;
                    }

                    break;

                case ShapeType.TriangleMesh:

                    TriangleMesh mesh = meshes[shape.index];
                    int root = mesh.bvh_root_index;
                    int end = bvh_nodes[root].next_index;

                    int index = root;
                    while(index >= 0 && index < end){
                        BVHNode node = bvh_nodes[index];

                        if(!intersect_aabb(node.bbox_min, node.bbox_max, obj_origin, obj_dir, EPSILON, closest_depth)){
                            index = node.next_index;
                            continue;
                        }

                        if(node.is_leaf){
                            for (uint j = node.triangle_begin; j < node.triangle_end; j++){
                                uint3 tri = triangles[j] + mesh.vertex_offset;
                                float3 p0 = vertices[tri.x];
                                float3 p1 = vertices[tri.y];
                                float3 p2 = vertices[tri.z];

                                float3 n0 = vertex_normals[tri.x];
                                float3 n1 = vertex_normals[tri.y];
                                float3 n2 = vertex_normals[tri.z];
                                
                                float t;
                                float3 barycentrics;
                                if (intersect_tri(p0, p1, p2, obj_origin, obj_dir, EPSILON, closest_depth, barycentrics, t)){
                                    closest_shape = shape;
                                    closest_depth = t;
                                    obj_closest_n = barycentrics[0]*n0 + barycentrics[1]*n1 + barycentrics[2]*n2;
                                    hit = true;
                                }
                            }
                            index = node.next_index;
                        } else {
                            index = index + 1;
                        }
                    }
                    

                    
                    break;
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
        float3 n = normalize(xform_vec(transpose(closest_shape.world_to_obj), obj_closest_n));


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
                // transform the ray to object space
                float3 obj_shadow_origin = xform_pt(shape.world_to_obj, shadow_origin);
                float3 obj_shadow_dir = xform_vec(shape.world_to_obj, shadow_dir);

                switch(shape.type){
                    case ShapeType.Sphere:

                        Sphere sphere = spheres[shape.index];
                        
                        float2 t;
                        //Skip if there are no roots
                        if (intersect_sphere(obj_shadow_origin, sphere, obj_shadow_dir, t)) {
                            float t0 = min(t.x, t.y);
                            float t1 = max(t.x, t.y);
                            float intersect = (t0 > EPSILON) ? t0 : t1;
                            if (intersect >= EPSILON && intersect <= d) {
                                shadow = true;
                            }
                        }

                        break;

                    case ShapeType.TriangleMesh:
                        TriangleMesh mesh = meshes[shape.index];

                        int root = mesh.bvh_root_index;
                        int end = bvh_nodes[root].next_index;

                        int index = root;
                        while(index >= 0 && index < end && !shadow){
                            BVHNode node = bvh_nodes[index];

                            if(!intersect_aabb(node.bbox_min, node.bbox_max, obj_shadow_origin, obj_shadow_dir, EPSILON, d)){
                                index = node.next_index;
                                continue;
                            }

                            if(node.is_leaf){
                                for (uint j = node.triangle_begin; j < node.triangle_end; j++){
                                    uint3 tri = triangles[j] + mesh.vertex_offset;
                                    float3 p0 = vertices[tri.x];
                                    float3 p1 = vertices[tri.y];
                                    float3 p2 = vertices[tri.z];

                                    float t;
                                    float3 barycentrics;
                                    if (intersect_tri(p0, p1, p2, obj_shadow_origin, obj_shadow_dir, EPSILON, d, barycentrics, t)){
                                        shadow = true;
                                        break;
                                    }
                                }
                                index = node.next_index;
                            } else {
                                index = index + 1;
                            }
                        }
                        
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
