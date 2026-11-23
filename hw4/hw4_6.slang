struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

float sphere_sdf(float3 center, float r, float3 p) { 
    return length(p - center) - r;
}

float plane_sdf(float4 plane, float3 p){
    return dot(plane.xyz, p) - plane.w;
}

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}


[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.6. Your code below.

    // loop through all spheres and find the sphere with the smallest signed distance
    float smallest_sphere_dist = 1.0/0.0;
    float4 closest_sphere;
    for(int i = 0; i < num_spheres; i++){
        float4 sphere = spheres[i];
        float signed_dist = sphere_sdf(sphere.xyz, sphere.w, positions[tid]);
        if(signed_dist < smallest_sphere_dist){
            smallest_sphere_dist = signed_dist;
            closest_sphere = sphere;
        }
    }

    // loop through all planes and find the plane with the smallest signed distance
    float smallest_plane_dist = 1.0/0.0;
    float4 closest_plane;
    for(int i = 0; i < num_planes; i++){
        float4 plane = planes[i];
        float signed_dist = plane_sdf(plane, positions[tid]);
        if(signed_dist < smallest_plane_dist){
            smallest_plane_dist = signed_dist;
            closest_plane = plane;
        }
    }

    // loop through all triangles and find the triangle with the smallest signed distance
    float smallest_tri_dist = 1.0/0.0;
    float4 tri_plane = float4(0.0);
    uint3 closest_triangle = uint3(0, 0, 0);
    float min_dist = contact_margin + collision_distance;
    bool found_tri = false;

    int node_index = 0;
    while(node_index >= 0){
        BVHNode node = bvh_nodes[node_index];
        if (point_aabb_distance(positions[tid], node.bbox_min, node.bbox_max) > min_dist){
            node_index = node.next_index;
            continue;
        }

        if(node.is_leaf){
            for(int i = node.triangle_begin; i < node.triangle_end; i ++){
                uint3 tri = mesh_triangles[i];
                //Get vertices of current triangle
                float3 p0 = mesh_vertices[tri.x];
                float3 p1 = mesh_vertices[tri.y];
                float3 p2 = mesh_vertices[tri.z];

                ClosestPoint curr_point = closest_point_triangle(positions[tid], p0, p1, p2);
                float curr_length = length(positions[tid] - curr_point.position);

                if(curr_length < min_dist){
                    min_dist = curr_length;
                    closest_triangle = tri;
                    found_tri = true;
                }
            }
            //So that the while loop condition is no longer met and we stop
            node_index = node.next_index;
        } else {
            node_index += 1;
        }
    }

    if(found_tri){
        // Face normal
        float3 p0 = mesh_vertices[closest_triangle.x];
        float3 p1 = mesh_vertices[closest_triangle.y];
        float3 p2 = mesh_vertices[closest_triangle.z];
        float3 face_norm = normalize(cross(p1 - p0, p2 - p0));
    
        // Phong normal    
        float3 n0 = mesh_vertex_normals[closest_triangle.x];
        float3 n1 = mesh_vertex_normals[closest_triangle.y];
        float3 n2 = mesh_vertex_normals[closest_triangle.z];
        ClosestPoint closest_point = closest_point_triangle(positions[tid], p0, p1, p2);
        float3 phong_norm = normalize(closest_point.bary.x * n0 + closest_point.bary.y * n1 + closest_point.bary.z * n2);
        if (dot(face_norm, phong_norm) < 0.0) {
            face_norm = -face_norm;
        }
        // Use face_norm as the normal of our contact plane.
        tri_plane = float4(face_norm, dot(face_norm, closest_point.position));
        smallest_tri_dist = plane_sdf(tri_plane, positions[tid]);
    }


    // compare smallest plane, smallest sphere, and smallest triangle to find the ultimate tinyness (planes win in case of tie)
    // also detect collision
    contact_planes[tid] = float4(0.0);
    if(smallest_sphere_dist < smallest_plane_dist && smallest_sphere_dist < smallest_tri_dist){
        if(smallest_sphere_dist < (contact_margin + collision_distance)){
            float3 n = normalize(positions[tid] - closest_sphere.xyz);
            float d = dot(n, closest_sphere.xyz) + closest_sphere.w;
            contact_planes[tid] = float4(n, d + collision_distance);
        }
    } else if (smallest_plane_dist < smallest_sphere_dist && smallest_plane_dist < smallest_tri_dist){
        if(smallest_plane_dist < (contact_margin + collision_distance)){
            contact_planes[tid] = float4(closest_plane.xyz, closest_plane.w + collision_distance);
        }
    } else if (found_tri){
        if(smallest_tri_dist < (contact_margin + collision_distance)){
            contact_planes[tid] = float4(tri_plane.xyz, tri_plane.w + collision_distance);
        }
    }

    
}
