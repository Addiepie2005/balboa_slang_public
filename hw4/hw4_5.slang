float sphere_sdf(float3 center, float r, float3 p) { 
    return length(p - center) - r;
}

float plane_sdf(float4 plane, float3 p){
    return dot(plane.xyz, p) - plane.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated 
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below 
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.5. Your code below.

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

    // compare smallest plane and smallest sphere to find the ultimate tinyness (planes win in case of tie)
    // also detect collision
    contact_planes[tid] = float4(0.0);
    if(smallest_sphere_dist < smallest_plane_dist){
        if(smallest_sphere_dist < (contact_margin + collision_distance)){
            float3 n = normalize(positions[tid] - closest_sphere.xyz);
            float d = dot(n, closest_sphere.xyz) + closest_sphere.w;
            contact_planes[tid] = float4(n, d + collision_distance);
        }
    } else {
        if(smallest_plane_dist < (contact_margin + collision_distance)){
            contact_planes[tid] = float4(closest_plane.xyz, closest_plane.w + collision_distance);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         // Contact plane: dot(n, x) = d, with unit normal n
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.5. Your code below.

    // added infinite mass check, we should not move
    if(inv_masses[tid] == 0){
        positions_out[tid] = positions_in[tid];
        return;
    }

    // grab the start and end points of the neighbors for this point
    uint neighbor_start = row_ptr[tid];
    uint neighbor_end = row_ptr[tid + 1];

    float w = inv_masses[tid];
    float3 dir;
    float dist;
    float l;
    float3 corr_vec = float3(0.0);
    int count = 0;
    for(uint i = neighbor_start; i < neighbor_end; i++){
        // grab the neighbor node that we are working with
        uint neighbor_tid = neighbors[i];

        count++;
        float w_neighbor = inv_masses[neighbor_tid];
        dir = positions_in[neighbor_tid] - positions_in[tid];
        dist = length(dir);
        // assume rest length is indexed the same way neighbors is as it is per edge
        l = neighbor_rest[i];

        if(dist > 1e-8){
            corr_vec += (w / (w + w_neighbor)) * (dist - l) * (dir / dist);
        }

    }

    float3 x;
    if(count == 0){
        x = positions_in[tid];
    } else {
        x = positions_in[tid] + (corr_vec / count);
    }


    // check if there is a contact
    if(all(contact_planes[tid] == float4(0.0))){
        positions_out[tid] = x;
        return;
    }

    // normal direction correction
    float3 n = contact_planes[tid].xyz;

    float pen_depth = -min(plane_sdf(contact_planes[tid], x), 0.0);

    positions_out[tid] = x + (normalize(n) * pen_depth);

    // tangential direction correction
    float3 tangent_dir = x - old_positions[tid] - (dot(x - old_positions[tid], n) * n);
    float tangent_len = length(tangent_dir);

    if(tangent_len <= static_friction * pen_depth){
        // add static contribution
        positions_out[tid] -= tangent_dir;
    } else {
        // add kinetic contribution
        positions_out[tid] -= (kinetic_friction * pen_depth) * normalize(tangent_dir);
    }

}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.5. Your code below.
    
    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;
}
