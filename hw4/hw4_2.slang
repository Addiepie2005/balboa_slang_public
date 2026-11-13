[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {

    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    // added infinite mass check, we should not move
    if(inv_masses[tid] == 0){
        old_positions[tid] = positions[tid];
        return;
    }

    //Applies gravity (0, −9.8, 0) to the velocity and update the position 

    old_positions[tid] = positions[tid];

    velocities[tid].y = velocities[tid].y + (dt * -9.8);

    positions[tid] = positions[tid] + (dt * velocities[tid]);

}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
                         uniform uint num_particles) {

    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    // added infinite mass check, we should not move
    if(inv_masses[tid] == 0){
        positions_out[tid] = positions_in[tid];
        return;
    }
    
    //The project kernel does the actual Jacobi iteration

    //for each particle, we look at its neighbors and compute the “correction” for each edge using the distance constraints

    // neighbor 1 to us, w is the inverse mass
    float w = inv_masses[tid];
    float3 dir;
    float dist;
    float l;
    float3 corr_vec = float3(0.0);
    int count = 0;

    if(tid > 0){
        count++;
        float w_bef = inv_masses[tid - 1];
        dir = positions_in[tid - 1] - positions_in[tid];
        dist = length(dir);
        l = rest_lengths[tid - 1];

        if(dist > 1e-8){
            corr_vec += (w / (w + w_bef)) * (dist - l) * (dir / dist);
        }
    }

    // us to neighbor 2
    if(tid < num_particles - 1){
        count++;
        float w_aft = inv_masses[tid + 1];
        dir = positions_in[tid + 1] - positions_in[tid];
        dist = length(dir);
        l = rest_lengths[tid];
        
        if(dist > 1e-8){
            corr_vec += (w / (w + w_aft)) * (dist - l) * (dir / dist);
        }
    }

    if(count == 0){
        positions_out[tid] = positions_in[tid];
        return;
    }
    positions_out[tid] = positions_in[tid] + (corr_vec / count);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     // time step size
                     uniform float dt) {
    // added infinite mass check, we should not move
    if (tid >= num_particles || inv_masses[tid] == 0) {
        return;
    }
    // TODO: HW 4.2. Your code below.

    if(positions[tid].y <= -0.97){
        positions[tid].y = -0.97;
    }

    velocities[tid] = (positions[tid] - old_positions[tid]) / dt;

}
