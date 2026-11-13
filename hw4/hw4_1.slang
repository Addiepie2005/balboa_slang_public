[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.1. Your code below.

    //Find v_n+1
    velocities[tid].y = velocities[tid].y + (dt * -9.8);

    //and update the position
    positions[tid] = positions[tid] + (dt * velocities[tid]);
    
    //reverse velocity and clip if needed
    if(positions[tid].y <= -0.97){
        positions[tid].y = -0.97;
        velocities[tid].y = -velocities[tid].y;
    }
}

