[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }
    // TODO: HW 4.4. Your code below.

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

    
    if(count == 0){
        positions_out[tid] = positions_in[tid];
        return;
    }
    positions_out[tid] = positions_in[tid] + (corr_vec / count);

}
