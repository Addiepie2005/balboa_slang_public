[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }    
    
    // TODO: HW 4.3. Your code below.

    // added infinite mass check, we should not move
    if(inv_masses[tid] == 0){
        positions_out[tid] = positions_in[tid];
        return;
    }

    int2 curr = int2(tid % grid_size.x, tid / grid_size.x);




    //structural constraint --------------------------------------------------------------------
    float w = inv_masses[tid];
    float3 dir;
    float dist;
    float l;
    float3 corr_vec = float3(0.0);
    int count = 0;

    //up neighbor
    if(curr.y < grid_size.y - 1){
        count ++;
        int2 up = int2(curr.x, curr.y + 1);
        uint up_tid = (up.y * grid_size.x) + up.x;
        float w_up = inv_masses[up_tid];
        dir = positions_in[up_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.y;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_up)) * (dist - l) * (dir / dist);
        }
    }

    //down neighbor
    if(curr.y > 0){
        count ++;
        int2 down = int2(curr.x, curr.y - 1);
        uint down_tid = (down.y * grid_size.x) + down.x;
        float w_down = inv_masses[down_tid];
        dir = positions_in[down_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.y;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_down)) * (dist - l) * (dir / dist);
        }
    }

    //left neighbor
    if(curr.x > 0){
        count ++;
        int2 left = int2(curr.x - 1, curr.y);
        uint left_tid = (left.y * grid_size.x) + left.x;
        float w_left = inv_masses[left_tid];
        dir = positions_in[left_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.x;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_left)) * (dist - l) * (dir / dist);
        }
    }

    //right neighbor
    if(curr.x < grid_size.x - 1){
        count ++;
        int2 right = int2(curr.x + 1, curr.y);
        uint right_tid = (right.y * grid_size.x) + right.x;
        float w_right = inv_masses[right_tid];
        dir = positions_in[right_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.x;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_right)) * (dist - l) * (dir / dist);
        }
    }






    //shearing constraint --------------------------------------------------------------------
    l = sqrt((spacing.y * spacing.y) + (spacing.x * spacing.x));

    //upper left neighbor
    if(curr.y < grid_size.y - 1 && curr.x > 0){
        count ++;
        int2 up_left = int2(curr.x - 1, curr.y + 1);
        uint up_left_tid = (up_left.y * grid_size.x) + up_left.x;
        float w_up_left = inv_masses[up_left_tid];
        dir = positions_in[up_left_tid] - positions_in[tid];
        dist = length(dir);
        
        if(dist > 1e-8){
            corr_vec += (w / (w + w_up_left)) * (dist - l) * (dir / dist);
        }
    }

    //upper right neighbor
    if(curr.y < grid_size.y - 1 && curr.x < grid_size.x - 1){
        count ++;
        int2 up_right = int2(curr.x + 1, curr.y + 1);
        uint up_right_tid = (up_right.y * grid_size.x) + up_right.x;
        float w_up_right = inv_masses[up_right_tid];
        dir = positions_in[up_right_tid] - positions_in[tid];
        dist = length(dir);
        
        if(dist > 1e-8){
            corr_vec += (w / (w + w_up_right)) * (dist - l) * (dir / dist);
        }
    }

    //lower right neighbor
    if(curr.y > 0 && curr.x < grid_size.x - 1){
        count ++;
        int2 down_right = int2(curr.x + 1, curr.y - 1);
        uint down_right_tid = (down_right.y * grid_size.x) + down_right.x;
        float w_down_right = inv_masses[down_right_tid];
        dir = positions_in[down_right_tid] - positions_in[tid];
        dist = length(dir);
        
        if(dist > 1e-8){
            corr_vec += (w / (w + w_down_right)) * (dist - l) * (dir / dist);
        }
    }

    //lower left neighbor
    if(curr.y > 0 && curr.x > 0){
        count ++;
        int2 down_left = int2(curr.x - 1, curr.y - 1);
        uint down_left_tid = (down_left.y * grid_size.x) + down_left.x;
        float w_down_left = inv_masses[down_left_tid];
        dir = positions_in[down_left_tid] - positions_in[tid];
        dist = length(dir);
        
        if(dist > 1e-8){
            corr_vec += (w / (w + w_down_left)) * (dist - l) * (dir / dist);
        }
    }





    //bending constraint --------------------------------------------------------------------
    
    //double up neighbor
    if(curr.y < grid_size.y - 2){
        count ++;
        int2 up = int2(curr.x, curr.y + 2);
        uint up_tid = (up.y * grid_size.x) + up.x;
        float w_up = inv_masses[up_tid];
        dir = positions_in[up_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.y * 2;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_up)) * (dist - l) * (dir / dist);
        }
    }

    //double down neighbor
    if(curr.y > 1){
        count ++;
        int2 down = int2(curr.x, curr.y - 2);
        uint down_tid = (down.y * grid_size.x) + down.x;
        float w_down = inv_masses[down_tid];
        dir = positions_in[down_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.y * 2;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_down)) * (dist - l) * (dir / dist);
        }
    }

    //double left neighbor
    if(curr.x > 1){
        count ++;
        int2 left = int2(curr.x - 2, curr.y);
        uint left_tid = (left.y * grid_size.x) + left.x;
        float w_left = inv_masses[left_tid];
        dir = positions_in[left_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.x * 2;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_left)) * (dist - l) * (dir / dist);
        }
    }

    //double right neighbor
    if(curr.x < grid_size.x - 2){
        count ++;
        int2 right = int2(curr.x + 2, curr.y);
        uint right_tid = (right.y * grid_size.x) + right.x;
        float w_right = inv_masses[right_tid];
        dir = positions_in[right_tid] - positions_in[tid];
        dist = length(dir);
        l = spacing.x * 2;

        if(dist > 1e-8){
            corr_vec += (w / (w + w_right)) * (dist - l) * (dir / dist);
        }
    }
    
    if(count == 0){
        positions_out[tid] = positions_in[tid];
        return;
    }
    positions_out[tid] = positions_in[tid] + (corr_vec / count);

}
