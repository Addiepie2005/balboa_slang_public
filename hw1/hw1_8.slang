import hw1_common;


bool in_circle(float2 xy, 
                float2 center,
                float radius){
    float c = distance(xy, center);
    return c <= radius;
}

bool in_circle_stroke(float2 xy, 
                      float2 center,
                      float radius,
                      float stroke_width){
    float c = distance(xy, center);
    return c >= radius - (stroke_width / 2) && c <= radius + (stroke_width / 2);
}

bool in_polyline(float2 xy, 
                 StructuredBuffer<float2> points,
                 int pts_begin, 
                 int pts_end, 
                 bool closed){

    int num_segments = pts_end - pts_begin;
    int count = 0;
    for(int i = 0; i < num_segments; i++){
        
        float2 p1 = points[pts_begin + i];
        float2 p2 = points[pts_begin + (i + 1) % num_segments];

        bool dir = (p1.y - p2.y) > 0;

        if ((p1.y <= xy.y) == (p2.y <= xy.y)){
            continue;
        } 

        // handle the vertical line case so we never divide by 0
        if(p1.x - p2.x == 0){
            if(xy.x < p1.x){
                if(dir){
                    count++;
                }else{
                    count--;
                }
            } 
            continue;
        }

        float m = (p1.y - p2.y) / (p1.x - p2.x);
        float b = p1.y - (m * p1.x);
        
        if (m == 0){ 
            continue;
        }

        float x_intersect = (xy.y - b)/m;
        if(x_intersect > xy.x){
            if(dir){
                count++;
            }else{
                count--;
            }
        }
    }

    if(count != 0){
        return true;
    }
    return false;
}


bool in_polyline_stroke(float2 xy, 
                        StructuredBuffer<float2> points,
                        int pts_begin, 
                        int pts_end, 
                        bool closed,
                        float stroke_width){
 
    int n = pts_end - pts_begin;
    int num_segments = n;
    if(!closed){
        num_segments = n - 1;
    }

    for (int i = 0; i < num_segments; i++){
        //Compute line equation

        float2 p1 = points[pts_begin + i];
        float2 p2 = points[pts_begin + (i + 1) % n];

        //compute projection + distance from line
        float l = dot((p2 - p1), (xy - p1)) / distance(p1, p2);

        float2 q_p = p1 + (l * normalize(p2 - p1));
        
        float dist = distance(q_p, xy);

        if(l > 0 && l < distance(p1, p2) && dist <= stroke_width / 2){
            return true;
        }
    }

    return false;
}

bool in_bezier(float2 xy, 
                 StructuredBuffer<float2> points,
                 int pts_begin, 
                 int pts_end, 
                 bool closed){

    int num_segments = (pts_end - pts_begin)/2;
    int count = 0;
    for(int i = 0; i < num_segments; i++){

        float2 p0 = points[pts_begin + (2*i)]; 
        float2 p1 = points[pts_begin + (2*i) + 1]; 
        float2 p2 = points[pts_begin + ((2*i) + 2) % (pts_end - pts_begin)];  

        float2 a = p0 - (2 * p1) + p2;
        float2 b = (-2 * p0) + (2 * p1);
        float2 c = p0;

        float2 t_roots;
        if(solve_quadratic(a.y, b.y, c.y - xy.y, t_roots)){
            for(int j = 0; j < t_roots.getCount(); j ++){
                if(t_roots[j] >= 0 && t_roots[j] < 1){
                    //solve for point
                    float x = (a.x * (t_roots[j]*t_roots[j])) + (b.x * t_roots[j]) + c.x;

                    //derivative for direction
                    float dy = (2 * a.y * t_roots[j]) + b.y;
                    
                    if(x > xy.x && dy != 0){
                        if(dy > 0){
                            count++;
                        } else {
                            count--;
                        }
                    }
                }
            }
        }
    }

    if(count != 0){
        return true;
    }
    return false;
}

bool in_bezier_stroke(float2 xy, 
                        StructuredBuffer<float2> points,
                        int pts_begin, 
                        int pts_end, 
                        bool closed,
                        float stroke_width){
 
    int num_segments = (pts_end - pts_begin)/2;

    for (int i = 0; i < num_segments; i++){
        //Compute line equation

        float2 p0 = points[pts_begin + (2*i)]; 
        float2 p1 = points[pts_begin + (2*i) + 1]; 
        float2 p2 = points[pts_begin + ((2*i) + 2) % (pts_end - pts_begin)];  

        float2 a = p0 - (2 * p1) + p2;
        float2 b = (-2 * p0) + (2 * p1);
        float2 c = p0;

        float3 t_roots;
        //Derivative of min function to find closest point on line
        int num_roots = solve_cubic(2*dot(a, a),
                                    3*dot(a, b),
                                    dot(b, b) + 2*dot(a, c-xy),
                                    dot(b, c-xy),
                                    t_roots);

        for(int j = 0; j < num_roots; j ++){
            if(t_roots[j] >= 0 && t_roots[j] <= 1){
                float2 closest_p = (a * (t_roots[j]*t_roots[j])) + (b * t_roots[j]) + c;
                float dist = distance(closest_p, xy);        
                if(dist <= stroke_width / 2){
                    return true;
                }
            }
        }
    }

    return false;
}





[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> quadratic_beziers,
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // TODO: your code here
    image[tid] = float4(background, 1.0);

    float r_offset = (1/((float)samples_per_axis * 2));
    float3 sum_color = float3(0.0, 0.0, 0.0);

    for(int r = 0; r < samples_per_axis; r++){
        float c_offset = (1/((float)samples_per_axis * 2));

        for(int c = 0; c < samples_per_axis; c++){
            float3 color = background;
            float2 canvas_p = float2(tid.x + r_offset, (height - 1 - tid.y) + c_offset);

            for(int i = num_shapes-1; i >= 0; i--){
                float3 object_p = mul(shapes[i].world_to_obj, float3(canvas_p, 1));

                switch(shapes[i].type){
                    case ShapeType.Circle:
                        Circle this_circle = circles[shapes[i].index];
                        if(shapes[i].use_fill){
                            if (in_circle(object_p.xy, this_circle.center, this_circle.radius)){
                                color = (shapes[i].fill_rgba.a * shapes[i].fill_rgba.rgb) + ((1 - shapes[i].fill_rgba.a) * color);
                            }
                        }

                        if(shapes[i].use_stroke){
                            if(in_circle_stroke(object_p.xy, this_circle.center, 
                              this_circle.radius, shapes[i].stroke_width)){
                                color = (shapes[i].stroke_rgba.a * shapes[i].stroke_rgba.rgb) + ((1 - shapes[i].stroke_rgba.a) * color);
                            }                
                        }
                        break;

                    case ShapeType.Polyline:
                        Polyline this_polyline = polylines[shapes[i].index];
                        
                        if(this_polyline.closed && shapes[i].use_fill && in_polyline(object_p.xy, points, 
                            this_polyline.pts_begin, this_polyline.pts_end, this_polyline.closed)){
                            color = (shapes[i].fill_rgba.a * shapes[i].fill_rgba.rgb) + ((1 - shapes[i].fill_rgba.a) * color);
                        }

                        if(shapes[i].use_stroke){
                            if(in_polyline_stroke(object_p.xy, points, this_polyline.pts_begin, this_polyline.pts_end, this_polyline.closed, shapes[i].stroke_width)){
                                color = (shapes[i].stroke_rgba.a * shapes[i].stroke_rgba.rgb) + ((1 - shapes[i].stroke_rgba.a) * color);
                            } else {
                                int first = this_polyline.closed ? this_polyline.pts_begin : this_polyline.pts_begin + 1;
                                int last  = this_polyline.closed ? this_polyline.pts_end   : this_polyline.pts_end - 1;
                                for (int j = first; j < last; j++){
                                    if(in_circle(object_p.xy, points[j], shapes[i].stroke_width/2)){
                                        color = (shapes[i].stroke_rgba.a * shapes[i].stroke_rgba.rgb) + ((1 - shapes[i].stroke_rgba.a) * color);
                                        break;
                                    }
                                }
                            }
                        }
                        break;

                    case ShapeType.QuadraticBezier:
                        QuadraticBezier this_quad = quadratic_beziers[shapes[i].index];
                        
                        if(this_quad.closed && shapes[i].use_fill && in_bezier(object_p.xy, points, 
                            this_quad.pts_begin, this_quad.pts_end, this_quad.closed)){
                            color = (shapes[i].fill_rgba.a * shapes[i].fill_rgba.rgb) + ((1 - shapes[i].fill_rgba.a) * color);
                        }

                        if(shapes[i].use_stroke){
                            if(in_bezier_stroke(object_p.xy, points, this_quad.pts_begin, this_quad.pts_end, this_quad.closed, shapes[i].stroke_width)){
                                color = (shapes[i].stroke_rgba.a * shapes[i].stroke_rgba.rgb) + ((1 - shapes[i].stroke_rgba.a) * color);
                            } else {
                                int first = this_quad.closed ? this_quad.pts_begin : this_quad.pts_begin + 2;
                                int last  = this_quad.closed ? this_quad.pts_end   : this_quad.pts_end - 2;
                                for (int j = first; j < last; j+=2){
                                    if(in_circle(object_p.xy, points[j], shapes[i].stroke_width/2)){
                                        color = (shapes[i].stroke_rgba.a * shapes[i].stroke_rgba.rgb) + ((1 - shapes[i].stroke_rgba.a) * color);
                                        break;
                                    }
                                }
                            }
                        }

                    break;
                }

            } 
            sum_color += color;
            c_offset += (1/(float)samples_per_axis);
        }
        r_offset += (1/(float)samples_per_axis);
    }

   image[tid] = float4(sum_color/((float)samples_per_axis * samples_per_axis), 1.0);
   
    
}
