import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    // TODO: your code here

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    //Check if line y = height - 1 - tid.y intersects with any of our polylines (and x > tid.x)
    if(is_closed && use_fill_color){
        int count = 0;
        for(int i = 0; i < num_polyline; i++){
            float2 p1 = polyline[i];
            float2 p2 = polyline[(i + 1) % num_polyline]; //CHANGE HERE: simpler wrap around logic

            bool dir = (p1.y - p2.y) > 0;

            //CHANGE HERE: <= which somehow fixed the stray line
            if ((p1.y <= height - 1 - tid.y) == (p2.y <= height - 1 - tid.y)){
                continue;
            } 

            if(p1.x - p2.x == 0){
                if(tid.x == p1.x) break; 
                if(tid.x < p1.x){
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

            float x_intersect = ((height - 1 - tid.y) - b)/m;
            if(x_intersect > tid.x){
                if(dir){
                    count++;
                }else{
                    count--;
                }
            }
        }

        if(count != 0){
            image[tid] = float4(fill_color, 1.0);
        }
    }


    for (int i = 0; i < num_polyline - 1; i++){
        //Compute line equation
        float2 p1 = polyline[i];
        float2 p2 = polyline[i+1];

        float2 translatedPoint = float2(tid.x, height - 1 - tid.y);

        //compute projection + distance from line
        float l = dot((p2 - p1), (translatedPoint - p1))/distance(p1, p2);

        float2 q_p = p1 + (l * normalize(p2 - p1));
        
        float dist = distance(q_p, translatedPoint);

        if (l > 0 && l < distance(p1, p2) && dist < stroke_width / 2){
            if(use_stroke_color){
                image[tid] = float4(stroke_color, 1.0);
            }
        }
    }

    if(is_closed){
        //Compute line equation
        float2 p1 = polyline[num_polyline - 1];
        float2 p2 = polyline[0];

        float2 translatedPoint = float2(tid.x, height - 1 - tid.y);

        //compute projection + distance from line
        float l = dot((p2 - p1), (translatedPoint - p1))/distance(p1, p2);

        float2 q_p = p1 + (l * normalize(p2 - p1));
        
        float dist = distance(q_p, translatedPoint);

        if (l > 0 && l < distance(p1, p2) && dist < stroke_width / 2){
            if(use_stroke_color){
                image[tid] = float4(stroke_color, 1.0);
            }
        }
    }

    int start = 1;
    int numPoints = num_polyline - 1;
    if(is_closed){
        start = 0;
        numPoints++;
    }

    if(use_stroke_color){ //CHANGE HERE: added if statement to fix black corners 
        //(didn't know what color to use so it zeroed out to black and made a tiny black circle)
        for (int i = start; i < numPoints; i++){
            float c = distance(polyline[i] + 0.5, float2(tid.x, height - 1 - tid.y) + 0.5);
            if (c <= stroke_width/2) {
                image[tid] = float4(stroke_color, 1.0);
            }
        }
    }
 
}
