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

    image[tid] = float4(1.0);
    // TODO: your code here

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);
    
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
}
