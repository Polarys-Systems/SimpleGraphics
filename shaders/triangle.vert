#version 460
#extension GL_EXT_buffer_reference : require

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer PointBuffer {
    vec2 values[];
};

layout(push_constant, std430) uniform Push {
    PointBuffer points;
    vec2 center;
    vec2 scale;
    vec2 marker_size;
} pc;

void main() {
    const vec2 corners[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2(1.0, -1.0),
        vec2(0.0, 1.0)
    );

    uint point_index = uint(gl_VertexIndex) / 3u;
    uint corner_index = uint(gl_VertexIndex) % 3u;
    vec2 position = (pc.points.values[point_index] - pc.center) * pc.scale;
    gl_Position = vec4(position + corners[corner_index] * pc.marker_size, 0.0, 1.0);
}
