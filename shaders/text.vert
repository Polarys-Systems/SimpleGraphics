#version 460
#extension GL_EXT_buffer_reference : require

struct GlyphInstance {
    vec2 position;
    vec2 size;
    vec2 uv_min;
    vec2 uv_max;
};

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer InstanceBuffer {
    GlyphInstance values[];
};

layout(push_constant, std430) uniform Push {
    InstanceBuffer instances;
    vec2 viewport_size;
} pc;

layout(location = 0) out vec2 uv;

void main() {
    const vec2 corners[6] = vec2[](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    GlyphInstance glyph = pc.instances.values[gl_InstanceIndex];
    vec2 corner = corners[gl_VertexIndex];
    vec2 pixel = glyph.position + corner * glyph.size;
    vec2 ndc = vec2(
        pixel.x * 2.0 / pc.viewport_size.x - 1.0,
        pixel.y * 2.0 / pc.viewport_size.y - 1.0
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
    uv = mix(glyph.uv_min, glyph.uv_max, corner);
}
