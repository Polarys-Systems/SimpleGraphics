#version 460

layout(set = 0, binding = 0) uniform texture2D sampled_images[];
layout(set = 2, binding = 0) uniform sampler samplers[];

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main() {
    float coverage = texture(sampler2D(sampled_images[0], samplers[0]), uv).r;
    color = vec4(0.96, 0.93, 0.82, coverage);
}
