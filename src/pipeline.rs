use crate::{device::DeviceState, *};
use std::rc::Rc;

/// Immutable pipeline using the device's common descriptor/push-constant layout.
pub struct PSO {
    pub(crate) state: Rc<DeviceState>,
    pub(crate) raw: vk::Pipeline,
    pub(crate) bind_point: vk::PipelineBindPoint,
    pub(crate) mesh: bool,
}

/// Triangle-list pipeline with no vertex bindings: fetch vertices by device address.
/// Viewport/scissor are dynamic, samples = 1, SPIR-V entry point = `main`.
/// # Safety
/// SPIR-V must be valid for the enabled device features and documented shader ABI.
pub unsafe fn create_graphics_pso(
    device: &GpuDevice,
    desc: &GraphicsPSODesc<'_>,
) -> Result<PSO, GpuError> {
    create_raster_pipeline(
        device,
        desc.vertex_spirv,
        desc.fragment_spirv,
        desc.color_targets,
        [desc.depth_format, desc.stencil_format],
        &desc.rasterization,
        &desc.depth_stencil,
        false,
    )
}

/// Mesh+fragment pipeline. Task shaders are not part of this API's shader ABI.
/// # Safety
/// SPIR-V must be valid for the enabled device features and documented shader ABI.
pub unsafe fn create_mesh_pso(device: &GpuDevice, desc: &MeshPSODesc<'_>) -> Result<PSO, GpuError> {
    if !device.state.caps.mesh_shader {
        return Err(GpuError::Unsupported(format!(
            "{}: VK_EXT_mesh_shader / meshShader",
            device.state.caps.device_name.to_string_lossy()
        )));
    }
    create_raster_pipeline(
        device,
        desc.mesh_spirv,
        desc.fragment_spirv,
        desc.color_targets,
        [desc.depth_format, desc.stencil_format],
        &desc.rasterization,
        &desc.depth_stencil,
        true,
    )
}

/// Compute pipeline with entry point `main` and the common descriptor layout.
/// # Safety
/// SPIR-V must be valid for the enabled device features and documented shader ABI.
pub unsafe fn create_compute_pso(device: &GpuDevice, spirv: &[u32]) -> Result<PSO, GpuError> {
    validate_spirv(spirv)?;
    unsafe {
        let raw = device.state.raw();
        let shader =
            raw.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main");
        let result = raw.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
                .stage(stage)
                .layout(device.state.pipeline_layout)],
            None,
        );
        raw.destroy_shader_module(shader, None);
        finish_pipeline(&device.state, result, vk::PipelineBindPoint::COMPUTE, false)
    }
}

// Graphics and mesh pipelines differ only in the first shader and vertex input.
#[allow(clippy::too_many_arguments)]
fn create_raster_pipeline(
    device: &GpuDevice,
    first: &[u32],
    fragment: &[u32],
    targets: &[ColorTargetDesc],
    formats: [Format; 2],
    raster: &RasterizationState,
    depth: &DepthStencilState,
    mesh: bool,
) -> Result<PSO, GpuError> {
    validate_spirv(first)?;
    if !fragment.is_empty() {
        validate_spirv(fragment)?;
    }
    if targets.len() > device.state.caps.max_color_attachments as usize {
        return Err(GpuError::Unsupported("too many color attachments".into()));
    }
    if raster.depth_bias_clamp != 0.0 {
        return Err(GpuError::Unsupported(
            "depthBiasClamp is not enabled".into(),
        ));
    }
    if depth.depth_test && formats[0] == Format::Undefined
        || depth.stencil_test && formats[1] == Format::Undefined
    {
        return Err(GpuError::InvalidArgument(
            "depth/stencil test requires a corresponding attachment format",
        ));
    }
    let mut color_formats = Vec::new();
    let mut blends = Vec::new();
    for target in targets {
        if !supports_texture_format(device, target.format, TextureUsage::COLOR_ATTACHMENT) {
            return Err(GpuError::Unsupported(format!(
                "color attachment format {:?}",
                target.format
            )));
        }
        if target.write_mask & !0xf != 0 {
            return Err(GpuError::InvalidArgument(
                "color write_mask has bits outside RGBA",
            ));
        }
        if target.blend.enabled {
            let properties = unsafe {
                device.state.instance.get_physical_device_format_properties(
                    device.state.physical,
                    target.format.into(),
                )
            };
            if !properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
            {
                return Err(GpuError::Unsupported(format!(
                    "blending for {:?}",
                    target.format
                )));
            }
        }
        color_formats.push(target.format.into());
        blends.push(
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(target.blend.enabled)
                .src_color_blend_factor(blend_factor(target.blend.color.source))
                .dst_color_blend_factor(blend_factor(target.blend.color.destination))
                .color_blend_op(vk::BlendOp::from_raw(target.blend.color.operation as i32))
                .src_alpha_blend_factor(blend_factor(target.blend.alpha.source))
                .dst_alpha_blend_factor(blend_factor(target.blend.alpha.destination))
                .alpha_blend_op(vk::BlendOp::from_raw(target.blend.alpha.operation as i32))
                .color_write_mask(vk::ColorComponentFlags::from_raw(u32::from(
                    target.write_mask,
                ))),
        );
    }
    for (index, format) in formats.into_iter().enumerate() {
        if format == Format::Undefined {
            continue;
        }
        let info = TextureFormatInfo::get_texture_format_info(format);
        if (index == 0 && !info.depth) || (index == 1 && !info.stencil) {
            return Err(GpuError::InvalidArgument("invalid depth/stencil format"));
        }
        if !supports_texture_format(device, format, TextureUsage::DEPTH_STENCIL_ATTACHMENT) {
            return Err(GpuError::Unsupported(format!(
                "depth/stencil format {format:?}"
            )));
        }
    }
    if formats[0] != Format::Undefined
        && formats[1] != Format::Undefined
        && formats[0] != formats[1]
    {
        return Err(GpuError::InvalidArgument(
            "depth and stencil must use the same format when both are present",
        ));
    }
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let (cull, front) = match raster.cull {
        CullMode::None => (vk::CullModeFlags::NONE, vk::FrontFace::COUNTER_CLOCKWISE),
        CullMode::Clockwise => (vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE),
        CullMode::CounterClockwise => (vk::CullModeFlags::BACK, vk::FrontFace::CLOCKWISE),
    };
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(cull)
        .front_face(front)
        .line_width(1.0)
        .depth_bias_enable(raster.depth_bias_constant != 0.0 || raster.depth_bias_slope != 0.0)
        .depth_bias_constant_factor(raster.depth_bias_constant)
        .depth_bias_slope_factor(raster.depth_bias_slope);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let mut faces = [vk::StencilOpState::default(); 2];
    for (index, face) in [&depth.front, &depth.back].into_iter().enumerate() {
        faces[index] = vk::StencilOpState::default()
            .fail_op(vk::StencilOp::from_raw(face.fail as i32))
            .pass_op(vk::StencilOp::from_raw(face.pass as i32))
            .depth_fail_op(vk::StencilOp::from_raw(face.depth_fail as i32))
            .compare_op(vk::CompareOp::from_raw(face.compare as i32))
            .compare_mask(u32::from(depth.stencil_read_mask))
            .write_mask(u32::from(depth.stencil_write_mask))
            .reference(u32::from(face.reference));
    }
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth.depth_test)
        .depth_write_enable(depth.depth_write)
        .depth_compare_op(vk::CompareOp::from_raw(depth.depth_compare as i32))
        .stencil_test_enable(depth.stencil_test)
        .front(faces[0])
        .back(faces[1]);
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(formats[0].into())
        .stencil_attachment_format(formats[1].into());
    unsafe {
        let raw = device.state.raw();
        let first_module =
            raw.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(first), None)?;
        let mut fragment_module = vk::ShaderModule::null();
        if !fragment.is_empty() {
            match raw
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(fragment), None)
            {
                Ok(module) => fragment_module = module,
                Err(error) => {
                    raw.destroy_shader_module(first_module, None);
                    return Err(error.into());
                }
            }
        }
        let mut stages = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(if mesh {
                    vk::ShaderStageFlags::MESH_EXT
                } else {
                    vk::ShaderStageFlags::VERTEX
                })
                .module(first_module)
                .name(c"main"),
        ];
        if fragment_module != vk::ShaderModule::null() {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_module)
                    .name(c"main"),
            );
        }
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stages(&stages)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(device.state.pipeline_layout)
            .push_next(&mut rendering);
        if !mesh {
            info = info
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&assembly);
        }
        let result = raw.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None);
        raw.destroy_shader_module(fragment_module, None);
        raw.destroy_shader_module(first_module, None);
        finish_pipeline(&device.state, result, vk::PipelineBindPoint::GRAPHICS, mesh)
    }
}

fn finish_pipeline(
    state: &Rc<DeviceState>,
    result: Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
    bind_point: vk::PipelineBindPoint,
    mesh: bool,
) -> Result<PSO, GpuError> {
    match result {
        Ok(pipelines) => Ok(PSO {
            state: state.clone(),
            raw: pipelines[0],
            bind_point,
            mesh,
        }),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                unsafe {
                    state.raw().destroy_pipeline(pipeline, None);
                }
            }
            Err(error.into())
        }
    }
}

fn validate_spirv(words: &[u32]) -> Result<(), GpuError> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(GpuError::InvalidArgument(
            "expected little-endian SPIR-V words with a valid header",
        ));
    }
    Ok(())
}

fn blend_factor(factor: BlendFactor) -> vk::BlendFactor {
    match factor {
        BlendFactor::Zero => vk::BlendFactor::ZERO,
        BlendFactor::One => vk::BlendFactor::ONE,
        BlendFactor::SourceColor => vk::BlendFactor::SRC_COLOR,
        BlendFactor::OneMinusSourceColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        BlendFactor::DestinationColor => vk::BlendFactor::DST_COLOR,
        BlendFactor::OneMinusDestinationColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
        BlendFactor::SourceAlpha => vk::BlendFactor::SRC_ALPHA,
        BlendFactor::OneMinusSourceAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        BlendFactor::DestinationAlpha => vk::BlendFactor::DST_ALPHA,
        BlendFactor::OneMinusDestinationAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        BlendFactor::SourceAlphaSaturate => vk::BlendFactor::SRC_ALPHA_SATURATE,
    }
}

pub fn destroy_pso(pso: PSO) {
    drop(pso);
}
impl Drop for PSO {
    fn drop(&mut self) {
        unsafe {
            self.state.raw().destroy_pipeline(self.raw, None);
        }
    }
}
