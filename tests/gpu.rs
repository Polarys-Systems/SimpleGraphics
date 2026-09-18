//! Opt-in real-driver tests: cargo test --test gpu -- --ignored --nocapture
//! Requires Vulkan 1.4, descriptor buffers, validation layers, and glslc in PATH.
use no_graphics_api_rs::*;
use std::{
    io::{Cursor, Write},
    process::{Command, Stdio},
};

fn compile_shader(source: &str, stage: &str) -> Vec<u32> {
    let mut child = Command::new("glslc")
        .arg(format!("-fshader-stage={stage}"))
        .args(["--target-env=vulkan1.3", "-o", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("glslc must be installed");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    ash::util::read_spv(&mut Cursor::new(output.stdout)).unwrap()
}

#[test]
#[ignore = "requires a Vulkan 1.4 GPU with descriptor buffers, validation layers and glslc"]
fn headless_copy_compute_descriptors_and_graphics() -> Result<(), Box<dyn std::error::Error>> {
    let mut device = create_device(&DeviceDesc {
        validation: true,
        descriptor_count: 32,
        ..DeviceDesc::default()
    })?;
    println!(
        "Testing {}",
        get_device_caps(&device).device_name.to_string_lossy()
    );
    let mut upload = create_gpu_heap(&device, 16, MemoryType::CpuVisible)?;
    let mut readback = create_gpu_heap(&device, 32, MemoryType::Readback)?;
    let output = create_gpu_heap(&device, 16, MemoryType::GpuOnly)?;
    let pixels = [
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
    ];
    unsafe {
        write_memory(&mut upload, 0, &pixels)?;
    }
    let sampled_texture = create_texture(
        &device,
        &TextureDesc {
            extent: U32x3 { x: 2, y: 2, z: 1 },
            usage: TextureUsage::SAMPLED | TextureUsage::TRANSFER_DESTINATION,
            ..TextureDesc::default()
        },
    )?;
    let storage_texture = create_texture(
        &device,
        &TextureDesc {
            extent: U32x3 { x: 2, y: 2, z: 1 },
            usage: TextureUsage::STORAGE | TextureUsage::TRANSFER_SOURCE,
            ..TextureDesc::default()
        },
    )?;
    let sampled_view = create_texture_view(&sampled_texture, &TextureDescriptorDesc::default())?;
    let storage_view = create_texture_view(&storage_texture, &TextureDescriptorDesc::default())?;
    let sampler = create_sampler(
        &device,
        &SamplerDesc {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            ..SamplerDesc::default()
        },
    )?;
    let mut sampled_heap = create_texture_descriptor_heap(&device, TextureDescriptorType::Sampled)?;
    let mut storage_heap = create_texture_descriptor_heap(&device, TextureDescriptorType::Storage)?;
    let mut sampler_heap = create_sampler_descriptor_heap(&device)?;
    unsafe {
        write_texture_descriptor(&mut sampled_heap, 7, &sampled_view)?;
        write_texture_descriptor(&mut storage_heap, 11, &storage_view)?;
        write_sampler_descriptor(&mut sampler_heap, 3, &sampler)?;
    }
    let shader = compile_shader(
        r#"
        #version 460
        #extension GL_EXT_buffer_reference : require
        #extension GL_EXT_nonuniform_qualifier : require
        layout(local_size_x = 1) in;
        layout(buffer_reference, std430, buffer_reference_align = 4) buffer Output { uint values[]; };
        layout(push_constant, std430) uniform Push { Output output_buffer; uint sampled_index; uint storage_index; uint sampler_index; } pc;
        layout(set = 0, binding = 0) uniform texture2D sampled_images[];
        layout(set = 1, binding = 0, rgba8) uniform image2D storage_images[];
        layout(set = 2, binding = 0) uniform sampler samplers[];
        void main() {
            vec4 color = texelFetch(sampler2D(sampled_images[nonuniformEXT(pc.sampled_index)], samplers[nonuniformEXT(pc.sampler_index)]), ivec2(0), 0);
            imageStore(storage_images[nonuniformEXT(pc.storage_index)], ivec2(0), color);
            pc.output_buffer.values[0] = 0x12345678u;
        }
    "#,
        "compute",
    );
    let pso = unsafe { create_compute_pso(&device, &shader)? };
    let timeline = create_timeline_semaphore(&device, 0)?;
    assert_eq!(timeline_completed_value(&timeline)?, 0);
    assert_eq!(wait_timeline(&timeline, 1, 0), Err(GpuError::Timeout));
    let mut command = begin_command(&device)?;
    unsafe {
        barrier(
            &mut command,
            &[
                Barrier::Texture {
                    texture: &sampled_texture,
                    before: (Stage::NONE, Access::NONE, TextureLayout::Undefined),
                    after: (
                        Stage::TRANSFER,
                        Access::TRANSFER_WRITE,
                        TextureLayout::TransferDestination,
                    ),
                },
                Barrier::Texture {
                    texture: &storage_texture,
                    before: (Stage::NONE, Access::NONE, TextureLayout::Undefined),
                    after: (Stage::COMPUTE, Access::SHADER_WRITE, TextureLayout::General),
                },
            ],
        )?;
        copy_memory_to_texture(
            &mut command,
            upload.range(0, 16)?,
            &sampled_texture,
            &TextureCopyDesc::default(),
            TextureAspect::Automatic,
        )?;
        barrier(
            &mut command,
            &[
                Barrier::Texture {
                    texture: &sampled_texture,
                    before: (
                        Stage::TRANSFER,
                        Access::TRANSFER_WRITE,
                        TextureLayout::TransferDestination,
                    ),
                    after: (Stage::COMPUTE, Access::SHADER_READ, TextureLayout::Sampled),
                },
                Barrier::Memory {
                    before: (Stage::HOST, Access::HOST_WRITE),
                    after: (Stage::COMPUTE, Access::DESCRIPTOR_READ),
                },
            ],
        )?;
        bind_pso(&mut command, &pso)?;
        bind_descriptor_heaps(&mut command, [&sampled_heap, &storage_heap, &sampler_heap])?;
        let mut push = Vec::new();
        push.extend_from_slice(&output.device_address().to_ne_bytes());
        for index in [7_u32, 11, 3] {
            push.extend_from_slice(&index.to_ne_bytes());
        }
        push_constants(&mut command, 0, &push)?;
        write_timestamp(&mut command, 0, Stage::COMPUTE)?;
        dispatch(&mut command, U32x3 { x: 1, y: 1, z: 1 })?;
        barrier(
            &mut command,
            &[
                Barrier::Texture {
                    texture: &storage_texture,
                    before: (Stage::COMPUTE, Access::SHADER_WRITE, TextureLayout::General),
                    after: (
                        Stage::TRANSFER,
                        Access::TRANSFER_READ,
                        TextureLayout::TransferSource,
                    ),
                },
                Barrier::Buffer {
                    range: output.range(0, 4)?,
                    before: (Stage::COMPUTE, Access::SHADER_WRITE),
                    after: (Stage::TRANSFER, Access::TRANSFER_READ),
                },
            ],
        )?;
        copy_texture_to_memory(
            &mut command,
            &storage_texture,
            readback.range(0, 4)?,
            &TextureCopyDesc {
                extent: U32x3 { x: 1, y: 1, z: 1 },
                ..TextureCopyDesc::default()
            },
            TextureAspect::Automatic,
        )?;
        copy_memory(&mut command, output.range(0, 4)?, readback.range(16, 4)?)?;
        barrier(
            &mut command,
            &[Barrier::Buffer {
                range: readback.range(0, 32)?,
                before: (Stage::TRANSFER, Access::TRANSFER_WRITE),
                after: (Stage::HOST, Access::HOST_READ),
            }],
        )?;
        write_timestamp(&mut command, 1, Stage::ALL_COMMANDS)?;
        submit(
            &mut device,
            &mut command,
            &[],
            &[(&timeline, 1, Stage::ALL_COMMANDS)],
        )?;
    }
    wait_timeline(&timeline, 1, u64::MAX)?;
    wait_command(&command, u64::MAX)?;
    assert!(timeline_completed_value(&timeline)? >= 1);
    let mut timestamps = [0; 2];
    let mask = read_timestamps(&command, 0, &mut timestamps)?;
    println!(
        "Compute + readback: {} ticks",
        timestamps[1].wrapping_sub(timestamps[0]) & mask
    );
    unsafe {
        invalidate_memory(readback.range(0, 32)?)?;
        let mapped = mapped_memory(&mut readback, 0, 32)?;
        assert_eq!(&mapped.cpu[0..4], &pixels[0..4]);
        assert_eq!(&mapped.cpu[16..20], &0x12345678_u32.to_ne_bytes());
    }

    // Render a full-screen triangle and verify the resulting color through a copy.
    let vertex = compile_shader(
        r#"
        #version 460
        void main() {
            vec2 p[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
            gl_Position = vec4(p[gl_VertexIndex], 0, 1);
        }
    "#,
        "vertex",
    );
    let fragment = compile_shader(
        r#"
        #version 460
        layout(location = 0) out vec4 color;
        void main() { color = vec4(0,1,0,1); }
    "#,
        "fragment",
    );
    let target = create_texture(
        &device,
        &TextureDesc {
            extent: U32x3 { x: 2, y: 2, z: 1 },
            usage: TextureUsage::COLOR_ATTACHMENT | TextureUsage::TRANSFER_SOURCE,
            ..TextureDesc::default()
        },
    )?;
    let target_view = create_render_view(&target, &RenderViewDesc::default())?;
    let targets = [ColorTargetDesc {
        format: Format::Rgba8Unorm,
        ..ColorTargetDesc::default()
    }];
    let graphics = unsafe {
        create_graphics_pso(
            &device,
            &GraphicsPSODesc {
                vertex_spirv: &vertex,
                fragment_spirv: &fragment,
                color_targets: &targets,
                ..GraphicsPSODesc::default()
            },
        )?
    };
    reset_command(&mut command, u64::MAX)?;
    unsafe {
        barrier(
            &mut command,
            &[Barrier::Texture {
                texture: &target,
                before: (Stage::NONE, Access::NONE, TextureLayout::Undefined),
                after: (
                    Stage::COLOR_OUTPUT,
                    Access::COLOR_WRITE,
                    TextureLayout::ColorAttachment,
                ),
            }],
        )?;
        let attachments = [ColorAttachment {
            render_view: Some(&target_view),
            load: LoadOp::Clear,
            ..ColorAttachment::default()
        }];
        begin_render_pass(
            &mut command,
            &RenderingDesc {
                colors: &attachments,
                ..RenderingDesc::default()
            },
            U32x2 { x: 2, y: 2 },
        )?;
        bind_pso(&mut command, &graphics)?;
        draw(&mut command, 3, 1, 0, 0)?;
        end_render_pass(&mut command)?;
        barrier(
            &mut command,
            &[Barrier::Texture {
                texture: &target,
                before: (
                    Stage::COLOR_OUTPUT,
                    Access::COLOR_WRITE,
                    TextureLayout::ColorAttachment,
                ),
                after: (
                    Stage::TRANSFER,
                    Access::TRANSFER_READ,
                    TextureLayout::TransferSource,
                ),
            }],
        )?;
        copy_texture_to_memory(
            &mut command,
            &target,
            readback.range(0, 16)?,
            &TextureCopyDesc::default(),
            TextureAspect::Automatic,
        )?;
        barrier(
            &mut command,
            &[Barrier::Buffer {
                range: readback.range(0, 16)?,
                before: (Stage::TRANSFER, Access::TRANSFER_WRITE),
                after: (Stage::HOST, Access::HOST_READ),
            }],
        )?;
        submit(
            &mut device,
            &mut command,
            &[(&timeline, 1, Stage::ALL_COMMANDS)],
            &[(&timeline, 2, Stage::ALL_COMMANDS)],
        )?;
    }
    wait_timeline(&timeline, 2, u64::MAX)?;
    wait_command(&command, u64::MAX)?;
    unsafe {
        invalidate_memory(readback.range(0, 16)?)?;
        let mapped = mapped_memory(&mut readback, 0, 16)?;
        for pixel in mapped.cpu.as_chunks::<4>().0 {
            assert_eq!(pixel, &[0, 255, 0, 255]);
        }
    }
    if get_device_caps(&device).mesh_shader {
        let mesh_shader = compile_shader(
            r#"
            #version 460
            #extension GL_EXT_mesh_shader : require
            layout(local_size_x = 1) in;
            layout(triangles, max_vertices = 3, max_primitives = 1) out;
            void main() {
                SetMeshOutputsEXT(3, 1);
                gl_MeshVerticesEXT[0].gl_Position = vec4(-1,-1,0,1);
                gl_MeshVerticesEXT[1].gl_Position = vec4(3,-1,0,1);
                gl_MeshVerticesEXT[2].gl_Position = vec4(-1,3,0,1);
                gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0,1,2);
            }
        "#,
            "mesh",
        );
        let mesh = unsafe {
            create_mesh_pso(
                &device,
                &MeshPSODesc {
                    mesh_spirv: &mesh_shader,
                    fragment_spirv: &fragment,
                    color_targets: &targets,
                    ..MeshPSODesc::default()
                },
            )?
        };
        reset_command(&mut command, u64::MAX)?;
        unsafe {
            barrier(
                &mut command,
                &[Barrier::Texture {
                    texture: &target,
                    before: (
                        Stage::TRANSFER,
                        Access::TRANSFER_READ,
                        TextureLayout::TransferSource,
                    ),
                    after: (
                        Stage::COLOR_OUTPUT,
                        Access::COLOR_WRITE,
                        TextureLayout::ColorAttachment,
                    ),
                }],
            )?;
            let attachments = [ColorAttachment {
                render_view: Some(&target_view),
                load: LoadOp::Clear,
                ..ColorAttachment::default()
            }];
            begin_render_pass(
                &mut command,
                &RenderingDesc {
                    colors: &attachments,
                    ..RenderingDesc::default()
                },
                U32x2 { x: 2, y: 2 },
            )?;
            bind_pso(&mut command, &mesh)?;
            draw_mesh(&mut command, U32x3 { x: 1, y: 1, z: 1 })?;
            end_render_pass(&mut command)?;
            barrier(
                &mut command,
                &[Barrier::Texture {
                    texture: &target,
                    before: (
                        Stage::COLOR_OUTPUT,
                        Access::COLOR_WRITE,
                        TextureLayout::ColorAttachment,
                    ),
                    after: (
                        Stage::TRANSFER,
                        Access::TRANSFER_READ,
                        TextureLayout::TransferSource,
                    ),
                }],
            )?;
            copy_texture_to_memory(
                &mut command,
                &target,
                readback.range(0, 16)?,
                &TextureCopyDesc::default(),
                TextureAspect::Automatic,
            )?;
            barrier(
                &mut command,
                &[Barrier::Buffer {
                    range: readback.range(0, 16)?,
                    before: (Stage::TRANSFER, Access::TRANSFER_WRITE),
                    after: (Stage::HOST, Access::HOST_READ),
                }],
            )?;
            submit(&mut device, &mut command, &[], &[])?;
        }
        wait_command(&command, u64::MAX)?;
        unsafe {
            invalidate_memory(readback.range(0, 16)?)?;
            let mapped = mapped_memory(&mut readback, 0, 16)?;
            for pixel in mapped.cpu.as_chunks::<4>().0 {
                assert_eq!(pixel, &[0, 255, 0, 255]);
            }
        }
        println!("Mesh pipeline rendered successfully");
        drop(mesh);
    } else {
        assert!(matches!(
            unsafe { create_mesh_pso(&device, &MeshPSODesc::default()) },
            Err(GpuError::Unsupported(_))
        ));
        println!("Mesh feature rejection verified");
    }
    // Drop children before reading the validation counter, to test cleanup too.
    drop((
        command,
        graphics,
        target_view,
        target,
        pso,
        sampled_view,
        storage_view,
        sampler,
        sampled_heap,
        storage_heap,
        sampler_heap,
        sampled_texture,
        storage_texture,
        upload,
        output,
        readback,
        timeline,
    ));
    assert_eq!(validation_error_count(&device), 0);
    destroy_device(device)?;
    Ok(())
}
