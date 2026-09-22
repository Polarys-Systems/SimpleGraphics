//! Cached compute-rasterized text: `cargo run -- --validation --frames 120`.

use simple_graphics::*;
use std::{collections::HashMap, error::Error, sync::Arc, time::Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod glyph_raster;
mod simple_ttf;

const FONT_PATH: &str = "./data/Syne/static/Syne-Regular.ttf";
const CACHE_CHARACTERS: &str = "Hello TextFPS:0123456789.";
const ATLAS_SIZE: u32 = 1024;
const ATLAS_PPEM: f32 = 96.0;
const MAX_GLYPH_INSTANCES: usize = 128;
const GLYPH_INSTANCE_SIZE: usize = 32;

#[derive(Clone, Copy)]
struct CachedBitmap {
    origin: [i32; 2],
    size: [u32; 2],
    uv_min: [f32; 2],
    uv_max: [f32; 2],
}

#[derive(Clone, Copy)]
struct CachedGlyph {
    advance: f32,
    bitmap: Option<CachedBitmap>,
}

struct PendingRaster {
    placement: glyph_raster::RasterPlacement,
    atlas_xy: [u32; 2],
    first_segment: u32,
    segment_count: u32,
}

struct TextRenderer {
    graphics_pipeline: PSO,
    _atlas_pipeline: PSO,
    _curve_buffer: GpuHeap,
    _atlas: Texture,
    _atlas_view: RenderView,
    sampled_heap: GpuHeap,
    storage_heap: GpuHeap,
    sampler_heap: GpuHeap,
    _sampler: Sampler,
    instance_buffers: Vec<GpuHeap>,
    glyphs: HashMap<char, CachedGlyph>,
    last_frame: Instant,
    smoothed_fps: f32,
    animation_time: f32,
}

impl TextRenderer {
    fn text_width(&self, text: &str, ppem: f32) -> f32 {
        let scale = ppem / ATLAS_PPEM;
        text.chars()
            .filter_map(|character| self.glyphs.get(&character))
            .map(|glyph| glyph.advance * scale)
            .sum()
    }

    fn append_text(
        &self,
        bytes: &mut Vec<u8>,
        text: &str,
        mut pen_x: f32,
        baseline_y: f32,
        ppem: f32,
    ) -> Result<(), GpuError> {
        let scale = ppem / ATLAS_PPEM;
        for character in text.chars() {
            let glyph = self
                .glyphs
                .get(&character)
                .ok_or(GpuError::InvalidArgument("text contains an uncached glyph"))?;
            if let Some(bitmap) = glyph.bitmap {
                for value in [
                    pen_x + bitmap.origin[0] as f32 * scale,
                    baseline_y + bitmap.origin[1] as f32 * scale,
                    bitmap.size[0] as f32 * scale,
                    bitmap.size[1] as f32 * scale,
                    bitmap.uv_min[0],
                    bitmap.uv_min[1],
                    bitmap.uv_max[0],
                    bitmap.uv_max[1],
                ] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            pen_x += glyph.advance * scale;
        }
        Ok(())
    }

    fn update_instances(&mut self, frame: usize, extent: U32x2) -> Result<u32, GpuError> {
        let now = Instant::now();
        let delta_time = now
            .duration_since(self.last_frame)
            .as_secs_f32()
            .clamp(0.000_1, 0.25);
        self.last_frame = now;
        self.animation_time = (self.animation_time + delta_time * 1.8) % std::f32::consts::TAU;
        let instantaneous_fps = 1.0 / delta_time;
        let smoothing = 1.0 - (-delta_time * 5.0).exp();
        self.smoothed_fps += (instantaneous_fps - self.smoothed_fps) * smoothing;

        let width = extent.x as f32;
        let height = extent.y as f32;
        let hello_size = (width / 7.5).min(height * 0.24).clamp(24.0, 128.0);
        let pulse = 1.0 + self.animation_time.sin() * 0.42;
        let fps_size = (height * 0.115).clamp(30.0, 72.0) * pulse;
        let fps_label = format!("FPS: {:03.0}", self.smoothed_fps.clamp(0.0, 999.0));

        let mut bytes = Vec::with_capacity(MAX_GLYPH_INSTANCES * GLYPH_INSTANCE_SIZE);
        let hello = "Hello Text";
        self.append_text(
            &mut bytes,
            hello,
            (width - self.text_width(hello, hello_size)) * 0.5,
            height * 0.42,
            hello_size,
        )?;
        self.append_text(
            &mut bytes,
            &fps_label,
            (width - self.text_width(&fps_label, fps_size)) * 0.5,
            height * 0.68,
            fps_size,
        )?;

        let instance_count = bytes.len() / GLYPH_INSTANCE_SIZE;
        if instance_count > MAX_GLYPH_INSTANCES {
            return Err(GpuError::InvalidArgument("text instance buffer is full"));
        }
        unsafe {
            write_memory(&mut self.instance_buffers[frame], 0, &bytes)?;
        }
        Ok(instance_count as u32)
    }
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    device: Option<GpuDevice>,
    renderer: Option<TextRenderer>,
    commands: Vec<CommandBuffer>,
    rendered: usize,
    frame_limit: Option<usize>,
    validation: bool,
    resize_test: bool,
    failure: Option<String>,
}

impl App {
    fn render(&mut self) -> Result<(), GpuError> {
        let frame_index = self.rendered % 2;
        let command = &mut self.commands[frame_index];
        reset_command(command, u64::MAX)?;
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let frame = match acquire(device, u64::MAX) {
            Ok(frame) => frame,
            Err(GpuError::OutOfDate) => {
                recreate_swapchain(device)?;
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let renderer = self.renderer.as_mut().ok_or(GpuError::InvalidArgument(
            "text renderer is not initialized",
        ))?;
        let instance_count = renderer.update_instances(frame_index, frame.extent)?;
        let instance_buffer = &renderer.instance_buffers[frame_index];

        unsafe {
            barrier(
                command,
                &[
                    Barrier::Buffer {
                        range: instance_buffer.range(0, instance_buffer.size())?,
                        before: (Stage::HOST, Access::HOST_WRITE),
                        after: (Stage::VERTEX, Access::SHADER_READ),
                    },
                    Barrier::Texture {
                        texture: frame.render_view.texture(),
                        before: (Stage::NONE, Access::NONE, TextureLayout::Undefined),
                        after: (
                            Stage::COLOR_OUTPUT,
                            Access::COLOR_WRITE,
                            TextureLayout::ColorAttachment,
                        ),
                    },
                ],
            )?;
            let colors = [ColorAttachment {
                render_view: Some(&frame.render_view),
                load: LoadOp::Clear,
                store: StoreOp::Store,
                clear: ClearColor {
                    x: 0.025,
                    y: 0.035,
                    z: 0.055,
                    w: 1.0,
                },
            }];
            begin_render_pass(
                command,
                &RenderingDesc {
                    colors: &colors,
                    ..RenderingDesc::default()
                },
                frame.extent,
            )?;
            bind_pso(command, &renderer.graphics_pipeline)?;
            bind_descriptor_heaps(
                command,
                [
                    &renderer.sampled_heap,
                    &renderer.storage_heap,
                    &renderer.sampler_heap,
                ],
            )?;
            let mut push = Vec::with_capacity(16);
            push.extend_from_slice(&instance_buffer.device_address().to_le_bytes());
            push.extend_from_slice(&(frame.extent.x as f32).to_le_bytes());
            push.extend_from_slice(&(frame.extent.y as f32).to_le_bytes());
            push_constants(command, 0, &push)?;
            draw(command, 6, instance_count, 0, 0)?;
            end_render_pass(command)?;
            barrier(
                command,
                &[Barrier::Texture {
                    texture: frame.render_view.texture(),
                    before: (
                        Stage::COLOR_OUTPUT,
                        Access::COLOR_WRITE,
                        TextureLayout::ColorAttachment,
                    ),
                    after: (Stage::NONE, Access::NONE, TextureLayout::Present),
                }],
            )?;
            match submit_and_present(device, command, &frame, &[], &[]) {
                Ok(true) | Err(GpuError::OutOfDate) => recreate_swapchain(device)?,
                Ok(false) => {}
                Err(error) => return Err(error),
            }
        }
        self.rendered += 1;
        Ok(())
    }

    fn shutdown(&mut self) {
        if let Some(device) = self.device.take() {
            if let Err(error) = wait_idle(&device) {
                self.failure = Some(error.to_string());
            }
            self.commands.clear();
            self.renderer = None;
            let errors = validation_error_count(&device);
            if errors > 0 {
                self.failure = Some(format!("{errors} Vulkan validation errors"));
            }
            if let Err(error) = destroy_device(device) {
                self.failure = Some(error.to_string());
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match event_loop
            .create_window(Window::default_attributes().with_title("NoGraphicsAPI - cached text"))
        {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.failure = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };
        let mut device = match create_device(&DeviceDesc {
            window: Some(window.clone()),
            swapchain_format: Format::Bgra8Srgb,
            validation: self.validation,
            ..DeviceDesc::default()
        }) {
            Ok(device) => device,
            Err(error) => {
                self.failure = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };

        let renderer = match create_text_renderer(&mut device) {
            Ok(renderer) => renderer,
            Err(error) => {
                self.failure = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };
        println!(
            "Device: {}",
            get_device_caps(&device).device_name.to_string_lossy()
        );
        for _ in 0..2 {
            match begin_command(&device) {
                Ok(command) => self.commands.push(command),
                Err(error) => {
                    self.failure = Some(error.to_string());
                    event_loop.exit();
                    return;
                }
            }
        }
        self.renderer = Some(renderer);
        self.device = Some(device);
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                match self.render() {
                    Ok(()) | Err(GpuError::WindowMinimized) => {}
                    Err(error) => {
                        self.failure = Some(error.to_string());
                        event_loop.exit();
                        return;
                    }
                }
                if self.resize_test
                    && self.rendered == 3
                    && let Some(window) = &self.window
                {
                    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(640, 360));
                }
                if self.resize_test
                    && self.rendered == 6
                    && let Some(device) = &mut self.device
                    && let Err(error) = recreate_swapchain(device)
                {
                    self.failure = Some(error.to_string());
                    event_loop.exit();
                    return;
                }
                if let Some(limit) = self.frame_limit
                    && self.rendered >= limit
                {
                    event_loop.exit();
                    return;
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(_) => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.shutdown();
    }
}

fn create_text_renderer(device: &mut GpuDevice) -> Result<TextRenderer, Box<dyn Error>> {
    if !supports_texture_format(
        device,
        Format::Rgba8Unorm,
        TextureUsage::SAMPLED | TextureUsage::STORAGE,
    ) {
        return Err(GpuError::Unsupported("RGBA8 sampled storage atlas".into()).into());
    }

    let parser = simple_ttf::TTFParser::read_from(FONT_PATH)?;
    let units_per_em = parser.units_per_em();
    let mut glyphs = HashMap::new();
    let mut segments = Vec::new();
    let mut pending = Vec::new();
    let mut atlas_x = 0;
    let mut atlas_y = 0;
    let mut row_height = 0;

    for character in CACHE_CHARACTERS.chars() {
        if glyphs.contains_key(&character) {
            continue;
        }
        let glyph_id = parser.glyph_index(character as u32);
        let metrics = parser
            .horizontal_metrics(glyph_id)
            .ok_or("font returned no horizontal metrics")?;
        let advance = f32::from(metrics.advance_width) * ATLAS_PPEM / f32::from(units_per_em);
        let Some(glyph) = parser.load_glyph(glyph_id)? else {
            glyphs.insert(
                character,
                CachedGlyph {
                    advance,
                    bitmap: None,
                },
            );
            continue;
        };
        let prepared = glyph_raster::prepare_glyph(&glyph)?;
        let Some(placement) = prepared.placement(ATLAS_PPEM, units_per_em, [0.0, 0.0])? else {
            glyphs.insert(
                character,
                CachedGlyph {
                    advance,
                    bitmap: None,
                },
            );
            continue;
        };
        if atlas_x + placement.size[0] > ATLAS_SIZE {
            atlas_x = 0;
            atlas_y += row_height + 1;
            row_height = 0;
        }
        if placement.size[0] > ATLAS_SIZE || atlas_y + placement.size[1] > ATLAS_SIZE {
            return Err("cached glyphs do not fit in the texture atlas".into());
        }
        let atlas_xy = [atlas_x, atlas_y];
        let first_segment = u32::try_from(segments.len())?;
        let segment_count = u32::try_from(prepared.segments.len())?;
        segments.extend_from_slice(&prepared.segments);
        pending.push(PendingRaster {
            placement,
            atlas_xy,
            first_segment,
            segment_count,
        });
        let atlas_scale = 1.0 / ATLAS_SIZE as f32;
        glyphs.insert(
            character,
            CachedGlyph {
                advance,
                bitmap: Some(CachedBitmap {
                    origin: placement.origin,
                    size: placement.size,
                    uv_min: [atlas_x as f32 * atlas_scale, atlas_y as f32 * atlas_scale],
                    uv_max: [
                        (atlas_x + placement.size[0]) as f32 * atlas_scale,
                        (atlas_y + placement.size[1]) as f32 * atlas_scale,
                    ],
                }),
            },
        );
        atlas_x += placement.size[0] + 1;
        row_height = row_height.max(placement.size[1]);
    }

    let total_segments = u32::try_from(segments.len())?;
    let mut jobs = vec![glyph_raster::RasterJob {
        rect: [0, 0, ATLAS_SIZE, ATLAS_SIZE],
        segment_range: [0, 0, total_segments, 0],
        mapping: [1.0, 0.0, 0.0, 0.0],
    }];
    jobs.extend(
        pending
            .iter()
            .map(|entry| {
                entry.placement.job(
                    entry.atlas_xy,
                    [ATLAS_SIZE; 2],
                    entry.first_segment,
                    entry.segment_count,
                    total_segments,
                )
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    let segment_bytes = glyph_raster::encode_segments(&segments);
    let mut curve_buffer =
        create_gpu_heap(device, segment_bytes.len() as u64, MemoryType::CpuVisible)?;
    unsafe {
        write_memory(&mut curve_buffer, 0, &segment_bytes)?;
    }

    let atlas = create_texture(
        device,
        &TextureDesc {
            extent: U32x3 {
                x: ATLAS_SIZE,
                y: ATLAS_SIZE,
                z: 1,
            },
            format: Format::Rgba8Unorm,
            usage: TextureUsage::SAMPLED | TextureUsage::STORAGE,
            ..TextureDesc::default()
        },
    )?;
    let atlas_view = create_texture_view(&atlas, &TextureDescriptorDesc::default())?;
    let sampler = create_sampler(
        device,
        &SamplerDesc {
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            address_w: AddressMode::ClampToEdge,
            ..SamplerDesc::default()
        },
    )?;
    let mut sampled_heap = create_texture_descriptor_heap(device, TextureDescriptorType::Sampled)?;
    let mut storage_heap = create_texture_descriptor_heap(device, TextureDescriptorType::Storage)?;
    let mut sampler_heap = create_sampler_descriptor_heap(device)?;
    unsafe {
        write_texture_descriptor(&mut sampled_heap, 0, &atlas_view)?;
        write_texture_descriptor(&mut storage_heap, 0, &atlas_view)?;
        write_sampler_descriptor(&mut sampler_heap, 0, &sampler)?;
    }

    let atlas_spirv = spirv_words(include_bytes!("../shaders/texture_atlas.comp.spv"));
    let atlas_pipeline = unsafe { create_compute_pso(device, &atlas_spirv)? };
    let vertex = spirv_words(include_bytes!("../shaders/text.vert.spv"));
    let fragment = spirv_words(include_bytes!("../shaders/text.frag.spv"));
    let targets = [ColorTargetDesc {
        format: Format::Bgra8Srgb,
        blend: BlendState {
            enabled: true,
            color: BlendComponentState {
                source: BlendFactor::SourceAlpha,
                destination: BlendFactor::OneMinusSourceAlpha,
                operation: BlendOp::Add,
            },
            alpha: BlendComponentState {
                source: BlendFactor::One,
                destination: BlendFactor::OneMinusSourceAlpha,
                operation: BlendOp::Add,
            },
        },
        ..ColorTargetDesc::default()
    }];
    let graphics_pipeline = unsafe {
        create_graphics_pso(
            device,
            &GraphicsPSODesc {
                vertex_spirv: &vertex,
                fragment_spirv: &fragment,
                color_targets: &targets,
                ..GraphicsPSODesc::default()
            },
        )?
    };

    let mut command = begin_command(device)?;
    unsafe {
        barrier(
            &mut command,
            &[
                Barrier::Texture {
                    texture: &atlas,
                    before: (Stage::NONE, Access::NONE, TextureLayout::Undefined),
                    after: (Stage::COMPUTE, Access::SHADER_WRITE, TextureLayout::General),
                },
                Barrier::Memory {
                    before: (Stage::HOST, Access::HOST_WRITE),
                    after: (
                        Stage::COMPUTE,
                        Access::SHADER_READ | Access::DESCRIPTOR_READ,
                    ),
                },
            ],
        )?;
        bind_pso(&mut command, &atlas_pipeline)?;
        bind_descriptor_heaps(&mut command, [&sampled_heap, &storage_heap, &sampler_heap])?;
        for job in &jobs {
            let mut push = [0_u8; 64];
            push[..8].copy_from_slice(&curve_buffer.device_address().to_le_bytes());
            push[16..].copy_from_slice(&job.to_le_bytes());
            push_constants(&mut command, 0, &push)?;
            let groups = job.workgroups();
            dispatch(
                &mut command,
                U32x3 {
                    x: groups[0],
                    y: groups[1],
                    z: groups[2],
                },
            )?;
        }
        barrier(
            &mut command,
            &[Barrier::Texture {
                texture: &atlas,
                before: (Stage::COMPUTE, Access::SHADER_WRITE, TextureLayout::General),
                after: (Stage::FRAGMENT, Access::SHADER_READ, TextureLayout::Sampled),
            }],
        )?;
        submit(device, &mut command, &[], &[])?;
    }
    wait_command(&command, u64::MAX)?;

    let mut instance_buffers = Vec::with_capacity(2);
    for _ in 0..2 {
        instance_buffers.push(create_gpu_heap(
            device,
            (MAX_GLYPH_INSTANCES * GLYPH_INSTANCE_SIZE) as u64,
            MemoryType::CpuVisible,
        )?);
    }
    Ok(TextRenderer {
        graphics_pipeline,
        _atlas_pipeline: atlas_pipeline,
        _curve_buffer: curve_buffer,
        _atlas: atlas,
        _atlas_view: atlas_view,
        sampled_heap,
        storage_heap,
        sampler_heap,
        _sampler: sampler,
        instance_buffers,
        glyphs,
        last_frame: Instant::now(),
        smoothed_fps: 60.0,
        animation_time: 0.0,
    })
}

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty());
    words
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut app = App::default();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--validation" => app.validation = true,
            "--resize-test" => app.resize_test = true,
            "--frames" => {
                app.frame_limit = Some(
                    arguments
                        .next()
                        .ok_or("--frames requires a count")?
                        .parse()?,
                )
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    EventLoop::new()?.run_app(&mut app)?;
    if let Some(error) = app.failure {
        return Err(error.into());
    }
    Ok(())
}
