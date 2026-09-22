//! Render a triangle: `cargo run -- --validation --frames 12`.
use simple_graphics::*;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    device: Option<GpuDevice>,
    pipeline: Option<PSO>,
    commands: Vec<CommandBuffer>,
    rendered: usize,
    frame_limit: Option<usize>,
    validation: bool,
    resize_test: bool,
    failure: Option<String>,
}

impl App {
    fn render(&mut self) -> Result<(), GpuError> {
        let pipeline = self.pipeline.as_ref().ok_or(GpuError::InvalidArgument(
            "graphics pipeline is not initialized",
        ))?;
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let command = &mut self.commands[self.rendered % 2];
        reset_command(command, u64::MAX)?;
        let frame = match acquire(device, u64::MAX) {
            Ok(frame) => frame,
            Err(GpuError::OutOfDate) => {
                recreate_swapchain(device)?;
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        // Each frame discards the previous image contents. The acquisition wait
        // in submit_and_present orders the transition after presentation.
        unsafe {
            barrier(
                command,
                &[Barrier::Texture {
                    texture: frame.render_view.texture(),
                    before: (Stage::NONE, Access::NONE, TextureLayout::Undefined),
                    after: (
                        Stage::COLOR_OUTPUT,
                        Access::COLOR_WRITE,
                        TextureLayout::ColorAttachment,
                    ),
                }],
            )?;
            let colors = [ColorAttachment {
                render_view: Some(&frame.render_view),
                load: LoadOp::Clear,
                store: StoreOp::Store,
                clear: ClearColor {
                    x: 0.03,
                    y: 0.12,
                    z: 0.3,
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
            bind_pso(command, pipeline)?;
            draw(command, 3, 1, 0, 0)?;
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
            self.pipeline = None;
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
            .create_window(Window::default_attributes().with_title("NoGraphicsAPI — Vulkan 1.4"))
        {
            Ok(window) => Arc::new(window),
            Err(error) => {
                self.failure = Some(error.to_string());
                event_loop.exit();
                return;
            }
        };
        let device = match create_device(&DeviceDesc {
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
        let vertex = spirv_words(include_bytes!("../shaders/triangle.vert.spv"));
        let fragment = spirv_words(include_bytes!("../shaders/triangle.frag.spv"));
        let targets = [ColorTargetDesc {
            format: Format::Bgra8Srgb,
            ..ColorTargetDesc::default()
        }];
        let pipeline = match unsafe {
            create_graphics_pso(
                &device,
                &GraphicsPSODesc {
                    vertex_spirv: &vertex,
                    fragment_spirv: &fragment,
                    color_targets: &targets,
                    ..GraphicsPSODesc::default()
                },
            )
        } {
            Ok(pipeline) => pipeline,
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
        self.pipeline = Some(pipeline);
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

fn spirv_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

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
