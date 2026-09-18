use crate::*;
use ash::{Entry, ext, khr};
use std::{
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Single-queue graphics/compute device. Kept on its creating thread.
/// Resource handles retain the backend; explicit destruction waits for idle.
pub struct GpuDevice {
    pub(crate) state: Rc<DeviceState>,
    pub(crate) swapchain: Option<Rc<TextureHeapOwner>>,
    pub(crate) views: Vec<Rc<RenderView>>,
    pub(crate) present_ready: Vec<vk::Semaphore>,
    pub(crate) acquire_ready: [vk::Semaphore; 2],
    pub(crate) acquire_fences: [vk::Fence; 2],
    pub(crate) acquire_pending: [bool; 2],
    pub(crate) retirement: vk::Semaphore,
    pub(crate) serial: u64,
    pub(crate) slot_values: [u64; 2],
    pub(crate) active: Option<(u32, usize)>,
    pub(crate) generation: u64,
    pub(crate) extent: U32x2,
    pub(crate) format: Format,
    pub(crate) present_failed: bool,
    image_count: u32,
}

// The only additional backend structure: RAII also handles partially initialized
// devices, and Rc prevents destruction of Vulkan parents before their children.
pub(crate) struct DeviceState {
    pub entry: Entry,
    pub instance: ash::Instance,
    pub raw: Option<ash::Device>,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub surface_api: khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    debug_api: Option<ext::debug_utils::Instance>,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_errors: Arc<AtomicUsize>,
    pub window: Option<Arc<Window>>,
    pub allocator: Option<vk_mem::Allocator>,
    pub descriptor_api: Option<ext::descriptor_buffer::Device>,
    pub mesh_api: Option<ext::mesh_shader::Device>,
    pub swapchain_api: Option<khr::swapchain::Device>,
    pub caps: DeviceCaps,
    pub limits: vk::PhysicalDeviceLimits,
    pub mesh_limits: vk::PhysicalDeviceMeshShaderPropertiesEXT<'static>,
    pub descriptor_sizes: [u64; 3],
    pub descriptor_offsets: [u64; 3],
    pub heap_sizes: [u64; 3],
    pub layouts: [vk::DescriptorSetLayout; 3],
    pub pipeline_layout: vk::PipelineLayout,
    pub timestamp_query_count: u32,
    pub timestamp_valid_bits: u32,
}

impl DeviceState {
    pub fn raw(&self) -> &ash::Device {
        self.raw.as_ref().expect("initialized device")
    }
    pub fn allocator(&self) -> &vk_mem::Allocator {
        self.allocator.as_ref().expect("initialized allocator")
    }
    pub fn descriptors(&self) -> &ext::descriptor_buffer::Device {
        self.descriptor_api
            .as_ref()
            .expect("descriptor buffers enabled")
    }
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        unsafe {
            if let Some(raw) = &self.raw {
                let _ = raw.device_wait_idle();
                raw.destroy_pipeline_layout(self.pipeline_layout, None);
                for layout in self.layouts {
                    raw.destroy_descriptor_set_layout(layout, None);
                }
            }
            self.allocator.take();
            if let Some(raw) = self.raw.take() {
                raw.destroy_device(None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_api.destroy_surface(self.surface, None);
            }
            if let Some(debug) = &self.debug_api {
                debug.destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Initialize Vulkan 1.4, a single graphics+compute queue, and descriptor heaps.
/// Missing requirements are reported for each rejected physical device.
/// A window is optional; its Arc is retained for the lifetime of the surface.
pub fn create_device(desc: &DeviceDesc) -> Result<GpuDevice, GpuError> {
    if desc.descriptor_count == 0 || desc.desired_swapchain_image_count == 0 {
        return Err(GpuError::InvalidArgument(
            "descriptor and swapchain counts must be nonzero",
        ));
    }
    unsafe {
        let entry = match Entry::load() {
            Ok(entry) => entry,
            Err(error) => return Err(GpuError::Loader(error.to_string())),
        };
        let version = entry
            .try_enumerate_instance_version()?
            .unwrap_or(vk::API_VERSION_1_0);
        if version < VULKAN_API_VERSION {
            return Err(GpuError::Unsupported(
                "Vulkan loader 1.4 is required".into(),
            ));
        }
        let mut extensions = Vec::new();
        if let Some(window) = &desc.window {
            let display = window.display_handle().map_err(window_handle_error)?;
            extensions
                .extend_from_slice(ash_window::enumerate_required_extensions(display.as_raw())?);
        }
        let mut layers = Vec::new();
        if desc.validation {
            let mut found = false;
            for layer in entry.enumerate_instance_layer_properties()? {
                if layer.layer_name_as_c_str()? == c"VK_LAYER_KHRONOS_validation" {
                    found = true;
                }
            }
            if !found {
                return Err(GpuError::Unsupported(
                    "VK_LAYER_KHRONOS_validation is not installed".into(),
                ));
            }
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
            extensions.push(ext::debug_utils::NAME.as_ptr());
            extensions.push(ext::validation_features::NAME.as_ptr());
        }
        let validation_errors = Arc::new(AtomicUsize::new(0));
        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback))
            .user_data(Arc::as_ptr(&validation_errors).cast_mut().cast());
        let app = vk::ApplicationInfo::default()
            .application_name(c"NoGraphicsAPI")
            .api_version(VULKAN_API_VERSION);
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        let enabled_validation = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        let mut validation_features =
            vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled_validation);
        if desc.validation {
            instance_info = instance_info
                .push_next(&mut debug_info)
                .push_next(&mut validation_features);
        }
        let instance = entry.create_instance(&instance_info, None)?;
        let surface_api = khr::surface::Instance::new(&entry, &instance);
        let mut state = DeviceState {
            entry,
            instance,
            raw: None,
            physical: vk::PhysicalDevice::null(),
            queue: vk::Queue::null(),
            queue_family: 0,
            surface_api,
            surface: vk::SurfaceKHR::null(),
            window: desc.window.clone(),
            allocator: None,
            debug_api: None,
            debug_messenger: vk::DebugUtilsMessengerEXT::null(),
            validation_errors,
            descriptor_api: None,
            mesh_api: None,
            swapchain_api: None,
            caps: DeviceCaps::default(),
            limits: vk::PhysicalDeviceLimits::default(),
            mesh_limits: vk::PhysicalDeviceMeshShaderPropertiesEXT::default(),
            descriptor_sizes: [0; 3],
            descriptor_offsets: [0; 3],
            heap_sizes: [0; 3],
            layouts: [vk::DescriptorSetLayout::null(); 3],
            pipeline_layout: vk::PipelineLayout::null(),
            timestamp_query_count: desc.timestamp_query_count,
            timestamp_valid_bits: 0,
        };
        if desc.validation {
            state.debug_api = Some(ext::debug_utils::Instance::new(
                &state.entry,
                &state.instance,
            ));
            state.debug_messenger = state
                .debug_api
                .as_ref()
                .expect("validation enabled")
                .create_debug_utils_messenger(&debug_info, None)?;
        }
        if let Some(window) = &state.window {
            state.surface = ash_window::create_surface(
                &state.entry,
                &state.instance,
                window
                    .display_handle()
                    .map_err(window_handle_error)?
                    .as_raw(),
                window
                    .window_handle()
                    .map_err(window_handle_error)?
                    .as_raw(),
                None,
            )?;
        }
        let mut rejected = Vec::new();
        let mut best_score = 0;
        for physical in state.instance.enumerate_physical_devices()? {
            let properties = state.instance.get_physical_device_properties(physical);
            let name = properties.device_name_as_c_str()?.to_string_lossy();
            if properties.api_version < VULKAN_API_VERSION {
                rejected.push(format!(
                    "{name}: Vulkan 1.4 required (reports {}.{})",
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version)
                ));
                continue;
            }
            let mut extension_names = Vec::new();
            for extension in state
                .instance
                .enumerate_device_extension_properties(physical)?
            {
                extension_names.push(extension.extension_name_as_c_str()?.to_owned());
            }
            let descriptor_buffer =
                extension_names.contains(&ext::descriptor_buffer::NAME.to_owned());
            let mesh_shader = extension_names.contains(&ext::mesh_shader::NAME.to_owned());
            let present = state.surface == vk::SurfaceKHR::null()
                || extension_names.contains(&khr::swapchain::NAME.to_owned());
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut descriptor = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
            let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut features12)
                .push_next(&mut features13);
            if descriptor_buffer {
                features = features.push_next(&mut descriptor);
            }
            if mesh_shader {
                features = features.push_next(&mut mesh);
            }
            state
                .instance
                .get_physical_device_features2(physical, &mut features);
            let base = features.features;
            let mut missing = required_features(
                properties.api_version,
                &base,
                &features12,
                &features13,
                descriptor.descriptor_buffer != 0,
            );
            if !present {
                missing.push("VK_KHR_swapchain");
            }
            let mut family = None;
            let queues = state
                .instance
                .get_physical_device_queue_family_properties(physical);
            for (index, queue) in queues.iter().enumerate() {
                if !queue
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                    || queue.queue_count == 0
                {
                    continue;
                }
                if state.surface != vk::SurfaceKHR::null()
                    && !state.surface_api.get_physical_device_surface_support(
                        physical,
                        index as u32,
                        state.surface,
                    )?
                {
                    continue;
                }
                family = Some(index as u32);
                break;
            }
            if family.is_none() {
                missing.push("one graphics+compute+present queue family");
            }
            let limits = properties.limits;
            let count = desc.descriptor_count;
            if count > limits.max_per_stage_descriptor_sampled_images
                || count > limits.max_per_stage_descriptor_storage_images
                || count > limits.max_per_stage_descriptor_samplers
                || count > limits.max_descriptor_set_sampled_images
                || count > limits.max_descriptor_set_storage_images
                || count > limits.max_descriptor_set_samplers
                || u64::from(count) * 3 > u64::from(limits.max_per_stage_resources)
            {
                missing.push("requested bindless descriptor count exceeds device limits");
            }
            if !missing.is_empty() {
                rejected.push(format!("{name}: {}", missing.join(", ")));
                continue;
            }
            let score = match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            };
            if score <= best_score {
                continue;
            }
            best_score = score;
            state.physical = physical;
            state.queue_family = family.expect("checked queue family");
            state.timestamp_valid_bits = queues[state.queue_family as usize].timestamp_valid_bits;
            state.limits = limits;
            state.caps = DeviceCaps {
                device_name: properties.device_name_as_c_str()?.to_owned(),
                api_version: properties.api_version,
                buffer_device_address: true,
                dynamic_rendering: true,
                synchronization2: true,
                timeline_semaphore: true,
                descriptor_buffer: true,
                bindless_descriptors: true,
                mesh_shader: mesh.mesh_shader != 0,
                sampler_anisotropy: base.sampler_anisotropy != 0,
                max_sampler_anisotropy: limits.max_sampler_anisotropy,
                max_color_attachments: limits.max_color_attachments,
                descriptor_count: count,
                max_push_data_size: u64::from(limits.max_push_constants_size),
                texture_heap_alignment: limits.buffer_image_granularity,
                timestamp_period_ns: limits.timestamp_period,
                sub_texel_precision_bits: limits.sub_texel_precision_bits,
                texture_compression_bc: base.texture_compression_bc != 0,
                texture_compression_astc: base.texture_compression_astc_ldr != 0,
                ..DeviceCaps::default()
            };
        }
        if best_score == 0 {
            return Err(GpuError::Unsupported(format!(
                "no suitable Vulkan 1.4 device; {}",
                rejected.join("; ")
            )));
        }
        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(state.queue_family)
            .queue_priorities(&priorities)];
        let mut extensions = vec![ext::descriptor_buffer::NAME.as_ptr()];
        if state.surface != vk::SurfaceKHR::null() {
            extensions.push(khr::swapchain::NAME.as_ptr());
        }
        if state.caps.mesh_shader {
            extensions.push(ext::mesh_shader::NAME.as_ptr());
        }
        let base = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .independent_blend(true)
            .shader_storage_image_extended_formats(true)
            .image_cube_array(true)
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .sampler_anisotropy(state.caps.sampler_anisotropy)
            .texture_compression_bc(state.caps.texture_compression_bc)
            .texture_compression_astc_ldr(state.caps.texture_compression_astc);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_partially_bound(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true);
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let mut descriptor =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
        let mut mesh =
            vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(state.caps.mesh_shader);
        let mut info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&extensions)
            .enabled_features(&base)
            .push_next(&mut features12)
            .push_next(&mut features13)
            .push_next(&mut descriptor);
        if state.caps.mesh_shader {
            info = info.push_next(&mut mesh);
        }
        state.raw = Some(state.instance.create_device(state.physical, &info, None)?);
        state.queue = state.raw().get_device_queue(state.queue_family, 0);
        state.descriptor_api = Some(ext::descriptor_buffer::Device::new(
            &state.instance,
            state.raw(),
        ));
        if state.caps.mesh_shader {
            state.mesh_api = Some(ext::mesh_shader::Device::new(&state.instance, state.raw()));
        }
        if state.surface != vk::SurfaceKHR::null() {
            state.swapchain_api = Some(khr::swapchain::Device::new(&state.instance, state.raw()));
        }
        let mut descriptor_properties = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut mesh_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
        let mut properties12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut descriptor_properties)
            .push_next(&mut properties12);
        if state.caps.mesh_shader {
            properties = properties.push_next(&mut mesh_properties);
        }
        state
            .instance
            .get_physical_device_properties2(state.physical, &mut properties);
        state.mesh_limits = mesh_properties;
        state.caps.max_timeline_value_difference =
            properties12.max_timeline_semaphore_value_difference;
        state.mesh_limits.p_next = std::ptr::null_mut();
        if descriptor_properties.max_descriptor_buffer_bindings < 3
            || descriptor_properties.max_resource_descriptor_buffer_bindings < 2
            || descriptor_properties.max_sampler_descriptor_buffer_bindings < 1
        {
            return Err(GpuError::Unsupported(format!(
                "{}: three descriptor buffer bindings required",
                state.caps.device_name.to_string_lossy()
            )));
        }
        state.caps.descriptor_buffer_alignment =
            descriptor_properties.descriptor_buffer_offset_alignment;
        state.descriptor_sizes = [
            descriptor_properties.sampled_image_descriptor_size as u64,
            descriptor_properties.storage_image_descriptor_size as u64,
            descriptor_properties.sampler_descriptor_size as u64,
        ];
        state.caps.texture_descriptor_size =
            state.descriptor_sizes[0].max(state.descriptor_sizes[1]);
        state.caps.sampler_descriptor_size = state.descriptor_sizes[2];
        let types = [
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::SAMPLER,
        ];
        for (index, descriptor_type) in types.into_iter().enumerate() {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(descriptor_type)
                .descriptor_count(desc.descriptor_count)
                .stage_flags(vk::ShaderStageFlags::ALL)];
            let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND];
            let mut binding_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
            state.layouts[index] = state.raw().create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
                    .bindings(&bindings)
                    .push_next(&mut binding_flags),
                None,
            )?;
            state.heap_sizes[index] = state
                .descriptors()
                .get_descriptor_set_layout_size(state.layouts[index]);
            state.descriptor_offsets[index] = state
                .descriptors()
                .get_descriptor_set_layout_binding_offset(state.layouts[index], 0);
            let maximum = if index == 2 {
                descriptor_properties.max_sampler_descriptor_buffer_range
            } else {
                descriptor_properties.max_resource_descriptor_buffer_range
            };
            if state.heap_sizes[index] > maximum {
                return Err(GpuError::Unsupported(
                    "descriptor heap exceeds descriptor-buffer range".into(),
                ));
            }
        }
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .size(state.limits.max_push_constants_size)];
        state.pipeline_layout = state.raw().create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&state.layouts)
                .push_constant_ranges(&ranges),
            None,
        )?;
        let mut allocator_info =
            vk_mem::AllocatorCreateInfo::new(&state.instance, state.raw(), state.physical);
        allocator_info.vulkan_api_version = VULKAN_API_VERSION;
        allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        state.allocator = Some(vk_mem::Allocator::new(allocator_info)?);
        let mut device = GpuDevice {
            state: Rc::new(state),
            swapchain: None,
            views: Vec::new(),
            present_ready: Vec::new(),
            acquire_ready: [vk::Semaphore::null(); 2],
            retirement: vk::Semaphore::null(),
            serial: 0,
            acquire_fences: [vk::Fence::null(); 2],
            acquire_pending: [false; 2],
            slot_values: [0; 2],
            active: None,
            generation: 0,
            extent: U32x2::default(),
            format: desc.swapchain_format,
            image_count: desc.desired_swapchain_image_count,
            present_failed: false,
        };
        let mut semaphore_type =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        device.retirement = device.state.raw().create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type),
            None,
        )?;
        for index in 0..2 {
            device.acquire_ready[index] = device
                .state
                .raw()
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            device.acquire_fences[index] = device
                .state
                .raw()
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
        }
        if device.state.window.is_some() {
            match recreate_swapchain(&mut device) {
                Ok(()) | Err(GpuError::WindowMinimized) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(device)
    }
}

fn window_handle_error(error: winit::raw_window_handle::HandleError) -> GpuError {
    GpuError::Unsupported(error.to_string())
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    unsafe {
        if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) && !user.is_null() {
            (*user.cast::<AtomicUsize>()).fetch_add(1, Ordering::Relaxed);
        }
        if let Some(data) = data.as_ref()
            && let Some(message) = data.message_as_c_str()
        {
            eprintln!("Vulkan validation: {}", message.to_string_lossy());
        }
    }
    vk::FALSE
}

/// Number of validation errors observed since instance creation (zero when disabled).
pub fn validation_error_count(device: &GpuDevice) -> usize {
    device.state.validation_errors.load(Ordering::Relaxed)
}

impl From<std::ffi::FromBytesUntilNulError> for GpuError {
    fn from(_: std::ffi::FromBytesUntilNulError) -> Self {
        Self::InvalidArgument("driver returned an unterminated name")
    }
}

pub(crate) fn required_features(
    version: u32,
    base: &vk::PhysicalDeviceFeatures,
    features12: &vk::PhysicalDeviceVulkan12Features<'_>,
    features13: &vk::PhysicalDeviceVulkan13Features<'_>,
    descriptors: bool,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    for (available, name) in [
        (version >= VULKAN_API_VERSION, "Vulkan 1.4"),
        (
            base.shader_int64 != 0,
            "shaderInt64 (device-address shader ABI)",
        ),
        (base.independent_blend != 0, "independentBlend"),
        (base.image_cube_array != 0, "imageCubeArray"),
        (
            base.fragment_stores_and_atomics != 0,
            "fragmentStoresAndAtomics",
        ),
        (
            base.vertex_pipeline_stores_and_atomics != 0,
            "vertexPipelineStoresAndAtomics",
        ),
        (
            base.shader_storage_image_extended_formats != 0,
            "shaderStorageImageExtendedFormats",
        ),
        (features12.buffer_device_address != 0, "bufferDeviceAddress"),
        (features12.timeline_semaphore != 0, "timelineSemaphore"),
        (features13.dynamic_rendering != 0, "dynamicRendering"),
        (features13.synchronization2 != 0, "synchronization2"),
        (
            features13.maintenance4 != 0,
            "maintenance4 (SPIR-V LocalSizeId)",
        ),
        (descriptors, "VK_EXT_descriptor_buffer / descriptorBuffer"),
        (
            features12.runtime_descriptor_array != 0,
            "runtimeDescriptorArray",
        ),
        (
            features12.descriptor_binding_partially_bound != 0,
            "descriptorBindingPartiallyBound",
        ),
        (
            features12.shader_sampled_image_array_non_uniform_indexing != 0,
            "shaderSampledImageArrayNonUniformIndexing",
        ),
        (
            features12.shader_storage_image_array_non_uniform_indexing != 0,
            "shaderStorageImageArrayNonUniformIndexing",
        ),
    ] {
        if !available {
            missing.push(name);
        }
    }
    missing
}

/// Query enabled capabilities, not just hardware-advertised feature bits.
pub fn get_device_caps(device: &GpuDevice) -> &DeviceCaps {
    &device.state.caps
}

/// Wait for this device's queue. Prefer timeline waits in the normal frame loop.
pub fn wait_idle(device: &GpuDevice) -> Result<(), GpuError> {
    unsafe {
        device.state.raw().device_wait_idle()?;
    }
    Ok(())
}

/// Wait for idle and release the device owner. Resources retain their backend.
pub fn destroy_device(device: GpuDevice) -> Result<(), GpuError> {
    wait_idle(&device)?;
    drop(device);
    Ok(())
}

/// Current drawable size in physical pixels; zero dimensions mean minimized.
pub fn get_drawable_extent(device: &GpuDevice) -> Result<U32x2, GpuError> {
    let Some(window) = &device.state.window else {
        return Ok(U32x2::default());
    };
    let size = window.inner_size();
    if size.width == 0 || size.height == 0 {
        return Ok(U32x2::default());
    }
    let caps = unsafe {
        device
            .state
            .surface_api
            .get_physical_device_surface_capabilities(device.state.physical, device.state.surface)?
    };
    if caps.current_extent.width != u32::MAX {
        return Ok(U32x2 {
            x: caps.current_extent.width,
            y: caps.current_extent.height,
        });
    }
    Ok(U32x2 {
        x: size
            .width
            .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
        y: size
            .height
            .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    })
}

/// Rebuild after a resize or `OutOfDate`. No frame may be outstanding.
/// This infrequent operation waits for idle; normal acquire/present does not.
pub fn recreate_swapchain(device: &mut GpuDevice) -> Result<(), GpuError> {
    if device.active.is_some() {
        return Err(GpuError::InvalidArgument(
            "present the acquired frame before resizing",
        ));
    }
    let Some(api) = &device.state.swapchain_api else {
        return Err(GpuError::Unsupported(
            "headless device has no swapchain".into(),
        ));
    };
    let extent = get_drawable_extent(device)?;
    if extent.x == 0 || extent.y == 0 {
        return Err(GpuError::WindowMinimized);
    }
    wait_idle(device)?;
    unsafe {
        let caps = device
            .state
            .surface_api
            .get_physical_device_surface_capabilities(
                device.state.physical,
                device.state.surface,
            )?;
        let formats = device
            .state
            .surface_api
            .get_physical_device_surface_formats(device.state.physical, device.state.surface)?;
        let requested = if device.format == Format::Undefined {
            Format::Bgra8Srgb
        } else {
            device.format
        };
        let mut selected = None;
        for format in formats {
            if (format.format == requested.into() || format.format == vk::Format::UNDEFINED)
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                selected = Some(vk::SurfaceFormatKHR {
                    format: requested.into(),
                    color_space: format.color_space,
                });
                break;
            }
        }
        let Some(format) = selected else {
            return Err(GpuError::Unsupported(format!(
                "surface does not support {requested:?} / SRGB_NONLINEAR"
            )));
        };
        if !caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        {
            return Err(GpuError::Unsupported("swapchain color attachments".into()));
        }
        let mut alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;
        for candidate in [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ] {
            if caps.supported_composite_alpha.contains(candidate) {
                alpha = candidate;
                break;
            }
        }
        let mut image_count = device.image_count.max(caps.min_image_count);
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        let old_swapchain = match &device.swapchain {
            Some(owner) => owner.swapchain,
            None => vk::SwapchainKHR::null(),
        };
        // Passing oldSwapchain retires it even when creation fails. Require a
        // successful rebuild before another acquisition in that case.
        device.present_failed = true;
        let handle = api.create_swapchain(
            &vk::SwapchainCreateInfoKHR::default()
                .surface(device.state.surface)
                .old_swapchain(old_swapchain)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(vk::Extent2D {
                    width: extent.x,
                    height: extent.y,
                })
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(alpha)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true),
            None,
        )?;
        let owner = Rc::new(TextureHeapOwner {
            state: device.state.clone(),
            allocation: None,
            swapchain: handle,
        });
        let images = api.get_swapchain_images(handle)?;
        let mut views = Vec::new();
        for image in images {
            let texture = Texture {
                owner: owner.clone(),
                image,
                desc: TextureDesc {
                    extent: U32x3 {
                        x: extent.x,
                        y: extent.y,
                        z: 1,
                    },
                    format: requested,
                    usage: TextureUsage::COLOR_ATTACHMENT,
                    ..TextureDesc::default()
                },
            };
            views.push(Rc::new(create_render_view(
                &texture,
                &RenderViewDesc::default(),
            )?));
        }
        // Store each successfully created semaphore immediately, so Drop cleans up failures.
        for semaphore in device.present_ready.drain(..) {
            device.state.raw().destroy_semaphore(semaphore, None);
        }
        device.views = views;
        device.swapchain = Some(owner);
        device.extent = extent;
        device.format = requested;
        device.generation += 1;
        for _ in &device.views {
            device.present_ready.push(
                device
                    .state
                    .raw()
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
            );
        }
        device.present_failed = false;
    }
    Ok(())
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        unsafe {
            let raw = self.state.raw();
            let _ = raw.device_wait_idle();
            for (index, fence) in self.acquire_fences.into_iter().enumerate() {
                if self.acquire_pending[index] {
                    let _ = raw.wait_for_fences(&[fence], true, u64::MAX);
                }
                raw.destroy_fence(fence, None);
            }
            for semaphore in self.acquire_ready {
                raw.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.present_ready.drain(..) {
                raw.destroy_semaphore(semaphore, None);
            }
            raw.destroy_semaphore(self.retirement, None);
        }
    }
}
