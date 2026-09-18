use crate::{device::DeviceState, *};
use std::rc::Rc;
use vk_mem::Alloc;

/// A buffer allocation. Suballocate with `range`; all buffers support addresses,
/// transfers, indices, indirect arguments and shader storage.
pub struct GpuHeap {
    pub(crate) owner: Box<GpuHeapOwner>,
}
pub struct GpuHeapOwner {
    pub(crate) state: Rc<DeviceState>,
    pub(crate) buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    address: u64,
    size: u64,
    mapped: *mut u8,
    pub(crate) descriptor_set: Option<usize>,
}

/// Borrowed buffer subrange; a GPU address is never a Rust slice.
#[derive(Clone, Copy)]
pub struct GpuRange<'a> {
    pub(crate) heap: &'a GpuHeap,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// Explicitly borrowed mapping, obtained through `mapped_memory`.
pub struct GpuCpuRange<'a> {
    pub gpu: u64,
    pub cpu: &'a mut [u8],
}

/// Shared allocation owner for an image, or the images belonging to a swapchain.
pub struct TextureHeapOwner {
    pub(crate) state: Rc<DeviceState>,
    pub(crate) allocation: Option<(vk::Image, vk_mem::Allocation)>,
    pub(crate) swapchain: vk::SwapchainKHR,
}

/// Retains an image allocation. VMA handles placement and suballocation.
pub struct TextureHeap {
    pub size: u64,
    _owner: Rc<TextureHeapOwner>,
}

#[derive(Clone)]
pub struct Texture {
    pub(crate) owner: Rc<TextureHeapOwner>,
    pub(crate) image: vk::Image,
    pub(crate) desc: TextureDesc,
}

/// An image view retaining its texture allocation.
pub struct RenderView {
    pub(crate) texture: Texture,
    pub(crate) raw: vk::ImageView,
    pub(crate) extent: U32x2,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) subresources: vk::ImageSubresourceRange,
    pub(crate) view_type: vk::ImageViewType,
}

/// Samplers need their own owner because descriptor bytes do not own Vulkan objects.
pub struct Sampler {
    pub(crate) state: Rc<DeviceState>,
    pub(crate) raw: vk::Sampler,
}

impl GpuHeap {
    pub fn size(&self) -> u64 {
        self.owner.size
    }
    pub fn device_address(&self) -> u64 {
        self.owner.address
    }
    pub fn range(&self, offset: u64, size: u64) -> Result<GpuRange<'_>, GpuError> {
        if size == 0 || offset > self.size() || size > self.size() - offset {
            return Err(GpuError::InvalidArgument(
                "buffer range is empty or out of bounds",
            ));
        }
        Ok(GpuRange {
            heap: self,
            offset,
            size,
        })
    }
}
impl GpuRange<'_> {
    pub fn device_address(&self) -> u64 {
        self.heap.device_address() + self.offset
    }
    pub fn size(&self) -> u64 {
        self.size
    }
}
impl Texture {
    pub fn description(&self) -> &TextureDesc {
        &self.desc
    }
}
impl RenderView {
    pub fn texture(&self) -> &Texture {
        &self.texture
    }
    pub fn extent(&self) -> U32x2 {
        self.extent
    }
}

/// Allocate buffer memory. Descriptor heaps use their dedicated creation functions.
pub fn create_gpu_heap(
    device: &GpuDevice,
    size: u64,
    memory: MemoryType,
) -> Result<GpuHeap, GpuError> {
    if matches!(
        memory,
        MemoryType::TextureDescriptorHeap | MemoryType::SamplerDescriptorHeap
    ) {
        return Err(GpuError::InvalidArgument(
            "use create_texture_descriptor_heap or create_sampler_descriptor_heap",
        ));
    }
    allocate_buffer(&device.state, size, memory, None)
}

fn allocate_buffer(
    state: &Rc<DeviceState>,
    size: u64,
    memory: MemoryType,
    set: Option<usize>,
) -> Result<GpuHeap, GpuError> {
    if size == 0 || size > isize::MAX as u64 {
        return Err(GpuError::InvalidArgument(
            "buffer size must fit a nonempty Rust slice",
        ));
    }
    let mut usage = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST;
    if let Some(index) = set {
        usage |= if index == 2 {
            vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
        } else {
            vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
        };
    } else {
        usage |= vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER;
    }
    let mut allocation_info = vk_mem::AllocationCreateInfo {
        usage: vk_mem::MemoryUsage::AutoPreferDevice,
        ..Default::default()
    };
    if memory != MemoryType::GpuOnly {
        allocation_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        allocation_info.flags = vk_mem::AllocationCreateFlags::MAPPED;
        allocation_info.flags |= if memory == MemoryType::Readback {
            vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
        } else {
            vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        };
    }
    let alignment = if set.is_some() {
        state.caps.descriptor_buffer_alignment.max(16)
    } else {
        16
    };
    unsafe {
        let (buffer, allocation) = state.allocator().create_buffer_with_alignment(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            &allocation_info,
            alignment,
        )?;
        let info = state.allocator().get_allocation_info(&allocation);
        let address = state
            .raw()
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        Ok(GpuHeap {
            owner: Box::new(GpuHeapOwner {
                state: state.clone(),
                buffer,
                allocation,
                address,
                size,
                mapped: info.mapped_data.cast(),
                descriptor_set: set,
            }),
        })
    }
}

/// Set 0 contains sampled images, set 1 storage images. Binding 0 is an array.
pub fn create_texture_descriptor_heap(
    device: &GpuDevice,
    kind: TextureDescriptorType,
) -> Result<GpuHeap, GpuError> {
    let set = match kind {
        TextureDescriptorType::Sampled => 0,
        TextureDescriptorType::Storage => 1,
    };
    allocate_buffer(
        &device.state,
        device.state.heap_sizes[set],
        MemoryType::TextureDescriptorHeap,
        Some(set),
    )
}
/// Set 2, binding 0 contains the sampler array.
pub fn create_sampler_descriptor_heap(device: &GpuDevice) -> Result<GpuHeap, GpuError> {
    allocate_buffer(
        &device.state,
        device.state.heap_sizes[2],
        MemoryType::SamplerDescriptorHeap,
        Some(2),
    )
}

/// Access a host-visible mapping. Call `flush_memory` after writes, and
/// `invalidate_memory` after GPU completion before reads on noncoherent memory.
/// # Safety
/// No GPU access or other host mapping may overlap this range during the borrow.
pub unsafe fn mapped_memory(
    heap: &mut GpuHeap,
    offset: u64,
    size: usize,
) -> Result<GpuCpuRange<'_>, GpuError> {
    heap.range(offset, size as u64)?;
    if heap.owner.mapped.is_null() {
        return Err(GpuError::InvalidArgument("heap is not CPU visible"));
    }
    let cpu =
        unsafe { std::slice::from_raw_parts_mut(heap.owner.mapped.add(offset as usize), size) };
    Ok(GpuCpuRange {
        gpu: heap.device_address() + offset,
        cpu,
    })
}
pub fn flush_memory(range: GpuRange<'_>) -> Result<(), GpuError> {
    if range.heap.owner.mapped.is_null() {
        return Err(GpuError::InvalidArgument("heap is not mapped"));
    }
    range.heap.owner.state.allocator().flush_allocation(
        &range.heap.owner.allocation,
        range.offset,
        range.size,
    )?;
    Ok(())
}
/// # Safety
/// GPU writes to the range must have completed. Host accesses must be synchronized.
pub unsafe fn invalidate_memory(range: GpuRange<'_>) -> Result<(), GpuError> {
    if range.heap.owner.mapped.is_null() {
        return Err(GpuError::InvalidArgument("heap is not mapped"));
    }
    range.heap.owner.state.allocator().invalidate_allocation(
        &range.heap.owner.allocation,
        range.offset,
        range.size,
    )?;
    Ok(())
}
/// Upload and flush bytes.
/// # Safety
/// The destination must not be in use by the GPU.
pub unsafe fn write_memory(heap: &mut GpuHeap, offset: u64, bytes: &[u8]) -> Result<(), GpuError> {
    unsafe {
        mapped_memory(heap, offset, bytes.len())?
            .cpu
            .copy_from_slice(bytes);
    }
    flush_memory(heap.range(offset, bytes.len() as u64)?)
}

pub(crate) fn image_usage(usage: TextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    for (source, target) in [
        (TextureUsage::SAMPLED, vk::ImageUsageFlags::SAMPLED),
        (TextureUsage::STORAGE, vk::ImageUsageFlags::STORAGE),
        (
            TextureUsage::COLOR_ATTACHMENT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ),
        (
            TextureUsage::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        ),
        (
            TextureUsage::TRANSFER_SOURCE,
            vk::ImageUsageFlags::TRANSFER_SRC,
        ),
        (
            TextureUsage::TRANSFER_DESTINATION,
            vk::ImageUsageFlags::TRANSFER_DST,
        ),
    ] {
        if usage.contains(source) {
            flags |= target;
        }
    }
    flags
}

/// Test optimal-tiled 2D support for the intended usage. Full dimensions, mip
/// counts, cube flags and image type are checked again by `create_texture`.
pub fn supports_texture_format(device: &GpuDevice, format: Format, usage: TextureUsage) -> bool {
    if format == Format::Undefined || usage == TextureUsage::NONE {
        return false;
    }
    unsafe {
        device
            .state
            .instance
            .get_physical_device_image_format_properties(
                device.state.physical,
                format.into(),
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                image_usage(usage),
                vk::ImageCreateFlags::empty(),
            )
            .is_ok()
    }
}

/// Allocate an optimal-tiled image in UNDEFINED layout. Use `barrier` before use.
pub fn create_texture(device: &GpuDevice, desc: &TextureDesc) -> Result<Texture, GpuError> {
    let state = &device.state;
    let (image_type, flags) = validate_texture_desc(desc)?;
    let extent = vk::Extent3D {
        width: desc.extent.x,
        height: desc.extent.y,
        depth: desc.extent.z,
    };
    unsafe {
        let supported = match state.instance.get_physical_device_image_format_properties(
            state.physical,
            desc.format.into(),
            image_type,
            vk::ImageTiling::OPTIMAL,
            image_usage(desc.usage),
            flags,
        ) {
            Ok(properties) => properties,
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
                return Err(GpuError::Unsupported(format!(
                    "{}: {:?} with {:?}",
                    state.caps.device_name.to_string_lossy(),
                    desc.format,
                    desc.usage
                )));
            }
            Err(error) => return Err(error.into()),
        };
        if extent.width > supported.max_extent.width
            || extent.height > supported.max_extent.height
            || extent.depth > supported.max_extent.depth
            || desc.mip_levels > supported.max_mip_levels
            || desc.layer_count > supported.max_array_layers
        {
            return Err(GpuError::Unsupported(
                "texture extent, mip count or array size exceeds device limits".into(),
            ));
        }
        let (image, allocation) = state.allocator().create_image(
            &vk::ImageCreateInfo::default()
                .flags(flags)
                .image_type(image_type)
                .format(desc.format.into())
                .extent(extent)
                .mip_levels(desc.mip_levels)
                .array_layers(desc.layer_count)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(image_usage(desc.usage))
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            &vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::AutoPreferDevice,
                ..Default::default()
            },
        )?;
        Ok(Texture {
            owner: Rc::new(TextureHeapOwner {
                state: state.clone(),
                allocation: Some((image, allocation)),
                swapchain: vk::SwapchainKHR::null(),
            }),
            image,
            desc: desc.clone(),
        })
    }
}

pub(crate) fn validate_texture_desc(
    desc: &TextureDesc,
) -> Result<(vk::ImageType, vk::ImageCreateFlags), GpuError> {
    let extent = desc.extent;
    if extent.x == 0
        || extent.y == 0
        || extent.z == 0
        || desc.mip_levels == 0
        || desc.layer_count == 0
        || desc.format == Format::Undefined
        || desc.usage == TextureUsage::NONE
    {
        return Err(GpuError::InvalidArgument(
            "texture dimensions, levels, layers, format and usage must be nonempty",
        ));
    }
    let mut flags = vk::ImageCreateFlags::empty();
    if desc.mutable_format {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    let image_type = match desc.texture_type {
        TextureType::OneD => {
            if extent.y != 1 || extent.z != 1 || desc.layer_count != 1 {
                return Err(GpuError::InvalidArgument(
                    "1D texture needs height/depth/layers = 1",
                ));
            }
            vk::ImageType::TYPE_1D
        }
        TextureType::ThreeD => {
            if desc.layer_count != 1 {
                return Err(GpuError::InvalidArgument(
                    "3D textures cannot have array layers",
                ));
            }
            vk::ImageType::TYPE_3D
        }
        _ => {
            if extent.z != 1 {
                return Err(GpuError::InvalidArgument("2D/cube texture depth must be 1"));
            }
            if desc.texture_type == TextureType::TwoD && desc.layer_count != 1 {
                return Err(GpuError::InvalidArgument("use TwoDArray for array layers"));
            }
            if matches!(
                desc.texture_type,
                TextureType::Cube | TextureType::CubeArray
            ) {
                if extent.x != extent.y
                    || !desc.layer_count.is_multiple_of(6)
                    || (desc.texture_type == TextureType::Cube && desc.layer_count != 6)
                {
                    return Err(GpuError::InvalidArgument(
                        "cube textures must be square with six layers per cube",
                    ));
                }
                flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
            }
            vk::ImageType::TYPE_2D
        }
    };
    if desc.mip_levels > 32 - extent.x.max(extent.y).max(extent.z).leading_zeros() {
        return Err(GpuError::InvalidArgument("too many mip levels"));
    }
    Ok((image_type, flags))
}

/// Obtain a shared owner of a texture's VMA allocation (not a second allocation).
pub fn get_texture_heap(texture: &Texture) -> TextureHeap {
    let size = match &texture.owner.allocation {
        Some((_, allocation)) => {
            texture
                .owner
                .state
                .allocator()
                .get_allocation_info(allocation)
                .size
        }
        None => 0,
    };
    TextureHeap {
        size,
        _owner: texture.owner.clone(),
    }
}

/// Single-mip, single-layer attachment view. 3D slice attachments are not enabled.
pub fn create_render_view(
    texture: &Texture,
    desc: &RenderViewDesc,
) -> Result<RenderView, GpuError> {
    if matches!(
        texture.desc.texture_type,
        TextureType::OneD | TextureType::ThreeD
    ) {
        return Err(GpuError::Unsupported(
            "attachment views require 2D textures or cube/array slices".into(),
        ));
    }
    create_texture_view(
        texture,
        &TextureDescriptorDesc {
            base_mip: desc.mip_level,
            mip_count: 1,
            base_layer: desc.slice,
            layer_count: 1,
            ..TextureDescriptorDesc::default()
        },
    )
}

/// View for sampled/storage descriptors. Zero counts mean all remaining levels/layers.
/// Format reinterpretation must belong to Vulkan's image-format compatibility class.
/// # Safety
/// When overriding the texture format, the two formats must be Vulkan-compatible.
pub unsafe fn create_reinterpreted_texture_view(
    texture: &Texture,
    desc: &TextureDescriptorDesc,
) -> Result<RenderView, GpuError> {
    create_view(texture, desc)
}
pub fn create_texture_view(
    texture: &Texture,
    desc: &TextureDescriptorDesc,
) -> Result<RenderView, GpuError> {
    if desc.format != Format::Undefined && desc.format != texture.desc.format {
        return Err(GpuError::InvalidArgument(
            "use create_reinterpreted_texture_view for format reinterpretation",
        ));
    }
    create_view(texture, desc)
}

fn create_view(texture: &Texture, desc: &TextureDescriptorDesc) -> Result<RenderView, GpuError> {
    let source = &texture.desc;
    if !source.usage.intersects(
        TextureUsage::SAMPLED
            | TextureUsage::STORAGE
            | TextureUsage::COLOR_ATTACHMENT
            | TextureUsage::DEPTH_STENCIL_ATTACHMENT,
    ) {
        return Err(GpuError::InvalidArgument(
            "transfer-only textures cannot have image views",
        ));
    }
    if desc.base_mip >= source.mip_levels || desc.base_layer >= source.layer_count {
        return Err(GpuError::InvalidArgument(
            "view base subresource out of bounds",
        ));
    }
    let mip_count = if desc.mip_count == 0 {
        source.mip_levels - desc.base_mip
    } else {
        desc.mip_count
    };
    let layer_count = if desc.layer_count == 0 {
        source.layer_count - desc.base_layer
    } else {
        desc.layer_count
    };
    if mip_count > source.mip_levels - desc.base_mip
        || layer_count > source.layer_count - desc.base_layer
    {
        return Err(GpuError::InvalidArgument(
            "view subresource range out of bounds",
        ));
    }
    let format = if desc.format == Format::Undefined {
        source.format
    } else {
        desc.format
    };
    if format != source.format && !source.mutable_format {
        return Err(GpuError::InvalidArgument(
            "texture was not created with mutable_format",
        ));
    }
    let info = TextureFormatInfo::get_texture_format_info(format);
    let aspect = match desc.aspect {
        TextureAspect::Automatic => {
            let mut mask = vk::ImageAspectFlags::empty();
            if info.depth {
                mask |= vk::ImageAspectFlags::DEPTH;
            }
            if info.stencil {
                mask |= vk::ImageAspectFlags::STENCIL;
            }
            if mask.is_empty() {
                mask = vk::ImageAspectFlags::COLOR;
            }
            mask
        }
        TextureAspect::Color if !info.depth && !info.stencil => vk::ImageAspectFlags::COLOR,
        TextureAspect::Depth if info.depth => vk::ImageAspectFlags::DEPTH,
        TextureAspect::Stencil if info.stencil => vk::ImageAspectFlags::STENCIL,
        _ => {
            return Err(GpuError::InvalidArgument(
                "view aspect is incompatible with texture format",
            ));
        }
    };
    let view_type = match source.texture_type {
        TextureType::OneD => vk::ImageViewType::TYPE_1D,
        TextureType::ThreeD => vk::ImageViewType::TYPE_3D,
        TextureType::Cube | TextureType::CubeArray
            if layer_count >= 6
                && layer_count.is_multiple_of(6)
                && desc.base_layer.is_multiple_of(6) =>
        {
            if layer_count == 6 {
                vk::ImageViewType::CUBE
            } else {
                vk::ImageViewType::CUBE_ARRAY
            }
        }
        _ if layer_count > 1 => vk::ImageViewType::TYPE_2D_ARRAY,
        _ => vk::ImageViewType::TYPE_2D,
    };
    let subresources = vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .base_mip_level(desc.base_mip)
        .level_count(mip_count)
        .base_array_layer(desc.base_layer)
        .layer_count(layer_count);
    let raw = unsafe {
        texture.owner.state.raw().create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(texture.image)
                .view_type(view_type)
                .format(format.into())
                .subresource_range(subresources),
            None,
        )?
    };
    Ok(RenderView {
        texture: texture.clone(),
        raw,
        aspect,
        subresources,
        view_type,
        extent: U32x2 {
            x: (source.extent.x >> desc.base_mip).max(1),
            y: (source.extent.y >> desc.base_mip).max(1),
        },
    })
}

/// Write a descriptor and flush its bytes. Sampled depth/stencil views must select
/// exactly one aspect. Sampled images use SHADER_READ_ONLY_OPTIMAL; storage uses GENERAL.
/// # Safety
/// The slot must not be in use. Keep the view alive through all GPU accesses.
pub unsafe fn write_texture_descriptor(
    heap: &mut GpuHeap,
    index: u32,
    view: &RenderView,
) -> Result<(), GpuError> {
    let Some(set @ 0..=1) = heap.owner.descriptor_set else {
        return Err(GpuError::InvalidArgument("not a texture descriptor heap"));
    };
    if !Rc::ptr_eq(&heap.owner.state, &view.texture.owner.state) {
        return Err(GpuError::InvalidArgument(
            "resources belong to different devices",
        ));
    }
    if index >= heap.owner.state.caps.descriptor_count {
        return Err(GpuError::InvalidArgument("descriptor index out of bounds"));
    }
    let usage = if set == 0 {
        TextureUsage::SAMPLED
    } else {
        TextureUsage::STORAGE
    };
    if !view.texture.desc.usage.contains(usage) || view.aspect.as_raw().count_ones() != 1 {
        return Err(GpuError::InvalidArgument(
            "descriptor view usage/aspect is incompatible",
        ));
    }
    if set == 1 && view.subresources.level_count != 1 {
        return Err(GpuError::InvalidArgument(
            "storage image views need exactly one mip level",
        ));
    }
    let image = vk::DescriptorImageInfo::default()
        .image_view(view.raw)
        .image_layout(if set == 0 {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        });
    let data = if set == 0 {
        vk::DescriptorDataEXT {
            p_sampled_image: &image,
        }
    } else {
        vk::DescriptorDataEXT {
            p_storage_image: &image,
        }
    };
    let descriptor_type = if set == 0 {
        vk::DescriptorType::SAMPLED_IMAGE
    } else {
        vk::DescriptorType::STORAGE_IMAGE
    };
    let size = heap.owner.state.descriptor_sizes[set];
    let offset = heap.owner.state.descriptor_offsets[set] + u64::from(index) * size;
    unsafe {
        let bytes =
            std::slice::from_raw_parts_mut(heap.owner.mapped.add(offset as usize), size as usize);
        heap.owner.state.descriptors().get_descriptor(
            &vk::DescriptorGetInfoEXT::default()
                .ty(descriptor_type)
                .data(data),
            bytes,
        );
    }
    flush_memory(heap.range(offset, size)?)
}

pub fn create_sampler(device: &GpuDevice, desc: &SamplerDesc) -> Result<Sampler, GpuError> {
    if desc.anisotropic && !device.state.caps.sampler_anisotropy {
        return Err(GpuError::Unsupported("samplerAnisotropy".into()));
    }
    let filters = [desc.min_filter, desc.mag_filter];
    let mut converted = [vk::Filter::NEAREST; 2];
    for (index, filter) in filters.into_iter().enumerate() {
        converted[index] = match filter {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        };
    }
    let mut addresses = [vk::SamplerAddressMode::REPEAT; 3];
    for (index, address) in [desc.address_u, desc.address_v, desc.address_w]
        .into_iter()
        .enumerate()
    {
        addresses[index] = match address {
            AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            AddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        };
    }
    let raw = unsafe {
        device.state.raw().create_sampler(
            &vk::SamplerCreateInfo::default()
                .min_filter(converted[0])
                .mag_filter(converted[1])
                .mipmap_mode(match desc.mip_filter {
                    Filter::Nearest => vk::SamplerMipmapMode::NEAREST,
                    Filter::Linear => vk::SamplerMipmapMode::LINEAR,
                })
                .address_mode_u(addresses[0])
                .address_mode_v(addresses[1])
                .address_mode_w(addresses[2])
                .anisotropy_enable(desc.anisotropic)
                .max_anisotropy(device.state.caps.max_sampler_anisotropy)
                .compare_enable(desc.compare_enabled)
                .compare_op(vk::CompareOp::from_raw(desc.compare as i32))
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )?
    };
    Ok(Sampler {
        state: device.state.clone(),
        raw,
    })
}

/// # Safety
/// The slot must not be in use. Keep the sampler alive through all GPU accesses.
pub unsafe fn write_sampler_descriptor(
    heap: &mut GpuHeap,
    index: u32,
    sampler: &Sampler,
) -> Result<(), GpuError> {
    if heap.owner.descriptor_set != Some(2)
        || index >= heap.owner.state.caps.descriptor_count
        || !Rc::ptr_eq(&heap.owner.state, &sampler.state)
    {
        return Err(GpuError::InvalidArgument(
            "sampler descriptor heap, index or device mismatch",
        ));
    }
    let size = heap.owner.state.descriptor_sizes[2];
    let offset = heap.owner.state.descriptor_offsets[2] + u64::from(index) * size;
    unsafe {
        let bytes =
            std::slice::from_raw_parts_mut(heap.owner.mapped.add(offset as usize), size as usize);
        heap.owner.state.descriptors().get_descriptor(
            &vk::DescriptorGetInfoEXT::default()
                .ty(vk::DescriptorType::SAMPLER)
                .data(vk::DescriptorDataEXT {
                    p_sampler: &sampler.raw,
                }),
            bytes,
        );
    }
    flush_memory(heap.range(offset, size)?)
}

pub fn destroy_gpu_heap(heap: GpuHeap) {
    drop(heap);
}
pub fn destroy_texture(texture: Texture) {
    drop(texture);
}
pub fn destroy_texture_heap(heap: TextureHeap) {
    drop(heap);
}
pub fn destroy_render_view(view: RenderView) {
    drop(view);
}
pub fn destroy_sampler(sampler: Sampler) {
    drop(sampler);
}

impl Drop for GpuHeapOwner {
    fn drop(&mut self) {
        unsafe {
            self.state
                .allocator()
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}
impl Drop for TextureHeapOwner {
    fn drop(&mut self) {
        unsafe {
            if let Some((image, mut allocation)) = self.allocation.take() {
                self.state.allocator().destroy_image(image, &mut allocation);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                let _ = self.state.raw().device_wait_idle();
                self.state
                    .swapchain_api
                    .as_ref()
                    .expect("windowed device")
                    .destroy_swapchain(self.swapchain, None);
            }
        }
    }
}
impl Drop for RenderView {
    fn drop(&mut self) {
        unsafe {
            self.texture
                .owner
                .state
                .raw()
                .destroy_image_view(self.raw, None);
        }
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.state.raw().destroy_sampler(self.raw, None);
        }
    }
}
