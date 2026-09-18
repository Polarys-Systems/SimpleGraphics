//! A small, explicit Vulkan 1.4 graphics API.
//!
//! Create an application-owned `winit` window (or a headless device), allocate
//! resources, record commands, then submit them. Descriptor sets 0, 1 and 2 are
//! sampled-image, storage-image and sampler arrays, respectively. Buffer access
//! uses device addresses; shaders use the `main` entry point.
//!
//! Resource owners keep the Vulkan device alive. Recording and submission are
//! deliberately low-level: their `unsafe` contracts require correct shader use,
//! synchronization and resource lifetimes until GPU completion. Dropping an
//! unsubmitted command is allowed. Dropping a submitted command waits for its
//! fence; normal reuse polls/waits only that command, never the entire device.
//! See the repository README for the shader and frame contracts.
#![doc = include_str!("../README.md")]

pub use ash::vk;
pub use harfrust;
/// Required loader/device version. The stable ash 0.38 bindings expose all core
/// commands used here; Vulkan 1.4 preserves those commands and feature structures.
pub const VULKAN_API_VERSION: u32 = vk::make_api_version(0, 1, 4, 0);
use std::{ffi::CString, sync::Arc};
use winit::window::Window;

mod command;
mod device;
mod pipeline;
mod resource;
#[cfg(test)]
mod tests;
pub use command::*;
pub use device::*;
pub use pipeline::*;
pub use resource::*;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct U32x2 {
    pub x: u32,
    pub y: u32,
}

impl U32x2 {
    pub fn compare(a: U32x2, b: U32x2) -> bool {
        a.x == b.x && a.y == b.y
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct U32x3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl U32x3 {
    pub fn compare(a: U32x3, b: U32x3) -> bool {
        a.x == b.x && a.y == b.y && a.z == b.z
    }
}

/// Recoverable driver failures and actionable unsupported-feature diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    DeviceLost,
    Unsupported(String),
    InvalidArgument(&'static str),
    DriverError(vk::Result),
    Loader(String),
    Timeout,
    OutOfDate,
    SurfaceLost,
    WindowMinimized,
}

impl From<vk::Result> for GpuError {
    fn from(result: vk::Result) -> Self {
        match result {
            vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost,
            vk::Result::TIMEOUT | vk::Result::NOT_READY => Self::Timeout,
            vk::Result::ERROR_OUT_OF_DATE_KHR => Self::OutOfDate,
            vk::Result::ERROR_SURFACE_LOST_KHR => Self::SurfaceLost,
            _ => Self::DriverError(result),
        }
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::Loader(message) => write!(f, "Vulkan loader: {message}"),
            Self::DriverError(result) => write!(f, "Vulkan: {result}"),
            _ => write!(f, "{self:?}"),
        }
    }
}
impl std::error::Error for GpuError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    R8Srgb,
    Rg8Srgb,
    Rgba8Srgb,
    Bgra8Srgb,

    Rgba4Unorm,
    R5g5b5a1Unorm,
    R5g6b5Unorm,

    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,

    R16Unorm,
    Rg16Unorm,
    Rgba16Unorm,

    R8Uint,
    Rg8Uint,
    Rgba8Uint,
    Bgra8Uint,

    R16Uint,
    Rg16Uint,
    Rgba16Uint,

    R32Uint,
    Rg32Uint,
    Rgb32Uint,
    Rgba32Uint,

    R16Float,
    Rg16Float,
    Rgba16Float,

    R32Float,
    Rg32Float,
    Rgb32Float,
    Rgba32Float,

    Rgb10a2Unorm,
    Rg11b10Float,

    D16Unorm,
    D24UnormS8Uint,
    D32Float,
    S8Uint,
    D32FloatS8Uint,

    EacRg,
    Astc4x4Srgb,
    Astc4x4Unorm,

    Bc3Srgb,
    Bc3Unorm,
    Bc5Rg,
    Bc7Srgb,
    Bc7Unorm,

    Undefined,
}

impl From<Format> for vk::Format {
    fn from(format: Format) -> vk::Format {
        match format {
            Format::R8Srgb => vk::Format::R8_SRGB,
            Format::Rg8Srgb => vk::Format::R8G8_SRGB,
            Format::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            Format::Bgra8Srgb => vk::Format::B8G8R8A8_SRGB,

            Format::Rgba4Unorm => vk::Format::R4G4B4A4_UNORM_PACK16,
            Format::R5g5b5a1Unorm => vk::Format::R5G5B5A1_UNORM_PACK16,
            Format::R5g6b5Unorm => vk::Format::R5G6B5_UNORM_PACK16,

            Format::R8Unorm => vk::Format::R8_UNORM,
            Format::Rg8Unorm => vk::Format::R8G8_UNORM,
            Format::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            Format::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,

            Format::R16Unorm => vk::Format::R16_UNORM,
            Format::Rg16Unorm => vk::Format::R16G16_UNORM,
            Format::Rgba16Unorm => vk::Format::R16G16B16A16_UNORM,

            Format::R8Uint => vk::Format::R8_UINT,
            Format::Rg8Uint => vk::Format::R8G8_UINT,
            Format::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            Format::Bgra8Uint => vk::Format::B8G8R8A8_UINT,

            Format::R16Uint => vk::Format::R16_UINT,
            Format::Rg16Uint => vk::Format::R16G16_UINT,
            Format::Rgba16Uint => vk::Format::R16G16B16A16_UINT,

            Format::R32Uint => vk::Format::R32_UINT,
            Format::Rg32Uint => vk::Format::R32G32_UINT,
            Format::Rgb32Uint => vk::Format::R32G32B32_UINT,
            Format::Rgba32Uint => vk::Format::R32G32B32A32_UINT,

            Format::R16Float => vk::Format::R16_SFLOAT,
            Format::Rg16Float => vk::Format::R16G16_SFLOAT,
            Format::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,

            Format::R32Float => vk::Format::R32_SFLOAT,
            Format::Rg32Float => vk::Format::R32G32_SFLOAT,
            Format::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
            Format::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,

            Format::Rgb10a2Unorm => vk::Format::A2B10G10R10_UNORM_PACK32,
            Format::Rg11b10Float => vk::Format::B10G11R11_UFLOAT_PACK32,

            Format::D16Unorm => vk::Format::D16_UNORM,
            Format::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
            Format::D32Float => vk::Format::D32_SFLOAT,
            Format::S8Uint => vk::Format::S8_UINT,
            Format::D32FloatS8Uint => vk::Format::D32_SFLOAT_S8_UINT,

            Format::EacRg => vk::Format::EAC_R11G11_UNORM_BLOCK,
            Format::Astc4x4Srgb => vk::Format::ASTC_4X4_SRGB_BLOCK,
            Format::Astc4x4Unorm => vk::Format::ASTC_4X4_UNORM_BLOCK,

            Format::Bc3Srgb => vk::Format::BC3_SRGB_BLOCK,
            Format::Bc3Unorm => vk::Format::BC3_UNORM_BLOCK,
            Format::Bc5Rg => vk::Format::BC5_UNORM_BLOCK,
            Format::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
            Format::Bc7Unorm => vk::Format::BC7_UNORM_BLOCK,

            Format::Undefined => vk::Format::UNDEFINED,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    CpuVisible,
    GpuOnly,
    Readback,
    TextureDescriptorHeap,
    SamplerDescriptorHeap,
}

#[derive(Debug, Clone, Copy)]
pub struct SizeAlign {
    pub size: u64,
    pub align: u64,
}

pub struct TextureFormatInfo {
    pub block_extent: U32x2,
    pub bytes_per_block: u32,
    pub depth: bool,
    pub stencil: bool,
}

impl TextureFormatInfo {
    pub fn get_texture_format_info(format: Format) -> Self {
        match format {
            Format::R8Srgb | Format::R8Unorm | Format::R8Uint | Format::S8Uint => Self {
                block_extent: U32x2 { x: 1, y: 1 },
                bytes_per_block: 1,
                depth: false,
                stencil: format == Format::S8Uint,
            },

            Format::Rg8Srgb
            | Format::Rgba4Unorm
            | Format::R5g5b5a1Unorm
            | Format::R5g6b5Unorm
            | Format::Rg8Unorm
            | Format::R16Unorm
            | Format::Rg8Uint
            | Format::R16Uint
            | Format::R16Float
            | Format::D16Unorm => Self {
                block_extent: U32x2 { x: 1, y: 1 },
                bytes_per_block: 2,
                depth: format == Format::D16Unorm,
                stencil: false,
            },

            Format::Rgba8Srgb
            | Format::Bgra8Srgb
            | Format::Rgba8Unorm
            | Format::Bgra8Unorm
            | Format::Rg16Unorm
            | Format::Rgba8Uint
            | Format::Bgra8Uint
            | Format::Rg16Uint
            | Format::R32Uint
            | Format::Rg16Float
            | Format::R32Float
            | Format::Rgb10a2Unorm
            | Format::Rg11b10Float
            | Format::D24UnormS8Uint
            | Format::D32Float => Self {
                block_extent: U32x2 { x: 1, y: 1 },
                bytes_per_block: 4,
                depth: matches!(format, Format::D24UnormS8Uint | Format::D32Float),
                stencil: format == Format::D24UnormS8Uint,
            },

            Format::Rgba16Unorm
            | Format::Rgba16Uint
            | Format::Rg32Uint
            | Format::Rgba16Float
            | Format::Rg32Float
            | Format::D32FloatS8Uint => Self {
                block_extent: U32x2 { x: 1, y: 1 },
                bytes_per_block: 8,
                depth: format == Format::D32FloatS8Uint,
                stencil: format == Format::D32FloatS8Uint,
            },

            Format::Rgb32Uint | Format::Rgb32Float => Self {
                block_extent: U32x2 { x: 1, y: 1 },
                bytes_per_block: 12,
                depth: false,
                stencil: false,
            },

            Format::Rgba32Uint | Format::Rgba32Float => Self {
                block_extent: U32x2 { x: 1, y: 1 },
                bytes_per_block: 16,
                depth: false,
                stencil: false,
            },

            Format::EacRg
            | Format::Astc4x4Srgb
            | Format::Astc4x4Unorm
            | Format::Bc3Srgb
            | Format::Bc3Unorm
            | Format::Bc5Rg
            | Format::Bc7Srgb
            | Format::Bc7Unorm => Self {
                block_extent: U32x2 { x: 4, y: 4 },
                bytes_per_block: 16,
                depth: false,
                stencil: false,
            },

            Format::Undefined => Self {
                block_extent: U32x2 { x: 0, y: 0 },
                bytes_per_block: 0,
                depth: false,
                stencil: false,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureType {
    OneD,
    TwoD,
    ThreeD,
    Cube,
    TwoDArray,
    CubeArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct TextureUsage(u32);

impl TextureUsage {
    pub const NONE: Self = Self(0);
    pub const SAMPLED: Self = Self(1 << 0);
    pub const STORAGE: Self = Self(1 << 1);
    pub const COLOR_ATTACHMENT: Self = Self(1 << 2);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(1 << 3);
    pub const TRANSFER_SOURCE: Self = Self(1 << 4);
    pub const TRANSFER_DESTINATION: Self = Self(1 << 5);

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureDescriptorType {
    Sampled,
    Storage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureAspect {
    Automatic,
    Color,
    Depth,
    Stencil,
}

#[derive(Debug, Clone, Copy)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CullMode {
    None,
    Clockwise,
    CounterClockwise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlendFactor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    SourceAlphaSaturate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Minimum,
    Maximum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum IndexType {
    Uint16,
    Uint32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LoadOp {
    Load,
    Clear,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Stage(u64);

impl Stage {
    pub const NONE: Self = Self(0);

    pub const INDIRECT: Self = Self(1 << 6);
    pub const INDEX_INPUT: Self = Self(1 << 7);
    pub const VERTEX: Self = Self(1 << 1);
    pub const MESH: Self = Self(1 << 9);
    pub const DEPTH_STENCIL_TESTS: Self = Self(1 << 8);
    pub const FRAGMENT: Self = Self(1 << 2);
    pub const COLOR_OUTPUT: Self = Self(1 << 4);
    pub const COMPUTE: Self = Self(1 << 3);
    pub const TRANSFER: Self = Self(1 << 0);
    pub const HOST: Self = Self(1 << 5);
    pub const ALL_COMMANDS: Self = Self(1 << 10);

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for Stage {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Stage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for Stage {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Access(u64);

impl Access {
    pub const NONE: Self = Self(0);

    pub const TRANSFER_READ: Self = Self(1 << 0);
    pub const TRANSFER_WRITE: Self = Self(1 << 1);
    pub const SHADER_READ: Self = Self(1 << 2);
    pub const SHADER_WRITE: Self = Self(1 << 3);
    pub const COLOR_READ: Self = Self(1 << 4);
    pub const COLOR_WRITE: Self = Self(1 << 5);
    pub const DEPTH_STENCIL_READ: Self = Self(1 << 6);
    pub const DEPTH_STENCIL_WRITE: Self = Self(1 << 7);
    pub const INDIRECT_READ: Self = Self(1 << 8);
    pub const INDEX_READ: Self = Self(1 << 9);
    pub const HOST_READ: Self = Self(1 << 10);
    pub const DESCRIPTOR_READ: Self = Self(1 << 11);
    pub const HOST_WRITE: Self = Self(1 << 12);

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for Access {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for Access {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct DeviceCaps {
    pub device_name: CString,
    pub api_version: u32,
    pub buffer_device_address: bool,
    pub dynamic_rendering: bool,
    pub synchronization2: bool,
    pub timeline_semaphore: bool,
    /// Maximum distance between a timeline's completed and outstanding values.
    pub max_timeline_value_difference: u64,
    pub descriptor_buffer: bool,
    pub bindless_descriptors: bool,
    pub mesh_shader: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
    pub max_color_attachments: u32,
    pub descriptor_count: u32,
    pub descriptor_buffer_alignment: u64,
    pub max_push_data_size: u64,

    /// Buffer/image granularity; individual images can need a larger alignment.
    pub texture_heap_alignment: u64,

    // Bytes per descriptor slot.
    pub texture_descriptor_size: u64,
    pub sampler_descriptor_size: u64,

    // Nanoseconds per timestamp tick.
    pub timestamp_period_ns: f32,

    // Fractional filtering precision, for conservative sampled-field bounds.
    pub sub_texel_precision_bits: u32,

    pub texture_compression_bc: bool,
    pub texture_compression_astc: bool,
    pub storage_input_output16: bool,
}

/// Create on the window's event-loop thread. `None` creates a headless device.
pub struct DeviceDesc {
    pub window: Option<Arc<Window>>,
    pub swapchain_format: Format,
    pub desired_swapchain_image_count: u32,
    pub timestamp_query_count: u32,
    /// Entries in each of the sampled-image, storage-image and sampler arrays.
    pub descriptor_count: u32,
    pub validation: bool,
}

impl Default for DeviceDesc {
    fn default() -> Self {
        Self {
            window: None,
            swapchain_format: Format::Undefined,
            desired_swapchain_image_count: 2,
            timestamp_query_count: 256,
            descriptor_count: 1024,
            validation: false,
        }
    }
}

/// An acquired image. Present it with `submit_and_present` before acquiring again.
pub struct SwapchainFrame {
    pub render_view: std::rc::Rc<RenderView>,
    pub extent: U32x2,
    pub suboptimal: bool,
    pub(crate) image_index: u32,
    pub(crate) generation: u64,
}

#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub texture_type: TextureType,
    pub extent: U32x3,
    pub mip_levels: u32,
    pub layer_count: u32,
    pub format: Format,
    pub mutable_format: bool,
    pub usage: TextureUsage,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            texture_type: TextureType::TwoD,
            extent: U32x3 { x: 1, y: 1, z: 1 },
            mip_levels: 1,
            layer_count: 1,
            format: Format::Rgba8Unorm,
            mutable_format: false,
            usage: TextureUsage::SAMPLED,
        }
    }
}

#[derive(Default)]
pub struct RenderViewDesc {
    pub mip_level: u32,
    pub slice: u32,
}

pub struct TextureDescriptorDesc {
    pub format: Format,
    pub aspect: TextureAspect,
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl Default for TextureDescriptorDesc {
    fn default() -> Self {
        Self {
            format: Format::Undefined,
            aspect: TextureAspect::Automatic,
            base_mip: 0,
            mip_count: 0,
            base_layer: 0,
            layer_count: 0,
        }
    }
}

pub struct TextureCopyDesc {
    pub mip_level: u32,
    pub base_slice: u32,
    pub slice_count: u32,
    pub offset: U32x3,
    pub extent: U32x3,
    pub row_pitch_bytes: u64,
    pub slice_pitch_bytes: u64,
}

impl Default for TextureCopyDesc {
    fn default() -> Self {
        Self {
            mip_level: 0,
            base_slice: 0,
            slice_count: 0,
            offset: U32x3 { x: 0, y: 0, z: 0 },
            extent: U32x3 { x: 0, y: 0, z: 0 },
            row_pitch_bytes: 0,
            slice_pitch_bytes: 0,
        }
    }
}

pub struct SamplerDesc {
    pub min_filter: Filter,
    pub mag_filter: Filter,
    pub mip_filter: Filter,

    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,

    pub anisotropic: bool,
    pub compare_enabled: bool,
    pub compare: CompareOp,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            min_filter: Filter::Linear,
            mag_filter: Filter::Linear,
            mip_filter: Filter::Linear,

            address_u: AddressMode::Repeat,
            address_v: AddressMode::Repeat,
            address_w: AddressMode::Repeat,

            anisotropic: false,
            compare_enabled: false,
            compare: CompareOp::LessEqual,
        }
    }
}

pub struct BlendComponentState {
    pub source: BlendFactor,
    pub destination: BlendFactor,
    pub operation: BlendOp,
}

impl Default for BlendComponentState {
    fn default() -> Self {
        Self {
            source: BlendFactor::One,
            destination: BlendFactor::Zero,
            operation: BlendOp::Add,
        }
    }
}

#[derive(Default)]
pub struct BlendState {
    pub enabled: bool,
    pub color: BlendComponentState,
    pub alpha: BlendComponentState,
}

pub struct ColorTargetDesc {
    pub format: Format,
    pub blend: BlendState,
    pub write_mask: u8,
}

impl Default for ColorTargetDesc {
    fn default() -> Self {
        Self {
            format: Format::Undefined,
            blend: BlendState::default(),
            write_mask: 0xf,
        }
    }
}

pub struct RasterizationState {
    pub cull: CullMode,
    pub depth_bias_constant: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope: f32,
}

impl Default for RasterizationState {
    fn default() -> Self {
        Self {
            cull: CullMode::None,
            depth_bias_constant: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope: 0.0,
        }
    }
}

pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

pub struct Scissor {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Default for Scissor {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        }
    }
}

pub struct StencilFaceState {
    pub compare: CompareOp,
    pub fail: StencilOp,
    pub pass: StencilOp,
    pub depth_fail: StencilOp,
    pub reference: u8,
}

impl Default for StencilFaceState {
    fn default() -> Self {
        Self {
            compare: CompareOp::Always,
            fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            reference: 0,
        }
    }
}

pub struct DepthStencilState {
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: CompareOp,

    pub stencil_test: bool,
    pub stencil_read_mask: u8,
    pub stencil_write_mask: u8,

    pub front: StencilFaceState,
    pub back: StencilFaceState,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_test: false,
            depth_write: false,
            depth_compare: CompareOp::LessEqual,

            stencil_test: false,
            stencil_read_mask: 0xff,
            stencil_write_mask: 0xff,

            front: StencilFaceState::default(),
            back: StencilFaceState::default(),
        }
    }
}

pub struct GraphicsPSODesc<'a> {
    pub vertex_spirv: &'a [u32],
    pub fragment_spirv: &'a [u32],

    pub color_targets: &'a [ColorTargetDesc],

    pub depth_format: Format,
    pub stencil_format: Format,

    pub rasterization: RasterizationState,
    pub depth_stencil: DepthStencilState,
}

impl Default for GraphicsPSODesc<'_> {
    fn default() -> Self {
        Self {
            vertex_spirv: &[],
            fragment_spirv: &[],
            color_targets: &[],
            depth_format: Format::Undefined,
            stencil_format: Format::Undefined,
            rasterization: RasterizationState::default(),
            depth_stencil: DepthStencilState::default(),
        }
    }
}

pub struct MeshPSODesc<'a> {
    pub mesh_spirv: &'a [u32],
    pub fragment_spirv: &'a [u32],

    pub color_targets: &'a [ColorTargetDesc],

    pub depth_format: Format,
    pub stencil_format: Format,

    pub rasterization: RasterizationState,
    pub depth_stencil: DepthStencilState,
}

impl Default for MeshPSODesc<'_> {
    fn default() -> Self {
        Self {
            mesh_spirv: &[],
            fragment_spirv: &[],
            color_targets: &[],
            depth_format: Format::Undefined,
            stencil_format: Format::Undefined,
            rasterization: RasterizationState::default(),
            depth_stencil: DepthStencilState::default(),
        }
    }
}

pub struct ClearColor {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl ClearColor {
    /// Clear an unsigned-integer color attachment. Preserves the integer bits in
    /// Vulkan's clear-value union; ordinary struct fields represent float clears.
    pub fn from_uint(value: [u32; 4]) -> Self {
        Self {
            x: f32::from_bits(value[0]),
            y: f32::from_bits(value[1]),
            z: f32::from_bits(value[2]),
            w: f32::from_bits(value[3]),
        }
    }
}

impl Default for ClearColor {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

pub struct ColorAttachment<'a> {
    pub render_view: Option<&'a RenderView>,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: ClearColor,
}

impl Default for ColorAttachment<'_> {
    fn default() -> Self {
        Self {
            render_view: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear: ClearColor::default(),
        }
    }
}

pub struct DepthAttachment<'a> {
    pub render_view: Option<&'a RenderView>,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: f32,
}

impl Default for DepthAttachment<'_> {
    fn default() -> Self {
        Self {
            render_view: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear: 1.0,
        }
    }
}

pub struct StencilAttachment<'a> {
    pub render_view: Option<&'a RenderView>,
    pub load: LoadOp,
    pub store: StoreOp,
    pub clear: u8,
}

impl Default for StencilAttachment<'_> {
    fn default() -> Self {
        Self {
            render_view: None,
            load: LoadOp::Load,
            store: StoreOp::Store,
            clear: 0,
        }
    }
}

#[derive(Default)]
pub struct RenderingDesc<'a> {
    pub colors: &'a [ColorAttachment<'a>],
    pub depth: DepthAttachment<'a>,
    pub stencil: StencilAttachment<'a>,
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}
impl std::ops::BitOrAssign for TextureUsage {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
