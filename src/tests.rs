use crate::{
    command::texture_copy_region, device::required_features, resource::validate_texture_desc, *,
};

#[test]
fn rejection_lists_all_missing_requirements() {
    let missing = required_features(
        vk::API_VERSION_1_3,
        &vk::PhysicalDeviceFeatures::default(),
        &vk::PhysicalDeviceVulkan12Features::default(),
        &vk::PhysicalDeviceVulkan13Features::default(),
        false,
    );
    for expected in [
        "Vulkan 1.4",
        "bufferDeviceAddress",
        "timelineSemaphore",
        "dynamicRendering",
        "synchronization2",
        "VK_EXT_descriptor_buffer / descriptorBuffer",
        "runtimeDescriptorArray",
        "shaderStorageImageArrayNonUniformIndexing",
    ] {
        assert!(
            missing.contains(&expected),
            "missing diagnostic: {expected}"
        );
    }
    assert_eq!(GpuError::from(vk::Result::TIMEOUT), GpuError::Timeout);
    assert_eq!(
        GpuError::from(vk::Result::ERROR_DEVICE_LOST),
        GpuError::DeviceLost
    );
    assert_eq!(
        GpuError::from(vk::Result::ERROR_OUT_OF_HOST_MEMORY),
        GpuError::DriverError(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
    );
}

#[test]
fn pitched_array_copy_uses_last_texel_not_full_padding() {
    let texture = TextureDesc {
        texture_type: TextureType::TwoDArray,
        extent: U32x3 { x: 3, y: 2, z: 1 },
        layer_count: 2,
        ..TextureDesc::default()
    };
    let copy = TextureCopyDesc {
        row_pitch_bytes: 16,
        slice_pitch_bytes: 48,
        ..TextureCopyDesc::default()
    };
    // 48 bytes to layer 1, 16 to its final row, then 12 bytes of actual texels.
    let region = texture_copy_region(&texture, &copy, TextureAspect::Automatic, 0, 76).unwrap();
    assert_eq!(region.buffer_row_length, 4);
    assert_eq!(region.buffer_image_height, 3);
    assert_eq!(region.image_subresource.layer_count, 2);
    assert!(texture_copy_region(&texture, &copy, TextureAspect::Automatic, 0, 75).is_err());
}

#[test]
fn compressed_copy_accepts_partial_edge_blocks_only() {
    let texture = TextureDesc {
        format: Format::Bc7Unorm,
        extent: U32x3 { x: 7, y: 5, z: 1 },
        ..TextureDesc::default()
    };
    assert!(
        texture_copy_region(
            &texture,
            &TextureCopyDesc::default(),
            TextureAspect::Automatic,
            0,
            64
        )
        .is_ok()
    );
    let partial = TextureCopyDesc {
        extent: U32x3 { x: 3, y: 4, z: 1 },
        ..TextureCopyDesc::default()
    };
    assert!(texture_copy_region(&texture, &partial, TextureAspect::Automatic, 0, 64).is_err());
    let edge = TextureCopyDesc {
        offset: U32x3 { x: 4, y: 4, z: 0 },
        ..TextureCopyDesc::default()
    };
    let region = texture_copy_region(&texture, &edge, TextureAspect::Automatic, 16, 16).unwrap();
    assert_eq!(region.image_extent.width, 3);
    assert_eq!(region.image_extent.height, 1);
    assert!(texture_copy_region(&texture, &edge, TextureAspect::Automatic, 4, 16).is_err());
}

#[test]
fn depth_stencil_copy_uses_aspect_specific_texel_size() {
    let texture = TextureDesc {
        format: Format::D32FloatS8Uint,
        extent: U32x3 { x: 4, y: 1, z: 1 },
        ..TextureDesc::default()
    };
    let copy = TextureCopyDesc::default();
    assert!(texture_copy_region(&texture, &copy, TextureAspect::Automatic, 0, 32).is_err());
    assert!(texture_copy_region(&texture, &copy, TextureAspect::Depth, 0, 16).is_ok());
    assert!(texture_copy_region(&texture, &copy, TextureAspect::Depth, 0, 15).is_err());
    assert!(texture_copy_region(&texture, &copy, TextureAspect::Stencil, 0, 4).is_ok());
    assert!(texture_copy_region(&texture, &copy, TextureAspect::Stencil, 0, 3).is_err());
}

#[test]
fn copy_rejects_pitch_overflow_and_invalid_subresources() {
    let texture = TextureDesc {
        extent: U32x3 { x: 8, y: 8, z: 1 },
        mip_levels: 4,
        ..TextureDesc::default()
    };
    let copies = [
        TextureCopyDesc {
            row_pitch_bytes: u64::MAX - 3,
            ..TextureCopyDesc::default()
        },
        TextureCopyDesc {
            row_pitch_bytes: 31,
            ..TextureCopyDesc::default()
        },
        TextureCopyDesc {
            slice_pitch_bytes: 255,
            ..TextureCopyDesc::default()
        },
        TextureCopyDesc {
            mip_level: 4,
            ..TextureCopyDesc::default()
        },
        TextureCopyDesc {
            base_slice: 1,
            ..TextureCopyDesc::default()
        },
        TextureCopyDesc {
            offset: U32x3 { x: 8, y: 0, z: 0 },
            ..TextureCopyDesc::default()
        },
    ];
    for copy in copies {
        assert!(
            texture_copy_region(&texture, &copy, TextureAspect::Automatic, 0, u64::MAX).is_err()
        );
    }
    let last_mip = TextureCopyDesc {
        mip_level: 3,
        ..TextureCopyDesc::default()
    };
    assert_eq!(
        texture_copy_region(&texture, &last_mip, TextureAspect::Automatic, 0, 4)
            .unwrap()
            .image_extent
            .width,
        1
    );
}

#[test]
fn texture_shape_and_mips_are_validated_before_driver_calls() {
    let mut desc = TextureDesc {
        texture_type: TextureType::Cube,
        extent: U32x3 { x: 16, y: 16, z: 1 },
        layer_count: 6,
        mip_levels: 5,
        ..TextureDesc::default()
    };
    assert!(
        validate_texture_desc(&desc)
            .unwrap()
            .1
            .contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
    );
    desc.mip_levels = 6;
    assert!(validate_texture_desc(&desc).is_err());
    desc.mip_levels = 1;
    desc.layer_count = 7;
    assert!(validate_texture_desc(&desc).is_err());
    desc.texture_type = TextureType::ThreeD;
    assert!(validate_texture_desc(&desc).is_err());
    desc.layer_count = 1;
    assert!(validate_texture_desc(&desc).is_ok());
}

#[test]
fn synchronization_masks_are_not_raw_api_enum_values() {
    assert_eq!(
        vk::PipelineStageFlags2::from(Stage::TRANSFER | Stage::COMPUTE),
        vk::PipelineStageFlags2::ALL_TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
    );
    assert_eq!(
        vk::PipelineStageFlags2::from(Stage::DEPTH_STENCIL_TESTS),
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
    );
    assert_eq!(
        vk::AccessFlags2::from(Access::DESCRIPTOR_READ | Access::HOST_WRITE),
        vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT | vk::AccessFlags2::HOST_WRITE
    );
    assert_eq!(
        vk::ImageLayout::from(TextureLayout::Present),
        vk::ImageLayout::PRESENT_SRC_KHR
    );
}
