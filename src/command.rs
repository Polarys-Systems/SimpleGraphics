use crate::{device::DeviceState, *};
use std::rc::Rc;

/// Reusable command allocation with its own fence and command pool.
pub struct CommandBuffer {
    state: Rc<DeviceState>,
    raw: vk::CommandBuffer,
    pool: vk::CommandPool,
    fence: vk::Fence,
    status: CommandStatus,
    rendering: bool,
    bound: Option<(vk::PipelineBindPoint, bool)>,
    queries: vk::QueryPool,
    written_queries: Vec<bool>,
}
#[derive(PartialEq, Eq)]
enum CommandStatus {
    Initial,
    Recording,
    Executable,
    Pending,
}

pub struct TimelineSemaphore {
    state: Rc<DeviceState>,
    raw: vk::Semaphore,
}

/// Image layouts exposed by this API. Layouts are explicit and never inferred
/// from CPU recording order, which need not match GPU submission order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureLayout {
    Undefined,
    General,
    Sampled,
    ColorAttachment,
    DepthStencilAttachment,
    TransferSource,
    TransferDestination,
    Present,
}

/// Synchronization2 dependency. Queue ownership transfers are unnecessary with
/// the single queue. Texture barriers cover the entire image, including all mips.
pub enum Barrier<'a> {
    Memory {
        before: (Stage, Access),
        after: (Stage, Access),
    },
    Buffer {
        range: GpuRange<'a>,
        before: (Stage, Access),
        after: (Stage, Access),
    },
    Texture {
        texture: &'a Texture,
        before: (Stage, Access, TextureLayout),
        after: (Stage, Access, TextureLayout),
    },
}

impl From<TextureLayout> for vk::ImageLayout {
    fn from(layout: TextureLayout) -> Self {
        match layout {
            TextureLayout::Undefined => Self::UNDEFINED,
            TextureLayout::General => Self::GENERAL,
            TextureLayout::Sampled => Self::SHADER_READ_ONLY_OPTIMAL,
            TextureLayout::ColorAttachment => Self::COLOR_ATTACHMENT_OPTIMAL,
            TextureLayout::DepthStencilAttachment => Self::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            TextureLayout::TransferSource => Self::TRANSFER_SRC_OPTIMAL,
            TextureLayout::TransferDestination => Self::TRANSFER_DST_OPTIMAL,
            TextureLayout::Present => Self::PRESENT_SRC_KHR,
        }
    }
}

impl From<Stage> for vk::PipelineStageFlags2 {
    fn from(stage: Stage) -> Self {
        let mut flags = Self::empty();
        for (source, target) in [
            (Stage::INDIRECT, Self::DRAW_INDIRECT),
            (Stage::INDEX_INPUT, Self::INDEX_INPUT),
            (Stage::VERTEX, Self::VERTEX_SHADER),
            (Stage::MESH, Self::MESH_SHADER_EXT),
            (
                Stage::DEPTH_STENCIL_TESTS,
                Self::EARLY_FRAGMENT_TESTS | Self::LATE_FRAGMENT_TESTS,
            ),
            (Stage::FRAGMENT, Self::FRAGMENT_SHADER),
            (Stage::COLOR_OUTPUT, Self::COLOR_ATTACHMENT_OUTPUT),
            (Stage::COMPUTE, Self::COMPUTE_SHADER),
            (Stage::TRANSFER, Self::ALL_TRANSFER),
            (Stage::HOST, Self::HOST),
            (Stage::ALL_COMMANDS, Self::ALL_COMMANDS),
        ] {
            if stage.intersects(source) {
                flags |= target;
            }
        }
        flags
    }
}
impl From<Access> for vk::AccessFlags2 {
    fn from(access: Access) -> Self {
        let mut flags = Self::empty();
        for (source, target) in [
            (Access::TRANSFER_READ, Self::TRANSFER_READ),
            (Access::TRANSFER_WRITE, Self::TRANSFER_WRITE),
            (Access::SHADER_READ, Self::SHADER_READ),
            (Access::SHADER_WRITE, Self::SHADER_WRITE),
            (Access::COLOR_READ, Self::COLOR_ATTACHMENT_READ),
            (Access::COLOR_WRITE, Self::COLOR_ATTACHMENT_WRITE),
            (
                Access::DEPTH_STENCIL_READ,
                Self::DEPTH_STENCIL_ATTACHMENT_READ,
            ),
            (
                Access::DEPTH_STENCIL_WRITE,
                Self::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            (Access::INDIRECT_READ, Self::INDIRECT_COMMAND_READ),
            (Access::INDEX_READ, Self::INDEX_READ),
            (Access::HOST_READ, Self::HOST_READ),
            (Access::HOST_WRITE, Self::HOST_WRITE),
            (Access::DESCRIPTOR_READ, Self::DESCRIPTOR_BUFFER_READ_EXT),
        ] {
            if access.intersects(source) {
                flags |= target;
            }
        }
        flags
    }
}

pub fn create_timeline_semaphore(
    device: &GpuDevice,
    initial_value: u64,
) -> Result<TimelineSemaphore, GpuError> {
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(initial_value);
    let raw = unsafe {
        device.state.raw().create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut kind),
            None,
        )?
    };
    Ok(TimelineSemaphore {
        state: device.state.clone(),
        raw,
    })
}
pub fn destroy_timeline_semaphore(semaphore: TimelineSemaphore) {
    drop(semaphore);
}
pub fn timeline_completed_value(semaphore: &TimelineSemaphore) -> Result<u64, GpuError> {
    Ok(unsafe {
        semaphore
            .state
            .raw()
            .get_semaphore_counter_value(semaphore.raw)?
    })
}
/// Wait for a value; timeout is nanoseconds, zero polls, u64::MAX waits indefinitely.
pub fn wait_timeline(
    semaphore: &TimelineSemaphore,
    value: u64,
    timeout_ns: u64,
) -> Result<(), GpuError> {
    unsafe {
        semaphore.state.raw().wait_semaphores(
            &vk::SemaphoreWaitInfo::default()
                .semaphores(&[semaphore.raw])
                .values(&[value]),
            timeout_ns,
        )?;
    }
    Ok(())
}
impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.state.raw().destroy_semaphore(self.raw, None);
        }
    }
}

/// Allocate and begin one primary command buffer. Reuse it with `reset_command`.
pub fn begin_command(device: &GpuDevice) -> Result<CommandBuffer, GpuError> {
    let mut command = CommandBuffer {
        state: device.state.clone(),
        raw: vk::CommandBuffer::null(),
        pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
        status: CommandStatus::Initial,
        rendering: false,
        bound: None,
        queries: vk::QueryPool::null(),
        written_queries: Vec::new(),
    };
    unsafe {
        let raw = device.state.raw();
        command.pool = raw.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(device.state.queue_family)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT),
            None,
        )?;
        command.raw = raw.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        command.fence = raw.create_fence(&vk::FenceCreateInfo::default(), None)?;
        if device.state.timestamp_query_count > 0 && device.state.timestamp_valid_bits > 0 {
            command.queries = raw.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(device.state.timestamp_query_count),
                None,
            )?;
            command
                .written_queries
                .resize(device.state.timestamp_query_count as usize, false);
        }
    }
    reset_command(&mut command, 0)?;
    Ok(command)
}

/// Wait for this command's previous submission, reset its pool and begin again.
/// A timeout leaves the pending command intact. Also cancels unsubmitted recording.
pub fn reset_command(command: &mut CommandBuffer, timeout_ns: u64) -> Result<(), GpuError> {
    unsafe {
        let raw = command.state.raw();
        if command.status == CommandStatus::Pending {
            raw.wait_for_fences(&[command.fence], true, timeout_ns)?;
        }
        raw.reset_command_pool(command.pool, vk::CommandPoolResetFlags::empty())?;
        command.status = CommandStatus::Initial;
        raw.begin_command_buffer(
            command.raw,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        command.status = CommandStatus::Recording;
        command.rendering = false;
        command.bound = None;
        command.written_queries.fill(false);
        if command.queries != vk::QueryPool::null() {
            raw.cmd_reset_query_pool(
                command.raw,
                command.queries,
                0,
                command.written_queries.len() as u32,
            );
        }
    }
    Ok(())
}

/// Wait for this command's submission without resetting its recording or queries.
pub fn wait_command(command: &CommandBuffer, timeout_ns: u64) -> Result<(), GpuError> {
    if command.status == CommandStatus::Pending {
        unsafe {
            command
                .state
                .raw()
                .wait_for_fences(&[command.fence], true, timeout_ns)?;
        }
    }
    Ok(())
}
pub fn destroy_command(command: CommandBuffer) {
    drop(command);
}
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            let raw = self.state.raw();
            if self.status == CommandStatus::Pending {
                let _ = raw.wait_for_fences(&[self.fence], true, u64::MAX);
            }
            raw.destroy_query_pool(self.queries, None);
            raw.destroy_fence(self.fence, None);
            raw.destroy_command_pool(self.pool, None);
        }
    }
}

impl CommandBuffer {
    fn recording(&self, outside_rendering: bool) -> Result<(), GpuError> {
        if self.status != CommandStatus::Recording {
            return Err(GpuError::InvalidArgument(
                "command is not recording; call reset_command",
            ));
        }
        if outside_rendering && self.rendering {
            return Err(GpuError::InvalidArgument(
                "command is not allowed inside rendering",
            ));
        }
        Ok(())
    }
    fn same_device(&self, state: &Rc<DeviceState>) -> Result<(), GpuError> {
        if !Rc::ptr_eq(&self.state, state) {
            return Err(GpuError::InvalidArgument(
                "resources belong to different devices",
            ));
        }
        Ok(())
    }
}

fn prepare_submit(command: &mut CommandBuffer) -> Result<(), GpuError> {
    if command.rendering {
        return Err(GpuError::InvalidArgument(
            "end_render_pass before submitting",
        ));
    }
    if command.status == CommandStatus::Recording {
        unsafe {
            command.state.raw().end_command_buffer(command.raw)?;
        }
        command.status = CommandStatus::Executable;
    }
    if command.status != CommandStatus::Executable {
        return Err(GpuError::InvalidArgument(
            "command must be recorded before submission",
        ));
    }
    unsafe {
        command.state.raw().reset_fences(&[command.fence])?;
    }
    Ok(())
}

// Shared by headless and presentation submissions. Tuples avoid public submit-info types.
fn semaphore_infos<'a>(
    state: &Rc<DeviceState>,
    entries: &[(&'a TimelineSemaphore, u64, Stage)],
) -> Result<Vec<vk::SemaphoreSubmitInfo<'a>>, GpuError> {
    let mut infos = Vec::with_capacity(entries.len());
    for (semaphore, value, stage) in entries {
        if !Rc::ptr_eq(state, &semaphore.state) {
            return Err(GpuError::InvalidArgument(
                "timeline belongs to a different device",
            ));
        }
        if *stage == Stage::NONE
            || stage.intersects(Stage::HOST)
            || (stage.intersects(Stage::MESH) && !state.caps.mesh_shader)
        {
            return Err(GpuError::InvalidArgument(
                "invalid semaphore pipeline stage",
            ));
        }
        for previous in &infos {
            let previous: &vk::SemaphoreSubmitInfo<'_> = previous;
            if previous.semaphore == semaphore.raw {
                return Err(GpuError::InvalidArgument(
                    "duplicate semaphore in wait/signal list",
                ));
            }
        }
        infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore.raw)
                .value(*value)
                .stage_mask((*stage).into()),
        );
    }
    Ok(infos)
}

/// End recording and submit without presenting. Wait/signal tuples are
/// `(timeline, value, stage)`; use ALL_COMMANDS for resource-retirement signals.
/// # Safety
/// Recorded resources, descriptor targets and device-address targets must remain
/// alive until GPU completion. All accesses/layouts must be synchronized. Signal
/// values must be strictly increasing and within maxTimelineSemaphoreValueDifference.
/// Every wait must eventually be satisfied; semaphore owners must outlive their uses.
pub unsafe fn submit(
    device: &mut GpuDevice,
    command: &mut CommandBuffer,
    waits: &[(&TimelineSemaphore, u64, Stage)],
    signals: &[(&TimelineSemaphore, u64, Stage)],
) -> Result<(), GpuError> {
    command.same_device(&device.state)?;
    let waits = semaphore_infos(&device.state, waits)?;
    let signals = semaphore_infos(&device.state, signals)?;
    prepare_submit(command)?;
    let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command.raw)];
    unsafe {
        device.state.raw().queue_submit2(
            device.state.queue,
            &[vk::SubmitInfo2::default()
                .wait_semaphore_infos(&waits)
                .command_buffer_infos(&commands)
                .signal_semaphore_infos(&signals)],
            command.fence,
        )?;
    }
    command.status = CommandStatus::Pending;
    Ok(())
}

/// Acquire one frame; at most one may be outstanding. Two acquisition slots
/// permit two GPU frames in flight. A minimized window returns WindowMinimized.
pub fn acquire(device: &mut GpuDevice, timeout_ns: u64) -> Result<SwapchainFrame, GpuError> {
    if device.active.is_some() {
        return Err(GpuError::InvalidArgument("a frame is already acquired"));
    }
    if device.present_failed {
        return Err(GpuError::OutOfDate);
    }
    let extent = get_drawable_extent(device)?;
    if device.state.window.is_none() {
        return Err(GpuError::Unsupported(
            "headless device has no swapchain".into(),
        ));
    }
    if extent.x == 0 || extent.y == 0 {
        return Err(GpuError::WindowMinimized);
    }
    if extent != device.extent {
        return Err(GpuError::OutOfDate);
    }
    let Some(owner) = &device.swapchain else {
        return Err(GpuError::OutOfDate);
    };
    if device.present_ready.len() != device.views.len() {
        return Err(GpuError::OutOfDate);
    }
    let slot = (device.serial % 2) as usize;
    unsafe {
        device.state.raw().wait_semaphores(
            &vk::SemaphoreWaitInfo::default()
                .semaphores(&[device.retirement])
                .values(&[device.slot_values[slot]]),
            timeout_ns,
        )?;
        if device.acquire_pending[slot] {
            device
                .state
                .raw()
                .wait_for_fences(&[device.acquire_fences[slot]], true, timeout_ns)?;
        }
        device
            .state
            .raw()
            .reset_fences(&[device.acquire_fences[slot]])?;
        device.acquire_pending[slot] = false;
        let (index, suboptimal) = device
            .state
            .swapchain_api
            .as_ref()
            .expect("windowed device")
            .acquire_next_image(
                owner.swapchain,
                timeout_ns,
                device.acquire_ready[slot],
                device.acquire_fences[slot],
            )?;
        device.acquire_pending[slot] = true;
        device.active = Some((index, slot));
        device.generation = device.generation.wrapping_add(1);
        Ok(SwapchainFrame {
            render_view: device.views[index as usize].clone(),
            extent: device.extent,
            suboptimal,
            image_index: index,
            generation: device.generation,
        })
    }
}

/// Submit and present an acquired frame. Returns whether rebuilding is recommended.
/// The frame token is invalidated after successful submission, including when
/// presentation returns an error. Validation failures before submission allow retry.
/// # Safety
/// The `submit` safety contract applies. The acquired image must be transitioned
/// to PRESENT layout by this command; use ALL_COMMANDS as its final signal scope.
pub unsafe fn submit_and_present(
    device: &mut GpuDevice,
    command: &mut CommandBuffer,
    frame: &SwapchainFrame,
    waits: &[(&TimelineSemaphore, u64, Stage)],
    signals: &[(&TimelineSemaphore, u64, Stage)],
) -> Result<bool, GpuError> {
    command.same_device(&device.state)?;
    let Some((index, slot)) = device.active else {
        return Err(GpuError::InvalidArgument("no acquired frame"));
    };
    if index != frame.image_index
        || device.generation != frame.generation
        || !Rc::ptr_eq(&frame.render_view, &device.views[index as usize])
    {
        return Err(GpuError::InvalidArgument(
            "stale or foreign swapchain frame",
        ));
    }
    let mut wait_infos = semaphore_infos(&device.state, waits)?;
    let mut signal_infos = semaphore_infos(&device.state, signals)?;
    let serial = device
        .serial
        .checked_add(1)
        .ok_or(GpuError::InvalidArgument("frame timeline exhausted"))?;
    wait_infos.push(
        vk::SemaphoreSubmitInfo::default()
            .semaphore(device.acquire_ready[slot])
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
    );
    signal_infos.push(
        vk::SemaphoreSubmitInfo::default()
            .semaphore(device.present_ready[index as usize])
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
    );
    signal_infos.push(
        vk::SemaphoreSubmitInfo::default()
            .semaphore(device.retirement)
            .value(serial)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
    );
    prepare_submit(command)?;
    let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command.raw)];
    unsafe {
        device.state.raw().queue_submit2(
            device.state.queue,
            &[vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_infos)
                .command_buffer_infos(&commands)
                .signal_semaphore_infos(&signal_infos)],
            command.fence,
        )?;
        command.status = CommandStatus::Pending;
        device.serial = serial;
        device.slot_values[slot] = serial;
        device.active = None;
        let swapchain = device
            .swapchain
            .as_ref()
            .expect("acquired swapchain")
            .swapchain;
        let result = device
            .state
            .swapchain_api
            .as_ref()
            .expect("windowed device")
            .queue_present(
                device.state.queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[device.present_ready[index as usize]])
                    .swapchains(&[swapchain])
                    .image_indices(&[index]),
            );
        match result {
            Ok(suboptimal) => Ok(suboptimal || frame.suboptimal),
            Err(error) => {
                device.present_failed = true;
                Err(error.into())
            }
        }
    }
}

/// Record batched memory, buffer and whole-image barriers.
/// # Safety
/// Stage/access pairs must be compatible, old layouts must match GPU state, and
/// dependencies must cover all hazards. Resources must survive submission.
pub unsafe fn barrier(
    command: &mut CommandBuffer,
    barriers: &[Barrier<'_>],
) -> Result<(), GpuError> {
    command.recording(true)?;
    let mut memory = Vec::new();
    let mut buffers = Vec::new();
    let mut images = Vec::new();
    for dependency in barriers {
        match dependency {
            Barrier::Memory { before, after } => {
                memory.push(
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(before.0.into())
                        .src_access_mask(before.1.into())
                        .dst_stage_mask(after.0.into())
                        .dst_access_mask(after.1.into()),
                );
            }
            Barrier::Buffer {
                range,
                before,
                after,
            } => {
                command.same_device(&range.heap.owner.state)?;
                buffers.push(
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(before.0.into())
                        .src_access_mask(before.1.into())
                        .dst_stage_mask(after.0.into())
                        .dst_access_mask(after.1.into())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(range.heap.owner.buffer)
                        .offset(range.offset)
                        .size(range.size),
                );
            }
            Barrier::Texture {
                texture,
                before,
                after,
            } => {
                command.same_device(&texture.owner.state)?;
                if after.2 == TextureLayout::Undefined {
                    return Err(GpuError::InvalidArgument("cannot transition to UNDEFINED"));
                }
                let format = TextureFormatInfo::get_texture_format_info(texture.desc.format);
                let mut aspect = vk::ImageAspectFlags::empty();
                if format.depth {
                    aspect |= vk::ImageAspectFlags::DEPTH;
                }
                if format.stencil {
                    aspect |= vk::ImageAspectFlags::STENCIL;
                }
                if aspect.is_empty() {
                    aspect = vk::ImageAspectFlags::COLOR;
                }
                images.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(before.0.into())
                        .src_access_mask(before.1.into())
                        .dst_stage_mask(after.0.into())
                        .dst_access_mask(after.1.into())
                        .old_layout(before.2.into())
                        .new_layout(after.2.into())
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(texture.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect)
                                .level_count(texture.desc.mip_levels)
                                .layer_count(texture.desc.layer_count),
                        ),
                );
            }
        }
    }
    unsafe {
        command.state.raw().cmd_pipeline_barrier2(
            command.raw,
            &vk::DependencyInfo::default()
                .memory_barriers(&memory)
                .buffer_memory_barriers(&buffers)
                .image_memory_barriers(&images),
        );
    }
    Ok(())
}

/// Copy equal-sized buffer ranges, with four-byte-aligned offsets and size.
/// # Safety
/// The ranges must be synchronized and remain alive through GPU completion.
pub unsafe fn copy_memory(
    command: &mut CommandBuffer,
    source: GpuRange<'_>,
    destination: GpuRange<'_>,
) -> Result<(), GpuError> {
    command.recording(true)?;
    command.same_device(&source.heap.owner.state)?;
    command.same_device(&destination.heap.owner.state)?;
    if source.size != destination.size
        || !(source.offset | destination.offset | source.size).is_multiple_of(4)
    {
        return Err(GpuError::InvalidArgument(
            "copy ranges need equal sizes and four-byte alignment",
        ));
    }
    if source.heap.owner.buffer == destination.heap.owner.buffer
        && source.offset < destination.offset + destination.size
        && destination.offset < source.offset + source.size
    {
        return Err(GpuError::InvalidArgument("buffer copy ranges overlap"));
    }
    unsafe {
        command.state.raw().cmd_copy_buffer(
            command.raw,
            source.heap.owner.buffer,
            destination.heap.owner.buffer,
            &[vk::BufferCopy::default()
                .src_offset(source.offset)
                .dst_offset(destination.offset)
                .size(source.size)],
        );
    }
    Ok(())
}

/// Upload to TRANSFER_DESTINATION layout. Pitch 0 means tightly packed.
/// # Safety
/// Source and destination must be synchronized and remain alive until completion.
pub unsafe fn copy_memory_to_texture(
    command: &mut CommandBuffer,
    source: GpuRange<'_>,
    texture: &Texture,
    desc: &TextureCopyDesc,
    aspect: TextureAspect,
) -> Result<(), GpuError> {
    command.recording(true)?;
    command.same_device(&source.heap.owner.state)?;
    command.same_device(&texture.owner.state)?;
    if !texture
        .desc
        .usage
        .contains(TextureUsage::TRANSFER_DESTINATION)
    {
        return Err(GpuError::InvalidArgument(
            "texture lacks TRANSFER_DESTINATION usage",
        ));
    }
    let region = texture_copy_region(&texture.desc, desc, aspect, source.offset, source.size)?;
    unsafe {
        command.state.raw().cmd_copy_buffer_to_image(
            command.raw,
            source.heap.owner.buffer,
            texture.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
    Ok(())
}
/// Read from TRANSFER_SOURCE layout. Wait, then invalidate mapped memory before reading.
/// # Safety
/// Source and destination must be synchronized and remain alive until completion.
pub unsafe fn copy_texture_to_memory(
    command: &mut CommandBuffer,
    texture: &Texture,
    destination: GpuRange<'_>,
    desc: &TextureCopyDesc,
    aspect: TextureAspect,
) -> Result<(), GpuError> {
    command.recording(true)?;
    command.same_device(&destination.heap.owner.state)?;
    command.same_device(&texture.owner.state)?;
    if !texture.desc.usage.contains(TextureUsage::TRANSFER_SOURCE) {
        return Err(GpuError::InvalidArgument(
            "texture lacks TRANSFER_SOURCE usage",
        ));
    }
    let region = texture_copy_region(
        &texture.desc,
        desc,
        aspect,
        destination.offset,
        destination.size,
    )?;
    unsafe {
        command.state.raw().cmd_copy_image_to_buffer(
            command.raw,
            texture.image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            destination.heap.owner.buffer,
            &[region],
        );
    }
    Ok(())
}

pub(crate) fn texture_copy_region(
    texture: &TextureDesc,
    desc: &TextureCopyDesc,
    aspect: TextureAspect,
    buffer_offset: u64,
    buffer_size: u64,
) -> Result<vk::BufferImageCopy, GpuError> {
    if desc.mip_level >= texture.mip_levels || desc.base_slice >= texture.layer_count {
        return Err(GpuError::InvalidArgument("copy subresource out of bounds"));
    }
    let mip = U32x3 {
        x: (texture.extent.x >> desc.mip_level).max(1),
        y: (texture.extent.y >> desc.mip_level).max(1),
        z: (texture.extent.z >> desc.mip_level).max(1),
    };
    let offset = desc.offset;
    if offset.x >= mip.x || offset.y >= mip.y || offset.z >= mip.z {
        return Err(GpuError::InvalidArgument("copy offset out of bounds"));
    }
    let extent = U32x3 {
        x: if desc.extent.x == 0 {
            mip.x - offset.x
        } else {
            desc.extent.x
        },
        y: if desc.extent.y == 0 {
            mip.y - offset.y
        } else {
            desc.extent.y
        },
        z: if desc.extent.z == 0 {
            mip.z - offset.z
        } else {
            desc.extent.z
        },
    };
    if extent.x > mip.x - offset.x || extent.y > mip.y - offset.y || extent.z > mip.z - offset.z {
        return Err(GpuError::InvalidArgument("copy extent out of bounds"));
    }
    if offset.x > i32::MAX as u32 || offset.y > i32::MAX as u32 || offset.z > i32::MAX as u32 {
        return Err(GpuError::InvalidArgument(
            "copy offsets exceed signed coordinate bounds",
        ));
    }
    let layers = if desc.slice_count == 0 {
        texture.layer_count - desc.base_slice
    } else {
        desc.slice_count
    };
    if layers > texture.layer_count - desc.base_slice {
        return Err(GpuError::InvalidArgument("copy layers out of bounds"));
    }
    let format = TextureFormatInfo::get_texture_format_info(texture.format);
    let mask = match aspect {
        TextureAspect::Automatic if format.depth && format.stencil => {
            return Err(GpuError::InvalidArgument(
                "select Depth or Stencil for combined formats",
            ));
        }
        TextureAspect::Automatic if format.depth => vk::ImageAspectFlags::DEPTH,
        TextureAspect::Automatic if format.stencil => vk::ImageAspectFlags::STENCIL,
        TextureAspect::Automatic | TextureAspect::Color if !format.depth && !format.stencil => {
            vk::ImageAspectFlags::COLOR
        }
        TextureAspect::Depth if format.depth => vk::ImageAspectFlags::DEPTH,
        TextureAspect::Stencil if format.stencil => vk::ImageAspectFlags::STENCIL,
        _ => {
            return Err(GpuError::InvalidArgument(
                "copy aspect incompatible with texture",
            ));
        }
    };
    let block = format.block_extent;
    if block.x == 0 {
        return Err(GpuError::InvalidArgument("undefined copy format"));
    }
    let bytes = if mask == vk::ImageAspectFlags::STENCIL {
        1
    } else if format.depth {
        if texture.format == Format::D16Unorm {
            2
        } else {
            4
        }
    } else {
        u64::from(format.bytes_per_block)
    };
    if !offset.x.is_multiple_of(block.x)
        || !offset.y.is_multiple_of(block.y)
        || (!extent.x.is_multiple_of(block.x) && offset.x + extent.x != mip.x)
        || (!extent.y.is_multiple_of(block.y) && offset.y + extent.y != mip.y)
        || !buffer_offset.is_multiple_of(bytes)
        || !buffer_offset.is_multiple_of(4)
    {
        return Err(GpuError::InvalidArgument(
            "copy needs block-aligned offsets/extent and aligned buffer offset",
        ));
    }
    let row_bytes = u64::from(extent.x.div_ceil(block.x)) * bytes;
    let rows = u64::from(extent.y.div_ceil(block.y));
    let row_pitch = if desc.row_pitch_bytes == 0 {
        row_bytes
    } else {
        desc.row_pitch_bytes
    };
    if row_pitch < row_bytes || !row_pitch.is_multiple_of(bytes) {
        return Err(GpuError::InvalidArgument(
            "row pitch is too small or not block aligned",
        ));
    }
    let minimum_slice = row_pitch
        .checked_mul(rows)
        .ok_or(GpuError::InvalidArgument("copy pitch overflow"))?;
    let slice_pitch = if desc.slice_pitch_bytes == 0 {
        minimum_slice
    } else {
        desc.slice_pitch_bytes
    };
    if slice_pitch < minimum_slice || !slice_pitch.is_multiple_of(row_pitch) {
        return Err(GpuError::InvalidArgument(
            "slice pitch is too small or not a multiple of row pitch",
        ));
    }
    let row_length = (row_pitch / bytes)
        .checked_mul(u64::from(block.x))
        .ok_or(GpuError::InvalidArgument("row length overflow"))?;
    let image_height = (slice_pitch / row_pitch)
        .checked_mul(u64::from(block.y))
        .ok_or(GpuError::InvalidArgument("image height overflow"))?;
    if row_length > u64::from(u32::MAX)
        || image_height > u64::from(u32::MAX)
        || row_pitch > i32::MAX as u64
    {
        return Err(GpuError::InvalidArgument(
            "copy pitch exceeds Vulkan limits",
        ));
    }
    let slices = u64::from(layers) * u64::from(extent.z);
    let mut required = slice_pitch
        .checked_mul(slices - 1)
        .ok_or(GpuError::InvalidArgument("copy size overflow"))?;
    required = required
        .checked_add(row_pitch * (rows - 1))
        .ok_or(GpuError::InvalidArgument("copy size overflow"))?;
    required = required
        .checked_add(row_bytes)
        .ok_or(GpuError::InvalidArgument("copy size overflow"))?;
    if required > buffer_size {
        return Err(GpuError::InvalidArgument(
            "buffer range is too small for pitched copy",
        ));
    }
    Ok(vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .buffer_row_length(row_length as u32)
        .buffer_image_height(image_height as u32)
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(mask)
                .mip_level(desc.mip_level)
                .base_array_layer(desc.base_slice)
                .layer_count(layers),
        )
        .image_offset(vk::Offset3D {
            x: offset.x as i32,
            y: offset.y as i32,
            z: offset.z as i32,
        })
        .image_extent(vk::Extent3D {
            width: extent.x,
            height: extent.y,
            depth: extent.z,
        }))
}

/// Begin dynamic rendering. Extent is explicit, so attachmentless rendering works.
/// Sets a full-size viewport and scissor; use setters to override them.
/// # Safety
/// Views must be valid attachments in attachment layouts, compatible with the
/// bound pipeline and render extent, and retained until GPU completion.
pub unsafe fn begin_render_pass(
    command: &mut CommandBuffer,
    desc: &RenderingDesc<'_>,
    extent: U32x2,
) -> Result<(), GpuError> {
    command.recording(true)?;
    if extent.x == 0
        || extent.y == 0
        || desc.colors.len() > command.state.caps.max_color_attachments as usize
    {
        return Err(GpuError::InvalidArgument(
            "invalid render extent or color attachment count",
        ));
    }
    let limits = &command.state.limits;
    if extent.x > limits.max_framebuffer_width
        || extent.y > limits.max_framebuffer_height
        || extent.x > limits.max_viewport_dimensions[0]
        || extent.y > limits.max_viewport_dimensions[1]
        || extent.x as f32 > limits.viewport_bounds_range[1]
        || extent.y as f32 > limits.viewport_bounds_range[1]
    {
        return Err(GpuError::InvalidArgument(
            "render extent exceeds framebuffer/viewport limits",
        ));
    }
    if desc.depth.render_view.is_some() && !(0.0..=1.0).contains(&desc.depth.clear) {
        return Err(GpuError::InvalidArgument("depth clear must be in 0..=1"));
    }
    let mut colors = Vec::new();
    for attachment in desc.colors {
        let view = attachment.render_view.ok_or(GpuError::InvalidArgument(
            "color attachment needs a render view",
        ))?;
        command.same_device(&view.texture.owner.state)?;
        if view.subresources.level_count != 1
            || !matches!(
                view.view_type,
                vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_2D_ARRAY
            )
        {
            return Err(GpuError::InvalidArgument(
                "attachments require single-mip 2D/2D-array views",
            ));
        }
        if !view
            .texture
            .desc
            .usage
            .contains(TextureUsage::COLOR_ATTACHMENT)
            || extent.x > view.extent.x
            || extent.y > view.extent.y
        {
            return Err(GpuError::InvalidArgument(
                "color attachment usage or extent mismatch",
            ));
        }
        colors.push(
            vk::RenderingAttachmentInfo::default()
                .image_view(view.raw)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(load_op(attachment.load))
                .store_op(store_op(attachment.store))
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [
                            attachment.clear.x,
                            attachment.clear.y,
                            attachment.clear.z,
                            attachment.clear.w,
                        ],
                    },
                }),
        );
    }
    let mut depth = vk::RenderingAttachmentInfo::default();
    let mut stencil = vk::RenderingAttachmentInfo::default();
    for view in [desc.depth.render_view, desc.stencil.render_view]
        .into_iter()
        .flatten()
    {
        command.same_device(&view.texture.owner.state)?;
        if view.subresources.level_count != 1
            || !matches!(
                view.view_type,
                vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_2D_ARRAY
            )
        {
            return Err(GpuError::InvalidArgument(
                "attachments require single-mip 2D/2D-array views",
            ));
        }
        if !view
            .texture
            .desc
            .usage
            .contains(TextureUsage::DEPTH_STENCIL_ATTACHMENT)
            || extent.x > view.extent.x
            || extent.y > view.extent.y
        {
            return Err(GpuError::InvalidArgument(
                "depth/stencil attachment usage or extent mismatch",
            ));
        }
    }
    if let (Some(depth), Some(stencil)) = (desc.depth.render_view, desc.stencil.render_view)
        && depth.raw != stencil.raw
    {
        return Err(GpuError::InvalidArgument(
            "depth and stencil must use the same image view",
        ));
    }
    if let Some(view) = desc.depth.render_view {
        if !view.aspect.contains(vk::ImageAspectFlags::DEPTH) {
            return Err(GpuError::InvalidArgument("depth view lacks depth aspect"));
        }
        depth = depth
            .image_view(view.raw)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(load_op(desc.depth.load))
            .store_op(store_op(desc.depth.store))
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: desc.depth.clear,
                    stencil: 0,
                },
            });
    }
    if let Some(view) = desc.stencil.render_view {
        if !view.aspect.contains(vk::ImageAspectFlags::STENCIL) {
            return Err(GpuError::InvalidArgument(
                "stencil view lacks stencil aspect",
            ));
        }
        stencil = stencil
            .image_view(view.raw)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(load_op(desc.stencil.load))
            .store_op(store_op(desc.stencil.store))
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: u32::from(desc.stencil.clear),
                },
            });
    }
    let mut info = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.x,
                height: extent.y,
            },
        })
        .layer_count(1)
        .color_attachments(&colors);
    if desc.depth.render_view.is_some() {
        info = info.depth_attachment(&depth);
    }
    if desc.stencil.render_view.is_some() {
        info = info.stencil_attachment(&stencil);
    }
    unsafe {
        command.state.raw().cmd_begin_rendering(command.raw, &info);
    }
    command.rendering = true;
    set_viewport(
        command,
        &Viewport {
            width: extent.x as f32,
            height: extent.y as f32,
            ..Viewport::default()
        },
    )?;
    set_scissor(
        command,
        &Scissor {
            width: extent.x,
            height: extent.y,
            ..Scissor::default()
        },
    )?;
    Ok(())
}

pub fn end_render_pass(command: &mut CommandBuffer) -> Result<(), GpuError> {
    command.recording(false)?;
    if !command.rendering {
        return Err(GpuError::InvalidArgument("no active render pass"));
    }
    unsafe {
        command.state.raw().cmd_end_rendering(command.raw);
    }
    command.rendering = false;
    Ok(())
}
fn load_op(op: LoadOp) -> vk::AttachmentLoadOp {
    match op {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadOp::Discard => vk::AttachmentLoadOp::DONT_CARE,
    }
}
fn store_op(op: StoreOp) -> vk::AttachmentStoreOp {
    match op {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::Discard => vk::AttachmentStoreOp::DONT_CARE,
    }
}

pub fn bind_pso(command: &mut CommandBuffer, pso: &PSO) -> Result<(), GpuError> {
    command.recording(false)?;
    command.same_device(&pso.state)?;
    unsafe {
        command
            .state
            .raw()
            .cmd_bind_pipeline(command.raw, pso.bind_point, pso.raw);
    }
    command.bound = Some((pso.bind_point, pso.mesh));
    Ok(())
}

/// Bind the sampled-image, storage-image and sampler heaps for the current PSO.
/// # Safety
/// Initialize every descriptor used by shaders and retain heaps/targets until completion.
pub unsafe fn bind_descriptor_heaps(
    command: &mut CommandBuffer,
    heaps: [&GpuHeap; 3],
) -> Result<(), GpuError> {
    command.recording(false)?;
    let Some((bind_point, _)) = command.bound else {
        return Err(GpuError::InvalidArgument(
            "bind a PSO before descriptor heaps",
        ));
    };
    let mut bindings = Vec::new();
    for (index, heap) in heaps.into_iter().enumerate() {
        command.same_device(&heap.owner.state)?;
        if heap.owner.descriptor_set != Some(index) {
            return Err(GpuError::InvalidArgument(
                "expected sampled, storage, sampler heaps in that order",
            ));
        }
        bindings.push(
            vk::DescriptorBufferBindingInfoEXT::default()
                .address(heap.device_address())
                .usage(if index == 2 {
                    vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                } else {
                    vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                }),
        );
    }
    unsafe {
        command
            .state
            .descriptors()
            .cmd_bind_descriptor_buffers(command.raw, &bindings);
        command
            .state
            .descriptors()
            .cmd_set_descriptor_buffer_offsets(
                command.raw,
                bind_point,
                command.state.pipeline_layout,
                0,
                &[0, 1, 2],
                &[0, 0, 0],
            );
    }
    Ok(())
}

pub fn push_constants(
    command: &mut CommandBuffer,
    offset: u32,
    bytes: &[u8],
) -> Result<(), GpuError> {
    command.recording(false)?;
    if bytes.is_empty()
        || !offset.is_multiple_of(4)
        || !bytes.len().is_multiple_of(4)
        || u64::from(offset) + bytes.len() as u64 > command.state.caps.max_push_data_size
    {
        return Err(GpuError::InvalidArgument(
            "push constants exceed limit or lack four-byte alignment",
        ));
    }
    unsafe {
        command.state.raw().cmd_push_constants(
            command.raw,
            command.state.pipeline_layout,
            vk::ShaderStageFlags::ALL,
            offset,
            bytes,
        );
    }
    Ok(())
}

pub fn set_viewport(command: &mut CommandBuffer, viewport: &Viewport) -> Result<(), GpuError> {
    command.recording(false)?;
    for value in [
        viewport.x,
        viewport.y,
        viewport.width,
        viewport.height,
        viewport.min_depth,
        viewport.max_depth,
    ] {
        if !value.is_finite() {
            return Err(GpuError::InvalidArgument("viewport values must be finite"));
        }
    }
    let limits = &command.state.limits;
    if viewport.width <= 0.0
        || viewport.height == 0.0
        || viewport.width > limits.max_viewport_dimensions[0] as f32
        || viewport.height.abs() > limits.max_viewport_dimensions[1] as f32
        || !(0.0..=1.0).contains(&viewport.min_depth)
        || !(0.0..=1.0).contains(&viewport.max_depth)
        || viewport.x < limits.viewport_bounds_range[0]
        || viewport.x + viewport.width > limits.viewport_bounds_range[1]
        || viewport.y.min(viewport.y + viewport.height) < limits.viewport_bounds_range[0]
        || viewport.y.max(viewport.y + viewport.height) > limits.viewport_bounds_range[1]
    {
        return Err(GpuError::InvalidArgument("viewport exceeds device bounds"));
    }
    unsafe {
        command.state.raw().cmd_set_viewport(
            command.raw,
            0,
            &[vk::Viewport {
                x: viewport.x,
                y: viewport.y,
                width: viewport.width,
                height: viewport.height,
                min_depth: viewport.min_depth,
                max_depth: viewport.max_depth,
            }],
        );
    }
    Ok(())
}
pub fn set_scissor(command: &mut CommandBuffer, scissor: &Scissor) -> Result<(), GpuError> {
    command.recording(false)?;
    if scissor.x < 0
        || scissor.y < 0
        || u64::from(scissor.x as u32) + u64::from(scissor.width) > i32::MAX as u64
        || u64::from(scissor.y as u32) + u64::from(scissor.height) > i32::MAX as u64
    {
        return Err(GpuError::InvalidArgument(
            "scissor exceeds signed coordinate bounds",
        ));
    }
    unsafe {
        command.state.raw().cmd_set_scissor(
            command.raw,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D {
                    x: scissor.x,
                    y: scissor.y,
                },
                extent: vk::Extent2D {
                    width: scissor.width,
                    height: scissor.height,
                },
            }],
        );
    }
    Ok(())
}

/// # Safety
/// Shader inputs, descriptors and attachment formats must match the bound PSO;
/// all referenced resources and addresses must be valid and synchronized.
pub unsafe fn draw(
    command: &mut CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) -> Result<(), GpuError> {
    command.recording(false)?;
    if !command.rendering || command.bound != Some((vk::PipelineBindPoint::GRAPHICS, false)) {
        return Err(GpuError::InvalidArgument(
            "draw needs an active render pass and graphics PSO",
        ));
    }
    unsafe {
        command.state.raw().cmd_draw(
            command.raw,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }
    Ok(())
}

/// # Safety
/// The `draw` contract applies. Index values must address valid vertices.
#[allow(clippy::too_many_arguments)]
pub unsafe fn draw_indexed(
    command: &mut CommandBuffer,
    indices: GpuRange<'_>,
    index_type: IndexType,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
) -> Result<(), GpuError> {
    command.recording(false)?;
    command.same_device(&indices.heap.owner.state)?;
    if !command.rendering
        || command.bound != Some((vk::PipelineBindPoint::GRAPHICS, false))
        || indices.heap.owner.descriptor_set.is_some()
    {
        return Err(GpuError::InvalidArgument(
            "indexed draw needs a render pass, graphics PSO and index buffer",
        ));
    }
    let (size, kind) = match index_type {
        IndexType::Uint16 => (2, vk::IndexType::UINT16),
        IndexType::Uint32 => (4, vk::IndexType::UINT32),
    };
    if !indices.offset.is_multiple_of(size)
        || (u64::from(first_index) + u64::from(index_count)) * size > indices.size
    {
        return Err(GpuError::InvalidArgument(
            "index range out of bounds or misaligned",
        ));
    }
    unsafe {
        command.state.raw().cmd_bind_index_buffer(
            command.raw,
            indices.heap.owner.buffer,
            indices.offset,
            kind,
        );
        command.state.raw().cmd_draw_indexed(
            command.raw,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }
    Ok(())
}

/// # Safety
/// Shader descriptor/address accesses must be valid, retained and synchronized.
pub unsafe fn dispatch(command: &mut CommandBuffer, groups: U32x3) -> Result<(), GpuError> {
    command.recording(true)?;
    if command.bound != Some((vk::PipelineBindPoint::COMPUTE, false)) {
        return Err(GpuError::InvalidArgument("dispatch needs a compute PSO"));
    }
    for (count, limit) in [groups.x, groups.y, groups.z]
        .into_iter()
        .zip(command.state.limits.max_compute_work_group_count)
    {
        if count > limit {
            return Err(GpuError::InvalidArgument(
                "compute group count exceeds device limit",
            ));
        }
    }
    unsafe {
        command
            .state
            .raw()
            .cmd_dispatch(command.raw, groups.x, groups.y, groups.z);
    }
    Ok(())
}
/// # Safety
/// The `draw` contract applies to the bound mesh pipeline.
pub unsafe fn draw_mesh(command: &mut CommandBuffer, groups: U32x3) -> Result<(), GpuError> {
    command.recording(false)?;
    let Some(mesh) = &command.state.mesh_api else {
        return Err(GpuError::Unsupported(
            "VK_EXT_mesh_shader / meshShader".into(),
        ));
    };
    if !command.rendering || command.bound != Some((vk::PipelineBindPoint::GRAPHICS, true)) {
        return Err(GpuError::InvalidArgument(
            "mesh draw needs a render pass and mesh PSO",
        ));
    }
    let limits = &command.state.mesh_limits;
    for (count, limit) in [groups.x, groups.y, groups.z]
        .into_iter()
        .zip(limits.max_mesh_work_group_count)
    {
        if count > limit {
            return Err(GpuError::InvalidArgument(
                "mesh workgroup count exceeds device limit",
            ));
        }
    }
    if u64::from(groups.x) * u64::from(groups.y) > u64::from(limits.max_mesh_work_group_total_count)
        || u64::from(groups.x) * u64::from(groups.y) * u64::from(groups.z)
            > u64::from(limits.max_mesh_work_group_total_count)
    {
        return Err(GpuError::InvalidArgument(
            "mesh total workgroup count exceeds device limit",
        ));
    }
    unsafe {
        mesh.cmd_draw_mesh_tasks(command.raw, groups.x, groups.y, groups.z);
    }
    Ok(())
}

pub fn write_timestamp(
    command: &mut CommandBuffer,
    index: u32,
    stage: Stage,
) -> Result<(), GpuError> {
    command.recording(false)?;
    if command.queries == vk::QueryPool::null() {
        return Err(GpuError::Unsupported(
            "timestamps are disabled or unsupported by this queue".into(),
        ));
    }
    let flags: vk::PipelineStageFlags2 = stage.into();
    if flags.as_raw().count_ones() != 1
        || stage.intersects(Stage::HOST)
        || (stage.intersects(Stage::MESH) && !command.state.caps.mesh_shader)
    {
        return Err(GpuError::InvalidArgument(
            "timestamp needs one supported device stage",
        ));
    }
    let Some(written) = command.written_queries.get_mut(index as usize) else {
        return Err(GpuError::InvalidArgument("timestamp index out of bounds"));
    };
    if *written {
        return Err(GpuError::InvalidArgument(
            "timestamp query already written in this recording",
        ));
    }
    unsafe {
        command
            .state
            .raw()
            .cmd_write_timestamp2(command.raw, flags, command.queries, index);
    }
    *written = true;
    Ok(())
}

/// Read ticks after command completion. Convert deltas with timestamp_period_ns;
/// counters wrap at the queue's `timestamp_valid_bits` (reported here as a mask).
pub fn read_timestamps(
    command: &CommandBuffer,
    first: u32,
    output: &mut [u64],
) -> Result<u64, GpuError> {
    if command.status != CommandStatus::Pending || output.is_empty() {
        return Err(GpuError::InvalidArgument(
            "timestamp results require a submitted command and nonempty output",
        ));
    }
    let end = (first as usize)
        .checked_add(output.len())
        .ok_or(GpuError::InvalidArgument("timestamp range overflow"))?;
    if end > command.written_queries.len() {
        return Err(GpuError::InvalidArgument("timestamp range out of bounds"));
    }
    for written in &command.written_queries[first as usize..end] {
        if !written {
            return Err(GpuError::InvalidArgument("timestamp query was not written"));
        }
    }
    wait_command(command, 0)?;
    unsafe {
        command.state.raw().get_query_pool_results(
            command.queries,
            first,
            output,
            vk::QueryResultFlags::TYPE_64,
        )?;
    }
    let bits = command.state.timestamp_valid_bits;
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    };
    for value in output {
        *value &= mask;
    }
    Ok(mask)
}
