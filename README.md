# NoGraphicsAPI-rs

A small, explicit graphics library for **Vulkan 1.4**, using Rust, `ash`,
`ash-window`, `winit`, and Vulkan Memory Allocator (`vk-mem`). It provides a
single graphics/compute/presentation queue, device-address-based buffer access,
descriptor-buffer heaps, dynamic rendering, and timeline synchronization.

## Build and run

Use Rust **1.88 or newer**, a C++ compiler for VMA, and an installed Vulkan loader.
On Linux, `winit` also needs the platform's Wayland/X11 development dependencies.

```sh
git submodule update --init --recursive
cargo run -- --validation
```

The executable in `src/main.rs` is a cached text-rendering example. It uses a
compute shader to rasterize TrueType outlines into an atlas once, then renders
"Hello Text" and a live, smoothly scaling FPS label from that atlas. The
application owns the window/event loop; the graphics library retains an
`Arc<Window>`. Windows and Linux surfaces go through `ash-window`.

For a bounded presentation/resize smoke test:

```sh
cargo run -- --validation --resize-test --frames 12
```

`validation: true` enables Khronos validation and synchronization validation.
Messages are printed to stderr; `validation_error_count` exposes the error count.
Without validation, the validation layer is not required.

### Pinned source dependencies

HarfRust is tracked as the `src/harfrust` Git submodule and is pinned to a
release commit. The root `Cargo.toml` depends on the `harfrust` crate inside
that submodule, while `Cargo.lock` pins its registry dependencies.

To update HarfRust, fetch its tags, check out the desired release in detached
HEAD mode, update the matching version in `Cargo.toml`, and regenerate the
lockfile:

```sh
git -C src/harfrust fetch --tags
git -C src/harfrust checkout --detach <release-tag>
cargo check
```

### Vulkan version and bindings

Both the loader and the selected physical device must support Vulkan 1.4.
`VULKAN_API_VERSION` is passed to instance creation and VMA. Older devices are
rejected with diagnostic errors.

The pinned `ash` 0.38 release has Vulkan 1.3.281-generated bindings. All commands
used here are Vulkan 1.2/1.3 core commands, which remain valid in Vulkan 1.4, or
explicitly enabled extension commands. The library does not expose newly added
1.4-only command wrappers. Current upstream `ash` changed its Rust interfaces
while retaining the 0.38 version number and is incompatible with the pinned
`vk-mem`; these dependencies intentionally remain on compatible revisions.

## Device requirements

`create_device(&DeviceDesc)` returns `Result<GpuDevice, GpuError>`. A default
description creates a headless device. Set `window: Some(window.clone())` for
presentation.

Required capabilities:

- Vulkan 1.4, buffer device addresses, and `shaderInt64`.
- Timeline semaphores, synchronization2, dynamic rendering, and maintenance4.
- `VK_EXT_descriptor_buffer` and `descriptorBuffer`.
- Runtime descriptor arrays, partially bound descriptors, and sampled/storage
  image nonuniform indexing.
- Independent blending, cube arrays, extended storage-image formats, and
  vertex/fragment shader stores and atomics.
- One queue family supporting graphics and compute, plus presentation when windowed.
- Descriptor-buffer bindings and limits sufficient for the configured arrays.

`DeviceDesc::descriptor_count` defaults to 1024 entries **per array**. Reduce it
if a device's limits require smaller arrays. Device selection prefers a suitable
discrete GPU, then integrated GPUs, then other devices. Rejections list missing
requirements by device.

Mesh shaders, BC/ASTC texture compression, and anisotropic filtering are queried
separately. `get_device_caps` reports enabled capabilities, not every feature the
driver advertises. `create_mesh_pso` and anisotropic sampler creation return
`GpuError::Unsupported` when the corresponding feature is unavailable. Optional
features such as 16-bit shader I/O are not enabled in this first implementation.

There is no software emulation or alternate descriptor-set renderer. Unsupported
requirements return errors, including unsupported texture format/usage pairs.

## API overview

All functions are re-exported from `lib.rs`. Implementation is split by concern:

| File | Responsibility |
| --- | --- |
| `src/lib.rs` | Public descriptions, enums, errors, format metadata |
| `src/device.rs` | Initialization, capabilities, surface and swapchain lifetime |
| `src/resource.rs` | Allocations, textures, views, samplers, descriptor bytes |
| `src/pipeline.rs` | Graphics, mesh, and compute pipelines |
| `src/command.rs` | Recording, barriers, submission, timelines, presentation |

The requested entry points are implemented:

- `create_device`, `destroy_device`, `get_device_caps`, `supports_texture_format`,
  `get_drawable_extent`, `wait_idle`.
- `create_timeline_semaphore`, `destroy_timeline_semaphore`,
  `timeline_completed_value`, `wait_timeline`.
- `acquire`, `submit_and_present`.
- `create_graphics_pso`, `create_mesh_pso`, `create_compute_pso`.
- `begin_command`, `submit`, `copy_memory`, `copy_memory_to_texture`,
  `copy_texture_to_memory`, `begin_render_pass`, `end_render_pass`, `barrier`.

Companion operations provide allocation/destruction, CPU mapping, descriptor
writing/binding, pipeline binding, push constants, viewport/scissor, drawing,
indexed drawing, mesh drawing, compute dispatch, timestamps, command reuse, and
swapchain rebuilding. Depth/stencil configuration lives in pipeline descriptions.

Fallible functions return `Result`; in particular,
`timeline_completed_value` returns `Result<u64, GpuError>`. Wait timeouts are in
nanoseconds: zero polls and `u64::MAX` waits indefinitely. `supports_texture_format`
returns `bool` for an optimal-tiled 2D format/usage combination; texture creation
also checks image type, dimensions, flags, array size and mip count.

## Ownership and synchronization

Handles own their Vulkan objects and release them on `Drop`. Explicit `destroy_*`
functions are also available. Resource owners retain the device backend; views
retain textures. All handles are deliberately thread-confined (`Rc`), which also
provides Vulkan's required external synchronization for the single queue.

This is an explicit GPU API: **recording a resource does not retain it until GPU
completion**. The `unsafe` contracts on submission, draw/dispatch, copies,
barriers, and descriptor writes require the caller to:

1. Keep every referenced resource alive through completion, including resources
   reached indirectly through descriptors or device addresses.
2. Provide correct stage/access dependencies and image layouts.
3. Avoid overwriting mapped memory or descriptor slots while the GPU uses them.
4. Use valid SPIR-V, shader inputs and pipeline-compatible attachments.

The safe configuration helpers check bounds/device identity where applicable;
they do not substitute for those GPU-side contracts. `PSO` creation is also
`unsafe` because shader validity and enabled-feature compatibility are caller
requirements, as with `ash`.

`begin_command` allocates and begins a primary command buffer. `submit` ends its
recording. Use `reset_command` to wait for **that command's fence**, recycle its
pool, and record again. It can also discard an unsubmitted recording. An already
submitted command cannot be resubmitted without resetting. `wait_command` waits
without discarding query results. Dropping a pending command waits for its fence;
ordinary resource destruction does not introduce hidden device-idle waits.

Timeline submission arguments are slices of `(timeline, value, stage)` tuples.
Use `Stage::ALL_COMMANDS` for resource-retirement signals. Signal values must
increase monotonically, remain within `DeviceCaps::max_timeline_value_difference`,
and all waits must eventually be satisfied.

Whole-image barriers cover every mip and layer. `Barrier::Memory` and
`Barrier::Buffer` express global and buffer-range dependencies. The API does not
track image layouts implicitly, so CPU recording order cannot accidentally become
an assumed GPU execution order.

### Upload and readback

`create_gpu_heap` creates a buffer supporting shader device addresses, storage,
index, indirect and transfer use. Suballocate with `heap.range(offset, size)`;
the resulting `GpuRange` retains a Rust borrow and exposes its device address.

```rust,no_run
use no_graphics_api_rs::*;

fn upload_example(device: &mut GpuDevice) -> Result<(), GpuError> {
    let mut upload = create_gpu_heap(device, 16, MemoryType::CpuVisible)?;
    let gpu = create_gpu_heap(device, 16, MemoryType::GpuOnly)?;
    let mut command = begin_command(device)?;

    unsafe {
        // Neither allocation has an outstanding GPU use.
        write_memory(&mut upload, 0, &[1; 16])?; // writes and flushes
        copy_memory(&mut command, upload.range(0, 16)?, gpu.range(0, 16)?)?;
        submit(device, &mut command, &[], &[])?;
    }
    wait_command(&command, u64::MAX)?;
    // Resources can now be released or safely reused.
    Ok(())
}
```

For readback, copy into `MemoryType::Readback`, record a transfer-write → host-read
barrier, wait for completion, call `invalidate_memory`, then borrow through
`mapped_memory`. For direct mapped writes, call `flush_memory` afterward.
Mapping requires exclusive CPU/GPU access to the range.

Images use VMA-managed GPU allocations. `get_texture_heap` retains an image's
allocation owner; it does not create a separate manually placed image heap.
Copy descriptions use bytes for row/slice pitches; zero pitches mean tightly
packed. Zero extents/counts select the remaining subresource extent/layers.
Combined depth/stencil copies require an explicit single aspect. Compressed
copies must obey block alignment, except for partial blocks at mip edges.

## Shader and descriptor contract

Every pipeline shares one device-owned pipeline layout:

| Set | Binding | Type | Creation function |
| --- | --- | --- | --- |
| 0 | 0 | `sampled image[descriptor_count]` | `create_texture_descriptor_heap(..., Sampled)` |
| 1 | 0 | `storage image[descriptor_count]` | `create_texture_descriptor_heap(..., Storage)` |
| 2 | 0 | `sampler[descriptor_count]` | `create_sampler_descriptor_heap` |

All three arrays are visible to every supported shader stage. Bind a PSO, then
call `bind_descriptor_heaps(command, [sampled, storage, samplers])` for the
pipeline bind point you will use. Graphics and compute bindings are distinct.
Unaccessed array entries may be uninitialized; accessed entries must be written.

Heap sizes, binding offsets and descriptor strides are queried from the driver.
Do not calculate slot offsets from `DeviceCaps::texture_descriptor_size`: it is
the larger of the sampled/storage descriptor sizes, not necessarily either
array's stride. Use `write_texture_descriptor` and `write_sampler_descriptor`.

Sampled descriptors use `TextureLayout::Sampled`; storage descriptors use
`TextureLayout::General`. Transition images accordingly before shader access.
Storage views contain exactly one mip; sampled depth/stencil views select one
aspect. Descriptor writes flush host memory but do not manage target lifetimes.

Shaders use entry point **`main`**. Graphics pipelines use triangle lists, one
sample per pixel, and device-address vertex fetching rather than fixed vertex
bindings. Mesh pipelines use mesh + optional fragment shaders, without a task
stage. Pipeline descriptions supply attachment formats, blending, rasterization,
and depth/stencil state. Viewport and scissor are dynamic and initially cover the
extent supplied to `begin_render_pass`.

Push constants are available up to `max_push_data_size`, with four-byte-aligned
offsets and sizes. Buffer addresses are 64-bit values; respect the shader's
buffer-reference alignment and Rust/GLSL layout when packing them. Explicit
enabled features permit the tested Vulkan 1.3 SPIR-V target, including LocalSizeId.
Additional shader capabilities still need corresponding enabled device features.

`tests/gpu.rs` contains a complete device-address and bindless GLSL compute
example, plus graphics and mesh shader examples compiled with `glslc`.

## Frames and resizing

1. Reset a completed command buffer.
2. `acquire(device, timeout)` to get the extent and render view.
3. Transition the image from its known layout (or `Undefined` to discard contents)
   to `ColorAttachment`.
4. Begin rendering, draw, end rendering.
5. Transition to `Present`.
6. `submit_and_present(device, command, &frame, waits, signals)`.

There may be one unsubmitted acquired frame at a time, with two acquisition
slots permitting two submitted GPU frames in flight. Acquisition semaphores are
reused after their submission completes; presentation semaphores are per image.
The library supplies the binary acquire/present waits/signals internally.

A frame token becomes invalid after successful submission, even if presentation
then fails. Errors detected before submission permit retrying the same frame.
Do not abandon an acquired frame while continuing to use the swapchain; present
it first, or destroy the device. Keeping the frame value after presentation
retains its view but does not permit presenting it again.

On `OutOfDate`, call `recreate_swapchain` and retry on the next frame. A `true`
result from `submit_and_present` also recommends rebuilding. Rebuilding waits
for idle and requires no outstanding acquired frame. `WindowMinimized` means
rendering should pause until the drawable has nonzero dimensions. `SurfaceLost`
requires recreating the windowed device. See `src/main.rs` for this sequence.

## Design choices

The pre-existing resource and description types are reused, with Rust ownership
and public caller-supplied fields. `DeviceInit` and the unfinished `new` entry
point were replaced by `Result<GpuDevice, GpuError>` and `create_device`.

The additional structures are:

- Private `DeviceState`: shared Vulkan parent lifetime and partial-init cleanup.
- Public `Sampler`: a sampler must stay alive independently of its descriptor bytes.

`Barrier` and `TextureLayout` are enums expressing synchronization information
absent from the original types. Helpers are limited to repeated operations and
validation/conversion logic. Authored library, example and test code use no closures.

## Verification

```sh
cargo fmt --package no-graphics-api-rs -- --check
cargo clippy --all-targets --no-deps -- -D warnings
cargo test
cargo doc --no-deps
```

Pure tests cover rejection diagnostics, synchronization conversions, texture
shape/mip validation, compressed edge copies, depth/stencil aspect sizes, pitch
overflow, and padded array transfers.

Opt-in real-driver tests require Vulkan 1.4 + descriptor buffers, Khronos
validation, and `glslc` in `PATH`:

```sh
cargo test --test gpu -- --ignored --nocapture
```

They verify actual readback values for buffer-address writes, bindless sampled
images/samplers, storage images, graphics rendering, and mesh rendering when
supported. Otherwise they verify the mesh unsupported-feature error. They also
exercise timeline timeouts/signals/waits, timestamps, command reuse, and resource
cleanup, requiring zero validation errors.
