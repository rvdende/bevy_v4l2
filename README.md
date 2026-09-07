# bevy_v4l2

Low-latency V4L2 webcam capture for [Bevy](https://bevy.org). Raw camera frames are read by the
GPU in place through DMA-BUF import, with no CPU copy, and converted from packed YUV to RGB in the
material's fragment shader. MJPEG streams are decoded on a thread when only a compressed mode
reaches the wanted frame rate.

```rust
use bevy::prelude::*;
use bevy_v4l2::{CameraFormat, DmabufTexturePlugin, FrameFormat, RequestedFormat, Webcam, WebcamPlugin};

fn main() {
    App::new()
        // Must come before DefaultPlugins so the Vulkan device gets the DMA-BUF extensions.
        .add_plugins(DmabufTexturePlugin)
        .add_plugins(DefaultPlugins)
        .add_plugins(WebcamPlugin)
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 1.3, 2.4).looking_at(Vec3::new(0.0, 0.45, 0.0), Vec3::Y),
    ));
    // 1280x720 at 60 fps; raw zero-copy if a raw mode reaches the rate, MJPEG otherwise. The
    // plugin attaches a plane sized to the camera's aspect ratio.
    commands.spawn((
        Webcam::new(RequestedFormat::Closest(CameraFormat::new(1280, 720, FrameFormat::Any, 60))),
        Transform::from_xyz(0.0, 0.45, 0.0),
    ));
}
```

`RequestedFormat` has `Exact`, `Closest`, `HighestResolution` and `HighestFrameRate`;
`FrameFormat` is `Any`, `Yuyv`, `Uyvy` or `Mjpeg`. `Webcam::want(w, h, fps)` is shorthand for the
`Closest`/`Any` request above, `Webcam::auto()` for `HighestResolution`, and `.device(path)`
picks a node other than `/dev/video0`. Give the entity your own `Mesh3d` to show the feed on any
shape. For lower-level use see
[`list_modes`](https://docs.rs/bevy_v4l2/latest/bevy_v4l2/fn.list_modes.html) and
[`choose_mode`](https://docs.rs/bevy_v4l2/latest/bevy_v4l2/fn.choose_mode.html) yourself.

## What is in the crate

| module | contents |
|---|---|
| `capture` | `Capture`: V4L2 streaming with `MMAP` buffers exported as DMA-BUF fds, newest-frame-wins delivery, damaged-frame accounting, CPU cache write-back |
| `dmabuf` | `DmabufTexturePlugin` (enables `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, `VK_EXT_queue_family_foreign` during Bevy's device creation) and `import_dmabuf_texture` |
| `plugin` | `WebcamPlugin`, the `Webcam` component, `WebcamMaterial` with the YUV shader, `WebcamFeed` (texture handle, stats), `WebcamStats` |
| `mjpeg` | `Decoder`: a decode thread with buffer recycling |
| `select` | `RequestedFormat`, `CameraFormat`, `FrameFormat`, `choose`: the mode policy, with tests |
| `controls` | `devices()`, `settings()`, `get_setting`, `set_setting`, `set_focus` (turns autofocus off first) |

Features: `dmabuf` (default) and `mjpeg` (default). Without `dmabuf` frames are uploaded with
`write_texture` from the mmap'd buffer.

## Frame paths

* **Raw 4:2:2 (YUYV/UYVY), zero-copy.** Each kernel buffer is exported once with `VIDIOC_EXPBUF`
  and imported into Vulkan as a linear `R8G8B8A8` image half the frame width (one texel per pixel
  pair). Per frame the render world records one GPU copy into the material's texture and re-queues
  the buffer when the GPU signals completion.
* **MJPEG.** A decode thread turns each JPEG into RGBA (zune-jpeg) and the render world uploads it.

### Cache coherence

uvcvideo writes frames with CPU `memcpy`. A GPU reading the DMA-BUF over PCIe does not snoop the
CPU cache, so lines still dirty in L3 arrive stale and show up as short horizontal streaks. The
capture thread therefore writes back the CPU cache for each frame after dequeue
(`clflushopt` + `sfence`, about 70 µs at 1080p). `Webcam { verify_frames: 90, .. }` reads
GPU copies back and diffs them against the kernel buffer to prove it on your hardware.

## Measured (Logitech BRIO, USB 3, RTX 4090, NVIDIA 595)

| mode | path | delivered |
|---|---|---|
| 1920x1080 YUYV 30 | DMA-BUF zero-copy | 30 fps, 70 µs flush |
| 1280x720 MJPEG 60 | CPU decode | 59 fps, 2.2 ms decode |
| 1920x1080 MJPEG 60 | CPU decode | 55 fps (debug build), 4.3 ms decode |

## Requirements

Linux. For zero-copy: a Vulkan driver with `VK_EXT_external_memory_dma_buf` and
`VK_EXT_image_drm_format_modifier` (NVIDIA 5xx and Mesa both qualify); otherwise the plugin falls
back to a CPU upload and reports it in `WebcamStats::mode`. Raw HD modes need a USB 3 link.

## License

MIT or Apache-2.0, at your option.
