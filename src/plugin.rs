//! [`WebcamPlugin`]: a V4L2 camera on a plane.
//!
//! Data path per frame, raw formats:
//! 1. The kernel driver writes into an `MMAP` buffer that was exported as a DMA-BUF at start-up.
//! 2. The capture thread dequeues it, writes back the CPU cache, and queues the buffer index.
//! 3. The render world records one GPU `copy_texture_to_texture` from the imported DMA-BUF image
//!    into the material's texture, then re-queues the buffer once the GPU signals completion.
//!    No CPU ever touches the pixels.
//! 4. The material's fragment shader converts packed YUV to RGB.
//!
//! MJPEG streams are decoded to RGBA on a thread and uploaded. If DMA-BUF import is unavailable
//! the plugin falls back to `queue.write_texture` from the mmap'd buffer and says so in
//! [`WebcamStats::mode`].

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use bevy::{
    asset::{RenderAssetUsages, embedded_asset},
    image::ImageSampler,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin},
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError, TextureDimension, TextureFormat, TextureUsages,
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderRef,
};

#[cfg(feature = "dmabuf")]
use crate::capture::DequeuedFrame;
use crate::capture::{Capture, CaptureConfig, FOURCC_MJPG, FOURCC_UYVY, FOURCC_YUYV, FrameLayout};
#[cfg(feature = "mjpeg")]
use crate::mjpeg::Decoder;

/// Puts a webcam feed on a plane. Add after `DefaultPlugins`.
///
/// For the zero-copy path, [`crate::DmabufTexturePlugin`] must be added *before* `DefaultPlugins`.
pub struct WebcamPlugin {
    pub config: CaptureConfig,
    /// Flip horizontally, like a mirror.
    pub mirror: bool,
    /// Skip DMA-BUF import and always upload through the CPU (for comparison).
    pub force_upload: bool,
    /// Spawn a plane showing the feed at start-up. Set to `false` to use [`WebcamShared::image`]
    /// and [`WebcamMaterial`] yourself.
    pub spawn_plane: bool,
    /// Height of the spawned plane in world units; width follows the aspect ratio.
    pub plane_height: f32,
    /// Where the spawned plane goes.
    pub plane_transform: Transform,
    /// Read back the first N GPU copies and compare them byte-for-byte with the kernel buffer.
    pub verify_frames: u32,
    /// Skip the foreign-queue acquire/release barriers (diagnostic).
    pub no_barrier: bool,
}

impl Default for WebcamPlugin {
    fn default() -> Self {
        Self {
            config: CaptureConfig::default(),
            mirror: false,
            force_upload: false,
            spawn_plane: true,
            plane_height: 0.9,
            plane_transform: Transform::from_xyz(0.0, 0.45, 0.0),
            verify_frames: 0,
            no_barrier: false,
        }
    }
}

impl Plugin for WebcamPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "webcam.wgsl");
        app.add_plugins((
            MaterialPlugin::<WebcamMaterial>::default(),
            ExtractResourcePlugin::<WebcamShared>::default(),
        ))
        .insert_resource(WebcamSettings {
            config: self.config.clone(),
            mirror: self.mirror,
            force_upload: self.force_upload,
            spawn_plane: self.spawn_plane,
            plane_height: self.plane_height,
            plane_transform: self.plane_transform,
            verify_frames: self.verify_frames,
            no_barrier: self.no_barrier,
        })
        .init_resource::<WebcamStatus>()
        .add_systems(Startup, start_webcam);

        app.sub_app_mut(RenderApp)
            .init_resource::<WebcamGpu>()
            .add_systems(
                Render,
                upload_webcam_frame.in_set(RenderSystems::PrepareResources),
            );
    }
}

#[derive(Resource, Clone)]
struct WebcamSettings {
    config: CaptureConfig,
    mirror: bool,
    force_upload: bool,
    spawn_plane: bool,
    plane_height: f32,
    plane_transform: Transform,
    verify_frames: u32,
    no_barrier: bool,
}

/// Outcome of opening the camera.
#[derive(Resource, Default, Debug)]
pub enum WebcamStatus {
    #[default]
    Starting,
    Running,
    Failed(String),
}

/// How frames reach the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferMode {
    #[default]
    Unknown,
    /// DMA-BUF imported as a Vulkan image; the GPU reads the kernel buffer directly.
    DmabufZeroCopy,
    /// `queue.write_texture` from the mmap'd buffer.
    CpuUpload,
    /// JPEG frames decoded on a CPU thread, then uploaded as RGBA.
    MjpegDecode,
}

impl TransferMode {
    pub fn label(self) -> &'static str {
        match self {
            TransferMode::DmabufZeroCopy => "DMA-BUF zero-copy",
            TransferMode::CpuUpload => "CPU upload",
            TransferMode::MjpegDecode => "MJPEG CPU decode",
            TransferMode::Unknown => "initialising",
        }
    }
}

/// Live statistics, updated by the render world. Read through [`WebcamShared::stats`].
#[derive(Debug, Clone, Default)]
pub struct WebcamStats {
    pub mode: TransferMode,
    /// Frames delivered to the GPU.
    pub frames: u64,
    /// Delivery rate, updated once a second.
    pub fps: f32,
    /// Time from the driver's timestamp to the GPU copy being issued, for the last frame.
    /// uvcvideo stamps the *start* of a frame, so this includes the frame's own transfer.
    pub last_latency: Duration,
    /// Frames the driver dropped for lack of free buffers.
    pub driver_dropped: u64,
    /// Frames the driver delivered damaged (USB packet loss).
    pub damaged: u64,
    /// Frames dequeued but skipped because a newer one was already available.
    pub skipped: u64,
    /// Duration of the last CPU cache write-back on the capture thread.
    pub flush_time: Duration,
    /// Duration of the last JPEG decode (MJPEG mode only).
    pub decode_time: Duration,
    /// JPEG frames that failed to decode.
    pub decode_failed: u64,
    pub error: Option<String>,
}

impl WebcamStats {
    /// One line summarising the stream, for an overlay.
    pub fn describe(&self, layout: &FrameLayout) -> String {
        let extra = match self.mode {
            TransferMode::MjpegDecode => {
                format!(
                    "decode {:.1} ms  failed {}",
                    self.decode_time.as_secs_f64() * 1e3,
                    self.decode_failed
                )
            }
            _ => format!("cache flush {:.0} us", self.flush_time.as_secs_f64() * 1e6),
        };
        let mut out = format!(
            "{}x{} {} @ {:.1} fps  |  path: {}  |  capture->GPU {:.1} ms  |  driver drops {}  damaged {}  skipped {}  |  {}",
            layout.width,
            layout.height,
            layout.fourcc,
            self.fps,
            self.mode.label(),
            self.last_latency.as_secs_f64() * 1e3,
            self.driver_dropped,
            self.damaged,
            self.skipped,
            extra
        );
        if let Some(e) = &self.error {
            out.push_str(&format!("  |  import error: {e}"));
        }
        out
    }
}

/// Shared between the main world and the render world once the camera is running.
#[derive(Resource, Clone)]
pub struct WebcamShared {
    pub capture: Arc<Capture>,
    /// The texture the material samples. Raw formats: `Rgba8Unorm`, half the frame width, one
    /// texel per pixel pair. MJPEG: `Rgba8UnormSrgb`, full width.
    pub image: Handle<Image>,
    pub layout: FrameLayout,
    pub stats: Arc<Mutex<WebcamStats>>,
    #[cfg(feature = "mjpeg")]
    pub decoder: Option<Arc<Decoder>>,
    force_upload: bool,
    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    verify_frames: u32,
    #[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
    no_barrier: bool,
}

impl ExtractResource for WebcamShared {
    type Source = WebcamShared;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Uniform for [`WebcamMaterial`].
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct WebcamParams {
    pub width: u32,
    pub height: u32,
    pub flags: u32,
    pub _pad: u32,
}

pub const FLAG_MIRROR: u32 = 1;
pub const FLAG_UYVY: u32 = 2;
pub const FLAG_FULL_RANGE: u32 = 4;
/// Texture already holds RGBA; sample it directly.
pub const FLAG_RGBA: u32 = 8;

/// Unlit material that converts the camera's native pixel format in the fragment shader.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct WebcamMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub frame: Handle<Image>,
    #[uniform(2)]
    pub params: WebcamParams,
}

impl Material for WebcamMaterial {
    fn fragment_shader() -> ShaderRef {
        "embedded://bevy_v4l2/webcam.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Visible from both sides.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Marker for the plane entity spawned by the plugin.
#[derive(Component)]
pub struct WebcamPlane;

fn start_webcam(
    mut commands: Commands,
    settings: Res<WebcamSettings>,
    mut status: ResMut<WebcamStatus>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<WebcamMaterial>>,
) {
    let fail = |status: &mut WebcamStatus, msg: String| {
        error!("{msg}");
        *status = WebcamStatus::Failed(msg);
    };
    let capture = match Capture::open(&settings.config) {
        Ok(c) => c,
        Err(e) => return fail(&mut status, format!("webcam unavailable: {e}")),
    };
    let layout = capture.layout();
    let fourcc = layout.fourcc;
    let mjpeg = fourcc == FOURCC_MJPG;
    if mjpeg && !cfg!(feature = "mjpeg") {
        return fail(
            &mut status,
            "stream is MJPEG but the `mjpeg` feature is disabled".into(),
        );
    }
    if !mjpeg && fourcc != FOURCC_YUYV && fourcc != FOURCC_UYVY {
        return fail(
            &mut status,
            format!("unsupported pixel format {fourcc}; only YUYV, UYVY and MJPG are handled"),
        );
    }
    if !mjpeg && layout.width % 2 != 0 {
        return fail(
            &mut status,
            format!(
                "frame width {} is odd; packed 4:2:2 needs an even width",
                layout.width
            ),
        );
    }

    // Raw 4:2:2: one texel per pixel pair (Y0, U, Y1, V). MJPEG: decoded sRGB RGBA, full width.
    let (tex_width, format) = if mjpeg {
        (layout.width, TextureFormat::Rgba8UnormSrgb)
    } else {
        (layout.width / 2, TextureFormat::Rgba8Unorm)
    };
    let size = Extent3d {
        width: tex_width,
        height: layout.height,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_uninit(
        size,
        TextureDimension::D2,
        format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
    image.sampler = if mjpeg {
        ImageSampler::linear()
    } else {
        ImageSampler::nearest()
    };
    let image = images.add(image);

    let mut flags = 0;
    if mjpeg {
        flags |= FLAG_RGBA;
    }
    if settings.mirror {
        flags |= FLAG_MIRROR;
    }
    if fourcc == FOURCC_UYVY {
        flags |= FLAG_UYVY;
    }
    if layout.full_range {
        flags |= FLAG_FULL_RANGE;
    }

    if settings.spawn_plane {
        let material = materials.add(WebcamMaterial {
            frame: image.clone(),
            params: WebcamParams {
                width: layout.width,
                height: layout.height,
                flags,
                _pad: 0,
            },
        });
        let height = settings.plane_height;
        let width = height * layout.width as f32 / layout.height as f32;
        commands.spawn((
            WebcamPlane,
            Mesh3d(meshes.add(Rectangle::new(width, height))),
            MeshMaterial3d(material),
            settings.plane_transform,
        ));
    }

    let capture = Arc::new(capture);
    #[cfg(feature = "mjpeg")]
    let decoder = mjpeg.then(|| {
        Arc::new(Decoder::start(
            Arc::clone(&capture),
            layout.width,
            layout.height,
        ))
    });
    commands.insert_resource(WebcamShared {
        capture,
        image,
        layout,
        stats: Arc::new(Mutex::new(WebcamStats::default())),
        #[cfg(feature = "mjpeg")]
        decoder,
        force_upload: settings.force_upload,
        verify_frames: settings.verify_frames,
        no_barrier: settings.no_barrier,
    });
    *status = WebcamStatus::Running;
    info!(
        "webcam running: {}x{} {}",
        layout.width, layout.height, fourcc
    );
}

/// Render-world state: imported textures, one per V4L2 buffer.
#[derive(Resource, Default)]
#[cfg_attr(not(feature = "dmabuf"), allow(dead_code))]
struct WebcamGpu {
    mode: TransferMode,
    imported: Vec<wgpu::Texture>,
    fps_window_start: Option<Instant>,
    fps_window_frames: u32,
    verified: u32,
}

impl WebcamGpu {
    fn count_frame(&mut self, stats: &mut WebcamStats) {
        let now = Instant::now();
        self.fps_window_frames += 1;
        stats.frames += 1;
        match self.fps_window_start {
            None => self.fps_window_start = Some(now),
            Some(start) if now - start >= Duration::from_secs(1) => {
                stats.fps = self.fps_window_frames as f32 / (now - start).as_secs_f32();
                self.fps_window_start = Some(now);
                self.fps_window_frames = 0;
            }
            _ => {}
        }
    }
}

fn upload_webcam_frame(
    shared: Option<Res<WebcamShared>>,
    mut gpu: ResMut<WebcamGpu>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    images: Res<RenderAssets<GpuImage>>,
    #[cfg(feature = "dmabuf")] features: Option<
        Res<bevy::render::renderer::raw_vulkan_init::AdditionalVulkanFeatures>,
    >,
) {
    let Some(shared) = shared else { return };
    let Some(gpu_image) = images.get(&shared.image) else {
        return;
    };

    if gpu.mode == TransferMode::Unknown {
        gpu.mode = decide_mode(
            &mut gpu,
            &shared,
            &device,
            #[cfg(feature = "dmabuf")]
            features.as_deref(),
        );
        shared.stats.lock().unwrap().mode = gpu.mode;
    }

    #[cfg(feature = "mjpeg")]
    if let Some(decoder) = shared.decoder.as_ref() {
        let Some(decoded) = decoder.try_recv_latest() else {
            return;
        };
        let layout = shared.layout;
        queue.write_texture(
            gpu_image.texture.as_image_copy(),
            &decoded.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(layout.width * 4),
                rows_per_image: None,
            },
            Extent3d {
                width: layout.width,
                height: layout.height,
                depth_or_array_layers: 1,
            },
        );
        let latency = crate::capture::monotonic_now().saturating_sub(decoded.timestamp);
        decoder.recycle(decoded.rgba);
        let mut stats = shared.stats.lock().unwrap();
        stats.last_latency = latency;
        stats.decode_time = decoded.decode_time;
        stats.decode_failed = decoder.failed();
        stats.driver_dropped = shared.capture.driver_dropped_count();
        stats.damaged = shared.capture.damaged_count();
        stats.skipped = decoder.skipped();
        gpu.count_frame(&mut stats);
        return;
    }

    let capture = &shared.capture;
    let before = capture.dequeued_count();
    let Some(frame) = capture.try_recv_latest() else {
        return;
    };
    let latency = frame.age();
    let index = frame.index as usize;
    let layout = shared.layout;
    let size = Extent3d {
        width: layout.width / 2,
        height: layout.height,
        depth_or_array_layers: 1,
    };

    match gpu.mode {
        #[cfg(feature = "dmabuf")]
        TransferMode::DmabufZeroCopy => {
            use crate::dmabuf::{ash::vk, record_foreign_acquire, record_foreign_release};
            let Some(src) = gpu.imported.get(index) else {
                error!("webcam: frame index {index} has no imported texture");
                capture.requeue(frame.index);
                return;
            };
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("webcam dmabuf copy"),
            });
            let layout_vk = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            if !shared.no_barrier {
                record_foreign_acquire(&mut encoder, device.wgpu_device(), src, layout_vk);
            }
            encoder.copy_texture_to_texture(
                src.as_image_copy(),
                gpu_image.texture.as_image_copy(),
                size,
            );
            if !shared.no_barrier {
                record_foreign_release(&mut encoder, device.wgpu_device(), src, layout_vk);
            }
            if gpu.verified < shared.verify_frames {
                gpu.verified += 1;
                verify_copy(
                    &device,
                    &queue,
                    encoder,
                    &gpu_image.texture,
                    size,
                    shared.as_ref(),
                    &frame,
                );
                capture.requeue(frame.index);
            } else {
                queue.submit([encoder.finish()]);
                // Hand the buffer back only once the GPU has finished reading it.
                let capture = Arc::clone(capture);
                queue.on_submitted_work_done(move || capture.requeue(frame.index));
            }
        }
        TransferMode::CpuUpload => {
            let buffer = &capture.buffers()[index];
            let data = &buffer.as_slice()[..(layout.stride * layout.height) as usize];
            queue.write_texture(
                gpu_image.texture.as_image_copy(),
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(layout.stride),
                    rows_per_image: None,
                },
                size,
            );
            // write_texture copies into staging memory synchronously, so the buffer is free now.
            capture.requeue(frame.index);
        }
        _ => unreachable!("mode decided above"),
    }

    let mut stats = shared.stats.lock().unwrap();
    stats.last_latency = latency;
    stats.driver_dropped = capture.driver_dropped_count();
    stats.damaged = capture.damaged_count();
    stats.flush_time = capture.last_flush_time();
    let consumed = capture.dequeued_count().saturating_sub(before);
    stats.skipped += consumed.saturating_sub(1);
    gpu.count_frame(&mut stats);
}

fn decide_mode(
    gpu: &mut WebcamGpu,
    shared: &WebcamShared,
    device: &RenderDevice,
    #[cfg(feature = "dmabuf")] features: Option<
        &bevy::render::renderer::raw_vulkan_init::AdditionalVulkanFeatures,
    >,
) -> TransferMode {
    #[cfg(feature = "mjpeg")]
    if shared.decoder.is_some() {
        info!("webcam: MJPEG stream, decoding on CPU thread");
        return TransferMode::MjpegDecode;
    }
    if shared.force_upload {
        info!("webcam: CPU upload forced by configuration");
        return TransferMode::CpuUpload;
    }
    #[cfg(feature = "dmabuf")]
    {
        if !crate::dmabuf::import_supported(features, device) {
            warn!("webcam: DMA-BUF import not supported by this device; using CPU upload");
            return TransferMode::CpuUpload;
        }
        match import_all(device, shared) {
            Ok(textures) => {
                gpu.imported = textures;
                info!(
                    "webcam: {} DMA-BUFs imported, zero-copy path active",
                    gpu.imported.len()
                );
                return TransferMode::DmabufZeroCopy;
            }
            Err(e) => {
                warn!("webcam: DMA-BUF import failed ({e}); using CPU upload");
                shared.stats.lock().unwrap().error = Some(e.to_string());
            }
        }
    }
    #[cfg(not(feature = "dmabuf"))]
    {
        let _ = (gpu, device);
        info!("webcam: built without the `dmabuf` feature; using CPU upload");
    }
    TransferMode::CpuUpload
}

#[cfg(feature = "dmabuf")]
fn import_all(
    device: &RenderDevice,
    shared: &WebcamShared,
) -> Result<Vec<wgpu::Texture>, crate::dmabuf::DmabufImportError> {
    use crate::dmabuf::{
        DRM_FORMAT_MOD_LINEAR, DmabufImageDesc, DmabufPlane, import_dmabuf_texture,
    };
    use std::os::fd::AsFd;
    let layout = shared.layout;
    shared
        .capture
        .buffers()
        .iter()
        .map(|buffer| {
            import_dmabuf_texture(
                device.wgpu_device(),
                &DmabufImageDesc {
                    label: Some("webcam dmabuf"),
                    width: layout.width / 2,
                    height: layout.height,
                    format: TextureFormat::Rgba8Unorm,
                    modifier: DRM_FORMAT_MOD_LINEAR,
                    plane: DmabufPlane {
                        fd: buffer.dmabuf.as_fd(),
                        offset: buffer.offset,
                        stride: layout.stride,
                    },
                    usage: TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING,
                },
            )
        })
        .collect()
}

/// Diagnostic: read the GPU-side copy back and diff it against the CPU view of the same kernel
/// buffer. Any mismatch means the GPU read stale or torn data.
#[cfg(feature = "dmabuf")]
fn verify_copy(
    device: &RenderDevice,
    queue: &RenderQueue,
    mut encoder: wgpu::CommandEncoder,
    dst: &wgpu::Texture,
    size: Extent3d,
    shared: &WebcamShared,
    frame: &DequeuedFrame,
) {
    let layout = shared.layout;
    let bytes_per_row = layout.stride;
    if !bytes_per_row.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) {
        warn!("verify: stride {bytes_per_row} not 256-aligned, skipping");
        return;
    }
    let total = (bytes_per_row * layout.height) as u64;
    let readback = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("webcam verify readback"),
        size: total,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        dst.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    if let Err(e) = device
        .wgpu_device()
        .poll(wgpu::PollType::wait_indefinitely())
    {
        warn!("verify: poll failed: {e:?}");
        return;
    }
    let gpu_bytes = slice.get_mapped_range();
    let cpu_bytes = &shared.capture.buffers()[frame.index as usize].as_slice()[..total as usize];
    let mut bad_lines = 0u64;
    let mut bad_rows = std::collections::BTreeSet::new();
    for (i, (a, b)) in gpu_bytes.chunks(64).zip(cpu_bytes.chunks(64)).enumerate() {
        if a != b {
            bad_lines += 1;
            bad_rows.insert(i as u64 * 64 / bytes_per_row as u64);
        }
    }
    drop(gpu_bytes);
    readback.unmap();
    let total_lines = total / 64;
    if bad_lines == 0 {
        info!(
            "verify: frame seq {} buffer {}: GPU copy matches kernel buffer ({total_lines} cache lines)",
            frame.sequence, frame.index
        );
    } else {
        warn!(
            "verify: frame seq {} buffer {}: {bad_lines}/{total_lines} stale 64-byte lines across {} rows (first rows {:?})",
            frame.sequence,
            frame.index,
            bad_rows.len(),
            bad_rows.iter().take(8).collect::<Vec<_>>()
        );
    }
}
