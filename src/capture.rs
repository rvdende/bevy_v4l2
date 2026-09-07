//! Low-latency V4L2 capture with DMA-BUF export.
//!
//! The kernel allocates the frame buffers (`V4L2_MEMORY_MMAP`), each buffer is exported as a
//! DMA-BUF file descriptor with `VIDIOC_EXPBUF`, and the descriptors are handed to the caller so
//! they can be imported directly into a GPU API (Vulkan `VK_EXT_external_memory_dma_buf`).
//! Frames therefore never pass through a userspace copy.
//!
//! A background thread blocks in `poll()` on the device and dequeues buffers as soon as the
//! driver marks them done. Buffers are returned to the driver with [`Capture::requeue`], which is
//! safe to call from any thread.

use std::{
    ffi::c_void,
    io,
    os::fd::{FromRawFd, OwnedFd},
    path::PathBuf,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use v4l::{
    Device, FourCC,
    buffer::Type as BufType,
    format::Format,
    memory::Memory,
    v4l_sys::{v4l2_buffer, v4l2_exportbuffer, v4l2_requestbuffers},
    v4l2,
    video::Capture as _,
};

pub use v4l::{FourCC as V4lFourCC, format::Format as V4lFormat};

/// FourCC for packed YUV 4:2:2, byte order `Y0 U0 Y1 V0`.
pub const FOURCC_YUYV: FourCC = FourCC { repr: *b"YUYV" };
/// FourCC for packed YUV 4:2:2, byte order `U0 Y0 V0 Y1`.
pub const FOURCC_UYVY: FourCC = FourCC { repr: *b"UYVY" };
/// FourCC for Motion-JPEG: one JPEG bitstream per frame.
pub const FOURCC_MJPG: FourCC = FourCC { repr: *b"MJPG" };
/// FourCC for two-plane 4:2:0 (Y plane, interleaved UV plane).
pub const FOURCC_NV12: FourCC = FourCC { repr: *b"NV12" };

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("failed to open {path}: {source}")]
    Open { path: PathBuf, source: io::Error },
    #[error("device does not support video capture")]
    NotCapture,
    #[error("device does not support streaming I/O (MMAP)")]
    NoStreaming,
    #[error("driver refused pixel format {requested}; it offers: {available}")]
    Format {
        requested: FourCC,
        available: String,
    },
    #[error("driver granted {0} buffers, need at least 2")]
    TooFewBuffers(u32),
    #[error("{op} failed: {source}")]
    Ioctl { op: &'static str, source: io::Error },
    #[error("mmap of buffer {index} failed: {source}")]
    Mmap { index: u32, source: io::Error },
}

fn ioctl_err(op: &'static str) -> impl FnOnce(io::Error) -> CaptureError {
    move |source| CaptureError::Ioctl { op, source }
}

/// What to open and how to configure it.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub device: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fourcc: FourCC,
    /// Requested frame rate. `None` keeps the driver default.
    pub fps: Option<u32>,
    /// Number of kernel buffers to request. More buffers tolerate GPU stalls; fewer keep the
    /// pipeline shallow. The consumer always takes the newest frame, so depth does not add
    /// latency by itself.
    pub buffer_count: u32,
    /// Write back the CPU cache for each frame after dequeue, so a GPU reading the DMA-BUF
    /// without snooping sees the driver's writes rather than stale DRAM. Costs ~50-100 µs per
    /// 1080p frame on the capture thread. Only matters for the zero-copy path.
    pub flush_cpu_cache: bool,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            device: PathBuf::from("/dev/video0"),
            width: 1280,
            height: 720,
            fourcc: FOURCC_YUYV,
            fps: Some(30),
            buffer_count: 4,
            flush_cpu_cache: true,
        }
    }
}

/// Negotiated stream layout, as reported by the driver after `VIDIOC_S_FMT`.
#[derive(Debug, Clone, Copy)]
pub struct FrameLayout {
    pub width: u32,
    pub height: u32,
    pub fourcc: FourCC,
    /// Bytes per row, including padding.
    pub stride: u32,
    /// Total bytes per frame.
    pub size: u32,
    /// Whether the driver reports full-range (0..255) rather than limited-range (16..235) YUV.
    pub full_range: bool,
    /// Frames are variable-length bitstreams (MJPEG) rather than fixed-size pixel arrays.
    pub compressed: bool,
}

/// One kernel frame buffer, exported as a DMA-BUF and also mapped into this process.
pub struct ExportedBuffer {
    pub index: u32,
    /// DMA-BUF file descriptor. Importers such as Vulkan take ownership of a *duplicate*; keep
    /// this one alive for the lifetime of the stream.
    pub dmabuf: OwnedFd,
    /// Byte length of the buffer as reported by `VIDIOC_QUERYBUF`.
    pub length: usize,
    /// Offset of the pixel data inside the DMA-BUF. Always 0 for single-plane MMAP buffers.
    pub offset: u64,
    mapping: NonNull<c_void>,
}

// SAFETY: the mapping is read-only from our side and the kernel owns synchronisation.
unsafe impl Send for ExportedBuffer {}
unsafe impl Sync for ExportedBuffer {}

impl ExportedBuffer {
    /// CPU view of the buffer for fallback upload paths. Only meaningful between dequeue and
    /// requeue of this buffer index.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: mapping is valid for `length` bytes until Drop.
        unsafe { std::slice::from_raw_parts(self.mapping.as_ptr() as *const u8, self.length) }
    }
}

impl Drop for ExportedBuffer {
    fn drop(&mut self) {
        // SAFETY: we created this mapping with mmap of exactly `length` bytes.
        let _ = unsafe { v4l2::munmap(self.mapping.as_ptr(), self.length) };
    }
}

/// A frame the driver has finished writing.
#[derive(Debug, Clone, Copy)]
pub struct DequeuedFrame {
    /// Index into [`Capture::buffers`].
    pub index: u32,
    /// Driver sequence counter; gaps mean the driver dropped frames.
    pub sequence: u32,
    /// Bytes the driver wrote.
    pub bytes_used: u32,
    /// Driver timestamp on `CLOCK_MONOTONIC`.
    pub timestamp: Duration,
    /// The driver flagged this frame as damaged (`V4L2_BUF_FLAG_ERROR`), typically USB packet
    /// loss. Rows not received keep whatever the buffer held before.
    pub error: bool,
}

impl DequeuedFrame {
    /// Time elapsed since the driver timestamped this frame.
    pub fn age(&self) -> Duration {
        monotonic_now().saturating_sub(self.timestamp)
    }
}

/// Current `CLOCK_MONOTONIC` reading, comparable with [`DequeuedFrame::timestamp`].
pub fn monotonic_now() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid pointer to a timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

struct Shared {
    device: Device,
    layout: FrameLayout,
    buffers: Vec<ExportedBuffer>,
    stop: AtomicBool,
    flush_cpu_cache: bool,
    flush_nanos: AtomicU64,
    dequeued: AtomicU64,
    dropped_by_driver: AtomicU64,
    damaged: AtomicU64,
}

impl Shared {
    fn fd(&self) -> i32 {
        self.device.handle().fd()
    }

    fn qbuf(&self, index: u32) -> io::Result<()> {
        let mut buf: v4l2_buffer = unsafe { std::mem::zeroed() };
        buf.type_ = BufType::VideoCapture as u32;
        buf.memory = Memory::Mmap as u32;
        buf.index = index;
        // SAFETY: valid v4l2_buffer for QBUF.
        unsafe {
            v4l2::ioctl(
                self.fd(),
                v4l2::vidioc::VIDIOC_QBUF,
                &mut buf as *mut _ as *mut c_void,
            )
        }
    }

    fn dqbuf(&self) -> io::Result<DequeuedFrame> {
        let mut buf: v4l2_buffer = unsafe { std::mem::zeroed() };
        buf.type_ = BufType::VideoCapture as u32;
        buf.memory = Memory::Mmap as u32;
        // SAFETY: valid v4l2_buffer for DQBUF.
        unsafe {
            v4l2::ioctl(
                self.fd(),
                v4l2::vidioc::VIDIOC_DQBUF,
                &mut buf as *mut _ as *mut c_void,
            )?
        };
        const V4L2_BUF_FLAG_ERROR: u32 = 0x0000_0040;
        let short = !self.layout.compressed && buf.bytesused < self.layout.size;
        Ok(DequeuedFrame {
            index: buf.index,
            sequence: buf.sequence,
            bytes_used: buf.bytesused,
            timestamp: Duration::new(
                buf.timestamp.tv_sec as u64,
                (buf.timestamp.tv_usec as u32) * 1000,
            ),
            error: buf.flags & V4L2_BUF_FLAG_ERROR != 0 || short,
        })
    }

    fn stream(&self, on: bool) -> io::Result<()> {
        let mut ty = BufType::VideoCapture as i32;
        let req = if on {
            v4l2::vidioc::VIDIOC_STREAMON
        } else {
            v4l2::vidioc::VIDIOC_STREAMOFF
        };
        // SAFETY: STREAMON/OFF take a pointer to the buffer type.
        unsafe { v4l2::ioctl(self.fd(), req, &mut ty as *mut _ as *mut c_void) }
    }
}

/// A running capture stream. Dropping it stops streaming and releases the buffers.
pub struct Capture {
    shared: Arc<Shared>,
    frames: Receiver<DequeuedFrame>,
    thread: Option<JoinHandle<()>>,
}

impl Capture {
    /// Open the device, negotiate the format, export the buffers and start streaming.
    pub fn open(config: &CaptureConfig) -> Result<Self, CaptureError> {
        let device = Device::with_path(&config.device).map_err(|source| CaptureError::Open {
            path: config.device.clone(),
            source,
        })?;
        let caps = device.query_caps().map_err(ioctl_err("VIDIOC_QUERYCAP"))?;
        tracing::info!(
            driver = %caps.driver, card = %caps.card, bus = %caps.bus, "opened {}", config.device.display()
        );
        if !caps
            .capabilities
            .contains(v4l::capability::Flags::VIDEO_CAPTURE)
        {
            return Err(CaptureError::NotCapture);
        }
        if !caps
            .capabilities
            .contains(v4l::capability::Flags::STREAMING)
        {
            return Err(CaptureError::NoStreaming);
        }

        let requested = Format::new(config.width, config.height, config.fourcc);
        let granted = device
            .set_format(&requested)
            .map_err(ioctl_err("VIDIOC_S_FMT"))?;
        if granted.fourcc != config.fourcc {
            let available = device
                .enum_formats()
                .map(|fs| {
                    fs.iter()
                        .map(|f| f.fourcc.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_else(|_| "<unknown>".into());
            return Err(CaptureError::Format {
                requested: config.fourcc,
                available,
            });
        }
        if let Some(fps) = config.fps {
            let params = v4l::video::capture::Parameters::with_fps(fps);
            match device.set_params(&params) {
                Ok(p) => tracing::info!(interval = ?p.interval, "frame interval set"),
                Err(e) => tracing::warn!("VIDIOC_S_PARM failed, keeping driver default: {e}"),
            }
        }
        let layout = FrameLayout {
            width: granted.width,
            height: granted.height,
            fourcc: granted.fourcc,
            stride: granted.stride,
            size: granted.size,
            full_range: matches!(
                granted.quantization,
                v4l::format::quantization::Quantization::FullRange
            ),
            compressed: granted.fourcc == FOURCC_MJPG,
        };
        tracing::info!(?layout, "negotiated format");

        let fd = device.handle().fd();

        // Request kernel-allocated buffers.
        let mut req: v4l2_requestbuffers = unsafe { std::mem::zeroed() };
        req.count = config.buffer_count.max(2);
        req.type_ = BufType::VideoCapture as u32;
        req.memory = Memory::Mmap as u32;
        // SAFETY: valid request struct.
        unsafe {
            v4l2::ioctl(
                fd,
                v4l2::vidioc::VIDIOC_REQBUFS,
                &mut req as *mut _ as *mut c_void,
            )
        }
        .map_err(ioctl_err("VIDIOC_REQBUFS"))?;
        if req.count < 2 {
            return Err(CaptureError::TooFewBuffers(req.count));
        }

        let mut buffers = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            let mut buf: v4l2_buffer = unsafe { std::mem::zeroed() };
            buf.type_ = BufType::VideoCapture as u32;
            buf.memory = Memory::Mmap as u32;
            buf.index = index;
            // SAFETY: valid v4l2_buffer for QUERYBUF.
            unsafe {
                v4l2::ioctl(
                    fd,
                    v4l2::vidioc::VIDIOC_QUERYBUF,
                    &mut buf as *mut _ as *mut c_void,
                )
            }
            .map_err(ioctl_err("VIDIOC_QUERYBUF"))?;
            let length = buf.length as usize;
            // SAFETY: union field `offset` is the valid member for MMAP buffers.
            let mmap_offset = unsafe { buf.m.offset } as libc::off_t;

            let mut exp: v4l2_exportbuffer = unsafe { std::mem::zeroed() };
            exp.type_ = BufType::VideoCapture as u32;
            exp.index = index;
            exp.plane = 0;
            exp.flags = libc::O_RDONLY as u32 | libc::O_CLOEXEC as u32;
            // SAFETY: valid v4l2_exportbuffer for EXPBUF.
            unsafe {
                v4l2::ioctl(
                    fd,
                    v4l2::vidioc::VIDIOC_EXPBUF,
                    &mut exp as *mut _ as *mut c_void,
                )
            }
            .map_err(ioctl_err("VIDIOC_EXPBUF"))?;
            // SAFETY: EXPBUF returned a fresh fd that we now own.
            let dmabuf = unsafe { OwnedFd::from_raw_fd(exp.fd) };

            // SAFETY: standard V4L2 MMAP of a queried buffer.
            let mapping = unsafe {
                v4l2::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd,
                    mmap_offset,
                )
            }
            .map_err(|source| CaptureError::Mmap { index, source })?;
            let mapping = NonNull::new(mapping).expect("mmap returned null");

            buffers.push(ExportedBuffer {
                index,
                dmabuf,
                length,
                offset: 0,
                mapping,
            });
        }

        let shared = Arc::new(Shared {
            device,
            layout,
            buffers,
            stop: AtomicBool::new(false),
            flush_cpu_cache: config.flush_cpu_cache,
            flush_nanos: AtomicU64::new(0),
            dequeued: AtomicU64::new(0),
            dropped_by_driver: AtomicU64::new(0),
            damaged: AtomicU64::new(0),
        });

        for i in 0..req.count {
            shared.qbuf(i).map_err(ioctl_err("VIDIOC_QBUF"))?;
        }
        shared.stream(true).map_err(ioctl_err("VIDIOC_STREAMON"))?;

        // Capacity equals buffer count so the poll thread never blocks on a slow consumer.
        let (tx, rx) = crossbeam_channel::bounded(req.count as usize);
        let thread = {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("v4l2-capture".into())
                .spawn(move || poll_loop(shared, tx))
                .expect("failed to spawn capture thread")
        };

        Ok(Self {
            shared,
            frames: rx,
            thread: Some(thread),
        })
    }

    pub fn layout(&self) -> FrameLayout {
        self.shared.layout
    }

    pub fn buffers(&self) -> &[ExportedBuffer] {
        &self.shared.buffers
    }

    /// Receiver for dequeued frames, in driver order.
    pub fn frames(&self) -> &Receiver<DequeuedFrame> {
        &self.frames
    }

    /// Take the newest available frame, returning any older pending frames to the driver.
    /// Returns `None` when no new frame has arrived since the last call.
    pub fn try_recv_latest(&self) -> Option<DequeuedFrame> {
        let mut latest = None;
        while let Ok(frame) = self.frames.try_recv() {
            if let Some(prev) = latest.replace(frame) {
                self.requeue(prev.index);
            }
        }
        latest
    }

    /// Give a buffer back to the driver. Call once per frame received, after the GPU has finished
    /// reading it. Safe to call from any thread.
    pub fn requeue(&self, index: u32) {
        if self.shared.stop.load(Ordering::Relaxed) {
            return;
        }
        if let Err(e) = self.shared.qbuf(index) {
            tracing::error!("VIDIOC_QBUF({index}) failed: {e}");
        }
    }

    /// Frames dequeued so far.
    pub fn dequeued_count(&self) -> u64 {
        self.shared.dequeued.load(Ordering::Relaxed)
    }

    /// Frames the driver reported dropping (sequence gaps), usually from buffer starvation.
    pub fn driver_dropped_count(&self) -> u64 {
        self.shared.dropped_by_driver.load(Ordering::Relaxed)
    }

    /// Frames delivered damaged (error flag or short), usually USB packet loss.
    pub fn damaged_count(&self) -> u64 {
        self.shared.damaged.load(Ordering::Relaxed)
    }

    /// Time the last CPU cache flush took, if enabled.
    pub fn last_flush_time(&self) -> Duration {
        Duration::from_nanos(self.shared.flush_nanos.load(Ordering::Relaxed))
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        if let Err(e) = self.shared.stream(false) {
            tracing::warn!("VIDIOC_STREAMOFF failed: {e}");
        }
    }
}

fn poll_loop(shared: Arc<Shared>, tx: Sender<DequeuedFrame>) {
    let fd = shared.fd();
    let mut last_sequence: Option<u32> = None;
    while !shared.stop.load(Ordering::Relaxed) {
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: valid pollfd array of length 1.
        let n = unsafe { libc::poll(&mut pfd, 1, 100) };
        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            tracing::error!("poll on capture device failed: {err}");
            break;
        }
        if n == 0 {
            continue;
        }
        if pfd.revents & libc::POLLERR != 0 {
            tracing::error!("capture device reported POLLERR; stopping");
            break;
        }
        // Drain everything that is ready so the consumer can pick the newest frame.
        loop {
            match shared.dqbuf() {
                Ok(frame) => {
                    if shared.flush_cpu_cache {
                        let t0 = std::time::Instant::now();
                        let buffer = &shared.buffers[frame.index as usize];
                        let len = (frame.bytes_used as usize).min(buffer.length);
                        flush_cpu_cache(&buffer.as_slice()[..len]);
                        shared
                            .flush_nanos
                            .store(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    shared.dequeued.fetch_add(1, Ordering::Relaxed);
                    if frame.error {
                        shared.damaged.fetch_add(1, Ordering::Relaxed);
                    }
                    if let Some(prev) = last_sequence {
                        let gap = frame.sequence.wrapping_sub(prev).saturating_sub(1);
                        if gap > 0 {
                            shared
                                .dropped_by_driver
                                .fetch_add(gap as u64, Ordering::Relaxed);
                        }
                    }
                    last_sequence = Some(frame.sequence);
                    match tx.try_send(frame) {
                        Ok(()) => {}
                        Err(TrySendError::Full(f)) => {
                            // Consumer is stalled; recycle immediately rather than starve the driver.
                            let _ = shared.qbuf(f.index);
                        }
                        Err(TrySendError::Disconnected(_)) => return,
                    }
                }
                Err(e) if e.raw_os_error() == Some(libc::EAGAIN) => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    tracing::error!("VIDIOC_DQBUF failed: {e}");
                    return;
                }
            }
        }
    }
}

/// One supported capture mode.
#[derive(Debug, Clone)]
pub struct CaptureMode {
    pub fourcc: FourCC,
    pub description: String,
    pub width: u32,
    pub height: u32,
    /// Supported frame rates, highest first.
    pub fps: Vec<f32>,
}

/// Enumerate every (format, size, rate) the device offers.
pub fn list_modes(path: impl AsRef<std::path::Path>) -> io::Result<Vec<CaptureMode>> {
    let device = Device::with_path(path)?;
    let mut modes = Vec::new();
    for fmt in device.enum_formats()? {
        for size in device.enum_framesizes(fmt.fourcc)? {
            for (width, height) in size
                .size
                .to_discrete()
                .into_iter()
                .map(|d| (d.width, d.height))
            {
                let mut fps: Vec<f32> = device
                    .enum_frameintervals(fmt.fourcc, width, height)?
                    .into_iter()
                    .map(|i| match i.interval {
                        v4l::frameinterval::FrameIntervalEnum::Discrete(f) => {
                            f.denominator as f32 / f.numerator.max(1) as f32
                        }
                        v4l::frameinterval::FrameIntervalEnum::Stepwise(s) => {
                            s.min.denominator as f32 / s.min.numerator.max(1) as f32
                        }
                    })
                    .collect();
                fps.sort_by(|a, b| b.partial_cmp(a).unwrap());
                modes.push(CaptureMode {
                    fourcc: fmt.fourcc,
                    description: fmt.description.clone(),
                    width,
                    height,
                    fps,
                });
            }
        }
    }
    Ok(modes)
}

/// Write back every cache line covering `bytes` so a device reading the memory directly sees
/// the current contents. On x86 this uses `clflushopt` (or `clflush`) followed by `sfence`.
/// On other architectures it is a no-op; the kernel's DMA API is expected to handle coherence.
pub fn flush_cpu_cache(bytes: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{__cpuid_count, _mm_clflush, _mm_sfence};
        use std::sync::OnceLock;
        if bytes.is_empty() {
            return;
        }
        static HAS_CLFLUSHOPT: OnceLock<bool> = OnceLock::new();
        // CPUID.(EAX=7,ECX=0):EBX bit 23.
        let opt = *HAS_CLFLUSHOPT.get_or_init(|| __cpuid_count(7, 0).ebx & (1 << 23) != 0);
        let start = bytes.as_ptr() as usize & !63;
        let end = bytes.as_ptr() as usize + bytes.len();
        // SAFETY: clflush/clflushopt accept any address inside a mapped page and never fault on
        // readable memory; `bytes` is a live mapping.
        unsafe {
            let mut p = start;
            if opt {
                while p < end {
                    std::arch::asm!("clflushopt [{0}]", in(reg) p, options(nostack, preserves_flags));
                    p += 64;
                }
            } else {
                while p < end {
                    _mm_clflush(p as *const u8);
                    p += 64;
                }
            }
            _mm_sfence();
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = bytes;
    }
}
