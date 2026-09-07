//! Low-latency V4L2 webcam capture for [Bevy](https://bevy.org).
//!
//! * [`capture`]: V4L2 streaming with kernel `MMAP` buffers exported as DMA-BUF file
//!   descriptors, newest-frame-wins delivery, and the CPU cache write-back that keeps a
//!   non-snooping GPU coherent with the driver's writes.
//! * [`dmabuf`] (feature `dmabuf`): enables the Vulkan extensions during Bevy's device creation
//!   and wraps a DMA-BUF as a `wgpu::Texture` with no copy.
//! * [`plugin`]: [`WebcamPlugin`], which puts the feed on a plane with a material whose shader
//!   converts packed YUV to RGB on the GPU; MJPEG streams are decoded on a thread
//!   (feature `mjpeg`).
//! * [`select`]: picks the best mode for a wanted size and rate, preferring raw (zero-copy).
//! * [`controls`]: device listing and V4L2 controls (brightness, focus, zoom, ...).
//!
//! ```no_run
//! use bevy::prelude::*;
//! use bevy_v4l2::{DmabufTexturePlugin, WebcamPlugin};
//!
//! App::new()
//!     // Must come before DefaultPlugins so the Vulkan device gets the DMA-BUF extensions.
//!     .add_plugins(DmabufTexturePlugin)
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(WebcamPlugin::want(1280, 720, 60.0))
//!     .run();
//! ```
//!
//! Linux only: everything here talks to `/dev/video*`.

#![cfg(target_os = "linux")]
#![recursion_limit = "256"]

pub mod capture;
pub mod controls;
#[cfg(feature = "dmabuf")]
pub mod dmabuf;
#[cfg(feature = "mjpeg")]
pub mod mjpeg;
pub mod plugin;
pub mod select;

pub use capture::*;
pub use controls::*;
#[cfg(feature = "dmabuf")]
pub use dmabuf::{DmabufImportEnabled, DmabufTexturePlugin};
pub use plugin::*;
pub use select::*;
pub use v4l::FourCC;
