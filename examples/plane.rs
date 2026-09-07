//! A webcam on a plane. Environment overrides: `WEBCAM_DEVICE`, `WEBCAM_WIDTH`, `WEBCAM_HEIGHT`,
//! `WEBCAM_FPS`, `WEBCAM_FORMAT` (any|yuyv|uyvy|mjpeg), `WEBCAM_VERIFY` (frames to check),
//! `WEBCAM_EXIT_AFTER` (seconds).

use bevy::prelude::*;
use bevy_v4l2::{
    CameraFormat, DmabufTexturePlugin, FrameFormat, RequestedFormat, Webcam, WebcamPlugin,
};

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    App::new()
        .add_plugins(DmabufTexturePlugin)
        .add_plugins(DefaultPlugins)
        .add_plugins(WebcamPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, exit_after)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.8, 2.0).looking_at(Vec3::new(0.0, 0.45, 0.0), Vec3::Y),
    ));
    let format = match env::<String>("WEBCAM_FORMAT", "any".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "yuyv" => FrameFormat::Yuyv,
        "uyvy" => FrameFormat::Uyvy,
        "mjpeg" | "mjpg" => FrameFormat::Mjpeg,
        _ => FrameFormat::Any,
    };
    let wanted = CameraFormat::new(
        env("WEBCAM_WIDTH", 1280),
        env("WEBCAM_HEIGHT", 720),
        format,
        env("WEBCAM_FPS", 60),
    );
    commands.spawn((
        Webcam {
            verify_frames: env("WEBCAM_VERIFY", 0),
            ..Webcam::new(RequestedFormat::Closest(wanted))
                .device(env::<String>("WEBCAM_DEVICE", "/dev/video0".into()))
        },
        Transform::from_xyz(0.0, 0.45, 0.0),
    ));
}

fn exit_after(time: Res<Time>, mut exit: MessageWriter<AppExit>) {
    let limit: f32 = env("WEBCAM_EXIT_AFTER", 0.0);
    if limit > 0.0 && time.elapsed_secs() > limit {
        exit.write(AppExit::Success);
    }
}
