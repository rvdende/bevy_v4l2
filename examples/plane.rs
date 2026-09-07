//! A webcam on a plane. Environment overrides: `WEBCAM_DEVICE`, `WEBCAM_WIDTH`, `WEBCAM_HEIGHT`,
//! `WEBCAM_FPS`, `WEBCAM_FORMAT` (any|yuyv|uyvy|mjpeg), `WEBCAM_VERIFY` (frames to check),
//! `WEBCAM_CALIBRATE=1` (run lens + rolling-shutter calibration and save `calibration.json`),
//! `WEBCAM_EXIT_AFTER` (seconds), `WEBCAM_HEADLESS=1` (no window; renders off-screen, handy for
//! CI or for verifying the capture path without disturbing the desktop).

use bevy::{app::ScheduleRunnerPlugin, prelude::*, window::ExitCondition, winit::WinitPlugin};
use bevy_v4l2::{
    Calibrate, CalibrationPlugin, CalibrationRun, CameraCalibration, CameraFormat,
    DmabufTexturePlugin, FrameFormat, RequestedFormat, Webcam, WebcamPlugin,
};

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let mut app = App::new();
    app.add_plugins(DmabufTexturePlugin);
    if env("WEBCAM_HEADLESS", 0) == 1 {
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<WinitPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                }),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_millis(2),
        ));
    } else {
        app.add_plugins(DefaultPlugins);
    }
    app.add_plugins((WebcamPlugin, CalibrationPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (exit_after, calibration_progress))
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
    let mut webcam = commands.spawn((
        Webcam {
            verify_frames: env("WEBCAM_VERIFY", 0),
            ..Webcam::new(RequestedFormat::Closest(wanted))
                .device(env::<String>("WEBCAM_DEVICE", "/dev/video0".into()))
        },
        Transform::from_xyz(0.0, 0.45, 0.0),
    ));
    if env("WEBCAM_CALIBRATE", 0) == 1 {
        webcam.insert(Calibrate::default());
    }
}

fn exit_after(time: Res<Time>, mut exit: MessageWriter<AppExit>) {
    let limit: f32 = env("WEBCAM_EXIT_AFTER", 0.0);
    if limit > 0.0 && time.elapsed_secs() > limit {
        exit.write(AppExit::Success);
    }
}

/// Logs calibration progress once a second and saves the result when it lands.
fn calibration_progress(
    time: Res<Time>,
    runs: Query<&CalibrationRun>,
    done: Query<&CameraCalibration, Added<CameraCalibration>>,
    mut next: Local<f32>,
) {
    for c in &done {
        let (h, v) = c.fov_deg();
        info!(
            "calibration: fov {h:.1}x{v:.1} deg, reproj {:.3} px, rolling shutter {:?}",
            c.reproj_error_px, c.rolling_shutter
        );
        if let Err(e) = c.save("calibration.json") {
            warn!("could not save calibration.json: {e}");
        } else {
            info!("saved calibration.json");
        }
    }
    if time.elapsed_secs() < *next {
        return;
    }
    *next = time.elapsed_secs() + 1.0;
    for run in &runs {
        let p = run.progress();
        info!(
            "calibration: {:?} - {} ({} corners)",
            p.phase,
            p.hint,
            p.detection.as_ref().map_or(0, |d| d.corners.len())
        );
    }
}
