use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

use bevy::prelude::*;

use super::{
    CameraCalibration, ChessboardSpec, Detection, LensCalibrator, RollingShutterCalibrator,
    SweepVerdict, ViewVerdict, detect,
};
use crate::plugin::WebcamFeed;

/// Drives [`Calibrate`] requests on [`crate::Webcam`] entities.
#[derive(Default)]
pub struct CalibrationPlugin;

impl Plugin for CalibrationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (start_runs, finish_runs));
    }
}

/// Insert on a webcam entity to calibrate it. Removed again when the run finishes; the result
/// is inserted as a [`CameraCalibration`] component and reported through [`CalibrationRun`].
#[derive(Component, Clone, Debug)]
pub struct Calibrate {
    pub board: ChessboardSpec,
    /// Distinct board views to collect before solving the lens.
    pub views_needed: usize,
    /// Accepted sweep frames to collect for the rolling-shutter estimate. `0` skips that step.
    pub sweep_frames_needed: usize,
}

impl Default for Calibrate {
    fn default() -> Self {
        Self {
            board: ChessboardSpec::default(),
            views_needed: 15,
            sweep_frames_needed: 40,
        }
    }
}

/// What the worker is doing.
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrationPhase {
    /// Collecting lens views: `accepted` of `needed`.
    Lens {
        accepted: usize,
        needed: usize,
    },
    Solving,
    /// Sweep the board sideways: `accepted` of `needed` moving frames.
    RollingShutter {
        accepted: usize,
        needed: usize,
    },
    Done,
    Failed(String),
}

/// Live snapshot for a UI.
#[derive(Debug, Clone)]
pub struct CalibrationProgress {
    pub phase: CalibrationPhase,
    /// Corners found in the latest frame, if any.
    pub detection: Option<Detection>,
    /// One-line hint for the operator.
    pub hint: String,
    /// The lens result once solved; rolling shutter filled in when that step completes.
    pub result: Option<CameraCalibration>,
}

/// Handle to a running or finished calibration. Dropping it stops the worker.
#[derive(Component)]
pub struct CalibrationRun {
    progress: Arc<Mutex<CalibrationProgress>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl CalibrationRun {
    pub fn progress(&self) -> CalibrationProgress {
        self.progress.lock().unwrap().clone()
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self.progress.lock().unwrap().phase,
            CalibrationPhase::Done | CalibrationPhase::Failed(_)
        )
    }
}

impl Drop for CalibrationRun {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn start_runs(
    mut commands: Commands,
    new: Query<(Entity, &Calibrate, &WebcamFeed), Without<CalibrationRun>>,
) {
    for (entity, request, feed) in &new {
        let tap = feed.capture.tap();
        let progress = Arc::new(Mutex::new(CalibrationProgress {
            phase: CalibrationPhase::Lens {
                accepted: 0,
                needed: request.views_needed,
            },
            detection: None,
            hint: "show the whole board to the camera".into(),
            result: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (progress, stop, request) = (progress.clone(), stop.clone(), request.clone());
            let (width, height) = (feed.layout.width, feed.layout.height);
            std::thread::Builder::new()
                .name("calibration".into())
                .spawn(move || worker(tap, request, width, height, progress, stop))
                .expect("spawn calibration thread")
        };
        info!(
            "calibration started ({} views, {} sweep frames)",
            request.views_needed, request.sweep_frames_needed
        );
        commands.entity(entity).insert(CalibrationRun {
            progress,
            stop,
            thread: Some(thread),
        });
    }
}

/// When a run finishes, attach the result and drop the request so the run can be restarted by
/// inserting [`Calibrate`] again.
fn finish_runs(mut commands: Commands, runs: Query<(Entity, &CalibrationRun), With<Calibrate>>) {
    for (entity, run) in &runs {
        if run.is_finished() {
            let p = run.progress();
            let mut e = commands.entity(entity);
            e.remove::<Calibrate>();
            if let Some(result) = p.result {
                e.insert(result);
            }
        }
    }
}

fn worker(
    tap: crate::capture::FrameTap,
    request: Calibrate,
    width: u32,
    height: u32,
    progress: Arc<Mutex<CalibrationProgress>>,
    stop: Arc<AtomicBool>,
) {
    let set = |f: &dyn Fn(&mut CalibrationProgress)| {
        let mut p = progress.lock().unwrap();
        f(&mut p);
    };
    let min_corners = (request.board.inner_count() / 4).max(8);
    let mut lens = LensCalibrator::new(request.board);
    let mut last_accept = std::time::Instant::now();

    // Step 1: lens views.
    let calibration = loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame) = tap.recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        let detection = detect(&frame, &request.board, min_corners);
        let hint = match &detection {
            // Hold still for a moment between views so motion blur does not enter the solve.
            Some(d) if last_accept.elapsed() > Duration::from_millis(700) => match lens.consider(d)
            {
                ViewVerdict::Accepted => {
                    last_accept = std::time::Instant::now();
                    format!("view {} taken; move the board", lens.len())
                }
                ViewVerdict::NotNovel => "move the board to a new position or angle".to_string(),
                ViewVerdict::TooFewCorners => "show more of the board".to_string(),
            },
            Some(_) => "hold...".to_string(),
            None => "board not found".to_string(),
        };
        let accepted = lens.len();
        set(&|p| {
            p.phase = CalibrationPhase::Lens {
                accepted,
                needed: request.views_needed,
            };
            p.detection = detection.clone();
            p.hint = hint.clone();
        });
        if accepted >= request.views_needed {
            set(&|p| p.phase = CalibrationPhase::Solving);
            match lens.solve(width, height) {
                Ok(c) => {
                    info!(
                        "lens: fx {:.1} fy {:.1} cx {:.1} cy {:.1} k1 {:.4} k2 {:.4} p1 {:.5} p2 {:.5} k3 {:.4}, reproj {:.3} px over {} views",
                        c.fx,
                        c.fy,
                        c.cx,
                        c.cy,
                        c.k1,
                        c.k2,
                        c.p1,
                        c.p2,
                        c.k3,
                        c.reproj_error_px,
                        c.views
                    );
                    break c;
                }
                Err(e) => {
                    set(&|p| p.phase = CalibrationPhase::Failed(format!("lens solve failed: {e}")));
                    return;
                }
            }
        }
    };
    set(&|p| p.result = Some(calibration.clone()));
    if request.sweep_frames_needed == 0 {
        set(&|p| p.phase = CalibrationPhase::Done);
        return;
    }

    // Step 2: rolling shutter sweep.
    let mut rs = RollingShutterCalibrator::new(calibration.clone(), request.board);
    set(&|p| {
        p.phase = CalibrationPhase::RollingShutter {
            accepted: 0,
            needed: request.sweep_frames_needed,
        };
        p.hint = "sweep the board sideways, keeping it fully in view".into();
    });
    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let Some(frame) = tap.recv_timeout(Duration::from_millis(200)) else {
            continue;
        };
        let detection = detect(&frame, &request.board, min_corners);
        let hint = match detection.as_ref().map(|d| rs.feed(d)) {
            None => "board not found".to_string(),
            Some(SweepVerdict::Incomplete) => "keep the whole board in view".into(),
            Some(SweepVerdict::NoPair) => "keep sweeping".into(),
            Some(SweepVerdict::Accepted { velocity_px_s, .. }) => {
                format!(
                    "sweep back and forth: {velocity_px_s:.0} px/s, speed range {:.0} px/s",
                    rs.speed_span()
                )
            }
        };
        let accepted = rs.samples();
        set(&|p| {
            p.phase = CalibrationPhase::RollingShutter {
                accepted,
                needed: request.sweep_frames_needed,
            };
            p.detection = detection.clone();
            p.hint = hint.clone();
        });
        if accepted >= request.sweep_frames_needed && rs.speed_span() >= rs.min_speed_span_px_s {
            let estimate = rs.estimate();
            let mut result = calibration.clone();
            result.rolling_shutter = estimate;
            if let Some(r) = estimate {
                info!(
                    "rolling shutter: {:.2} us/row, readout {:.2} ms, spread {:.2} us over {} frames",
                    r.line_delay_us, r.readout_ms, r.spread_us, r.samples
                );
            }
            set(&|p| {
                p.result = Some(result.clone());
                p.phase = CalibrationPhase::Done;
                p.hint = "done".into();
            });
            return;
        }
    }
}
