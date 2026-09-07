use super::{ChessboardSpec, Detection};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CorrespondenceView, DistortionParams, IntrinsicsParams, PlanarDataset, Pt2, Pt3, View,
};
use vision_calibration_pipeline::{
    planar_intrinsics::{PlanarIntrinsicsProblem, run_calibration},
    session::CalibrationSession,
};

/// Rolling-shutter result, see [`super::RollingShutterCalibrator`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RollingShutter {
    /// Time between the exposure of two consecutive rows, microseconds.
    pub line_delay_us: f64,
    /// `line_delay * height`: time to read the whole frame, milliseconds.
    pub readout_ms: f64,
    /// Frames the estimate is based on.
    pub samples: usize,
    /// RMS of the per-frame line-delay estimates around the mean, microseconds. Small means the
    /// sweeps agreed with each other.
    pub spread_us: f64,
}

/// A pinhole camera with Brown-Conrady distortion, in pixels, for one stream size.
///
/// Also a Bevy component: [`super::CalibrationPlugin`] attaches it to the webcam entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, bevy::prelude::Component)]
pub struct CameraCalibration {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub p1: f64,
    pub p2: f64,
    /// Mean reprojection error of the solve, pixels.
    pub reproj_error_px: f64,
    /// Views used.
    pub views: usize,
    pub board: ChessboardSpec,
    pub rolling_shutter: Option<RollingShutter>,
}

impl CameraCalibration {
    /// Apply the distortion model to normalised coordinates.
    pub fn distort_normalized(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
        let xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (xd, yd)
    }

    /// Normalised coordinates of a distorted pixel (iterative inverse of the model).
    pub fn undistort_normalized(&self, xd: f64, yd: f64) -> (f64, f64) {
        let (mut x, mut y) = (xd, yd);
        for _ in 0..10 {
            let r2 = x * x + y * y;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2;
            let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            x = (xd - dx) / radial;
            y = (yd - dy) / radial;
        }
        (x, y)
    }

    /// Where a distorted pixel would be in an ideal pinhole image with the same `K`.
    pub fn undistort_pixel(&self, px: f64, py: f64) -> (f64, f64) {
        let (x, y) = self.undistort_normalized((px - self.cx) / self.fx, (py - self.cy) / self.fy);
        (self.fx * x + self.cx, self.fy * y + self.cy)
    }

    /// Project a camera-space point (x right, y down, z forward) to a distorted pixel.
    pub fn project(&self, x: f64, y: f64, z: f64) -> Option<(f64, f64)> {
        if z <= 0.0 {
            return None;
        }
        let (xd, yd) = self.distort_normalized(x / z, y / z);
        Some((self.fx * xd + self.cx, self.fy * yd + self.cy))
    }

    /// Unit ray in camera space (x right, y down, z forward) through a distorted pixel.
    pub fn ray(&self, px: f64, py: f64) -> [f64; 3] {
        let (x, y) = self.undistort_normalized((px - self.cx) / self.fx, (py - self.cy) / self.fy);
        let n = (x * x + y * y + 1.0).sqrt();
        [x / n, y / n, 1.0 / n]
    }

    /// Horizontal and vertical field of view, degrees.
    pub fn fov_deg(&self) -> (f64, f64) {
        let h = 2.0 * ((self.width as f64 / 2.0) / self.fx).atan().to_degrees();
        let v = 2.0 * ((self.height as f64 / 2.0) / self.fy).atan().to_degrees();
        (h, v)
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(
            path,
            serde_json::to_string_pretty(self).map_err(std::io::Error::other)?,
        )
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        serde_json::from_str(&std::fs::read_to_string(path)?).map_err(std::io::Error::other)
    }
}

/// Collects diverse board views and solves for intrinsics.
pub struct LensCalibrator {
    pub spec: ChessboardSpec,
    /// Accept a view only if it has at least this fraction of the inner corners.
    pub min_corner_fraction: f32,
    /// Accept a view only if it differs this much (in cell widths) from every accepted view:
    /// centroid moved, or size/orientation changed by the equivalent amount.
    pub min_novelty_cells: f32,
    views: Vec<Detection>,
    signatures: Vec<[f32; 4]>,
}

/// Why a frame was or was not taken as a calibration view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewVerdict {
    Accepted,
    TooFewCorners,
    /// Too similar to a view already taken; move the board.
    NotNovel,
}

impl LensCalibrator {
    pub fn new(spec: ChessboardSpec) -> Self {
        Self {
            spec,
            min_corner_fraction: 0.75,
            min_novelty_cells: 1.5,
            views: Vec::new(),
            signatures: Vec::new(),
        }
    }

    pub fn views(&self) -> &[Detection] {
        &self.views
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// A pose signature in cell units: centroid x, centroid y, scale, orientation.
    fn signature(d: &Detection) -> [f32; 4] {
        let cell = d.cell_px.max(1.0);
        let c = d.centroid();
        // Orientation from the vector between the two corners farthest apart along the u axis.
        let (mut a, mut b) = (None, None);
        for k in &d.corners {
            if a.is_none_or(|p: &Corner| {
                k.grid.0 < p.grid.0 || (k.grid.0 == p.grid.0 && k.grid.1 < p.grid.1)
            }) {
                a = Some(k);
            }
            if b.is_none_or(|p: &Corner| {
                k.grid.0 > p.grid.0 || (k.grid.0 == p.grid.0 && k.grid.1 > p.grid.1)
            }) {
                b = Some(k);
            }
        }
        let angle = match (a, b) {
            (Some(a), Some(b)) => (b.px[1] - a.px[1]).atan2(b.px[0] - a.px[0]),
            _ => 0.0,
        };
        // Angle scaled so that 90° ≈ 3 cells of novelty; scale as log2 so halving == 2 cells.
        [c[0] / cell, c[1] / cell, 2.0 * cell.log2(), angle * 2.0]
    }

    pub fn consider(&mut self, d: &Detection) -> ViewVerdict {
        if (d.corners.len() as f32) < self.min_corner_fraction * self.spec.inner_count() as f32 {
            return ViewVerdict::TooFewCorners;
        }
        let sig = Self::signature(d);
        let novel = self.signatures.iter().all(|s| {
            let dist = (0..4).map(|i| (s[i] - sig[i]).powi(2)).sum::<f32>().sqrt();
            dist >= self.min_novelty_cells
        });
        if !novel {
            return ViewVerdict::NotNovel;
        }
        self.signatures.push(sig);
        self.views.push(d.clone());
        ViewVerdict::Accepted
    }

    /// Solve with all accepted views. Needs at least 3, better 10 or more.
    pub fn solve(&self, width: u32, height: u32) -> Result<CameraCalibration, String> {
        if self.views.len() < 3 {
            return Err(format!("need at least 3 views, have {}", self.views.len()));
        }
        let square = self.spec.square_mm as f64;
        let views: Vec<View<_>> = self
            .views
            .iter()
            .map(|d| {
                View::without_meta(CorrespondenceView {
                    points_3d: d
                        .corners
                        .iter()
                        .map(|c| Pt3::new(c.grid.0 as f64 * square, c.grid.1 as f64 * square, 0.0))
                        .collect(),
                    points_2d: d
                        .corners
                        .iter()
                        .map(|c| Pt2::new(c.px[0] as f64, c.px[1] as f64))
                        .collect(),
                    weights: vec![],
                })
            })
            .collect();
        let dataset = PlanarDataset::new(views).map_err(|e| e.to_string())?;
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::with_input(dataset)
            .map_err(|e| e.to_string())?;
        run_calibration(&mut session).map_err(|e| e.to_string())?;
        let export = session.export().map_err(|e| e.to_string())?;
        let cam = &export.params.camera;
        let IntrinsicsParams::FxFyCxCySkew { params: k } = &cam.intrinsics;
        let (k1, k2, k3, p1, p2) = match &cam.distortion {
            DistortionParams::BrownConrady5 { params: d } => (d.k1, d.k2, d.k3, d.p1, d.p2),
            DistortionParams::None => (0.0, 0.0, 0.0, 0.0, 0.0),
            other => {
                return Err(format!(
                    "solver returned an unexpected distortion model: {other:?}"
                ));
            }
        };
        Ok(CameraCalibration {
            width,
            height,
            fx: k.fx,
            fy: k.fy,
            cx: k.cx,
            cy: k.cy,
            k1,
            k2,
            k3,
            p1,
            p2,
            reproj_error_px: export.mean_reproj_error,
            views: self.views.len(),
            board: self.spec,
            rolling_shutter: None,
        })
    }
}

use super::Corner;
