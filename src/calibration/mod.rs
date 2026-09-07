//! Lens and rolling-shutter calibration with a printed chessboard.
//!
//! Step 1, lens: hold the board at a dozen or so different positions and angles. Each accepted
//! view contributes its labelled inner corners to Zhang's method with Brown-Conrady distortion
//! (via the `calibration-rs` crates), giving `fx, fy, cx, cy, k1, k2, k3, p1, p2`.
//!
//! Step 2, rolling shutter: sweep the board sideways. A rolling shutter exposes each row a little
//! later than the previous one, so a moving board is sheared: the residual of the corners
//! against the best-fit homography grows linearly with the row. The slope divided by the board's
//! image-space velocity is the line delay. See [`RollingShutterCalibrator`].
//!
//! [`CalibrationPlugin`] runs both steps on a worker thread for a [`Webcam`](crate::Webcam) entity
//! that carries a [`Calibrate`] component.

mod lens;
mod plugin;
mod rolling;

pub use lens::*;
pub use plugin::*;
pub use rolling::*;

use crate::capture::GrayFrame;
use calib_targets::{
    chessboard::ChessboardParams,
    detect::{default_chess_config, detect_chessboard},
};
use serde::{Deserialize, Serialize};

/// The printed target: `squares_x` by `squares_y` squares, each `square_mm` wide. The detector
/// labels the *inner* corners, of which there are `(squares_x - 1) * (squares_y - 1)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ChessboardSpec {
    pub squares_x: u32,
    pub squares_y: u32,
    pub square_mm: f32,
}

impl Default for ChessboardSpec {
    /// 10 by 7 squares of 20 mm: 9 by 6 inner corners.
    fn default() -> Self {
        Self {
            squares_x: 10,
            squares_y: 7,
            square_mm: 20.0,
        }
    }
}

impl ChessboardSpec {
    pub fn inner_corners(&self) -> (u32, u32) {
        (
            self.squares_x.saturating_sub(1),
            self.squares_y.saturating_sub(1),
        )
    }

    pub fn inner_count(&self) -> usize {
        let (a, b) = self.inner_corners();
        (a * b) as usize
    }
}

/// One labelled inner corner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    /// Grid label, rebased so the smallest detected label is (0, 0). Consistent within a frame;
    /// the board's orientation is not recovered, which calibration does not need.
    pub grid: (i32, i32),
    /// Pixel position.
    pub px: [f32; 2],
}

/// Corners found in one frame.
#[derive(Debug, Clone)]
pub struct Detection {
    pub corners: Vec<Corner>,
    /// Median cell size in pixels.
    pub cell_px: f32,
    pub width: u32,
    pub height: u32,
    pub sequence: u32,
    pub timestamp: std::time::Duration,
}

impl Detection {
    pub fn grid_extent(&self) -> (i32, i32) {
        let (mut du, mut dv) = (0, 0);
        for c in &self.corners {
            du = du.max(c.grid.0);
            dv = dv.max(c.grid.1);
        }
        (du + 1, dv + 1)
    }

    /// Every inner corner of `spec` was found.
    pub fn is_complete(&self, spec: &ChessboardSpec) -> bool {
        self.corners.len() == spec.inner_count()
    }

    pub fn centroid(&self) -> [f32; 2] {
        let n = self.corners.len().max(1) as f32;
        let (sx, sy) = self
            .corners
            .iter()
            .fold((0.0, 0.0), |(x, y), c| (x + c.px[0], y + c.px[1]));
        [sx / n, sy / n]
    }
}

/// Find the chessboard's inner corners in a luma frame. About 3 ms at 720p, 7 ms at 1080p.
/// Returns `None` when fewer than `min_corners` corners are labelled or the labelled grid is
/// larger than the board can be (a spurious detection).
pub fn detect(frame: &GrayFrame, spec: &ChessboardSpec, min_corners: usize) -> Option<Detection> {
    let img = image::GrayImage::from_raw(frame.width, frame.height, frame.data.clone())?;
    let det =
        detect_chessboard(&img, &default_chess_config(), &ChessboardParams::default()).ok()?;
    let corners: Vec<Corner> = det
        .corners
        .iter()
        .map(|c| Corner {
            grid: (c.grid.u, c.grid.v),
            px: [c.position.x, c.position.y],
        })
        .collect();
    if corners.len() < min_corners {
        return None;
    }
    let d = Detection {
        corners,
        cell_px: det.cell_size.unwrap_or(0.0),
        width: frame.width,
        height: frame.height,
        sequence: frame.sequence,
        timestamp: frame.timestamp,
    };
    let (ex, ey) = d.grid_extent();
    let (ix, iy) = spec.inner_corners();
    let (big, small) = (ix.max(iy) as i32, ix.min(iy) as i32);
    if ex.max(ey) > big || ex.min(ey) > small {
        return None;
    }
    Some(d)
}
