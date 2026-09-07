use super::{CameraCalibration, ChessboardSpec, Detection, RollingShutter};
use nalgebra::{DMatrix, Matrix3};

/// Estimates the rolling-shutter line delay from a board swept back and forth across the view.
///
/// A rolling shutter exposes each row later than the one above, so a board moving at image
/// velocity `v` is sheared by `v · τ` pixels per row. For a planar target that shear is an affine
/// map and is indistinguishable from the board's pose in any single frame. What *is* observable
/// is how the shear changes with velocity: over a sweep with varying speed (a back-and-forth pass
/// is ideal) the per-frame horizontal shear `g_f` and horizontal velocity `v_f` satisfy
/// `g_f = g_0 + v_f · τ`, where `g_0` is the board's static orientation. The slope of that
/// regression is the line delay `τ`.
///
/// Per frame, `g_f` comes from an affine least-squares fit of undistorted pixels against board
/// millimetres (x-change per row of the board's column direction), and `v_f` from the median
/// displacement of the same corners since the previous frame.
pub struct RollingShutterCalibrator {
    pub calibration: CameraCalibration,
    pub spec: ChessboardSpec,
    /// Velocity range (fastest minus slowest horizontal velocity, px/s) the samples must span
    /// before an estimate is reported.
    pub min_speed_span_px_s: f64,
    prev: Option<Detection>,
    /// (v_x, shear) per accepted frame.
    samples: Vec<(f64, f64)>,
}

/// Feedback for one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SweepVerdict {
    /// Board must be fully visible in this and the previous frame.
    Incomplete,
    /// Previous frame missing or not consecutive; this one becomes the reference.
    NoPair,
    /// Accepted with this horizontal velocity (px/s) and shear (px per row).
    Accepted { velocity_px_s: f64, shear: f64 },
}

impl RollingShutterCalibrator {
    pub fn new(calibration: CameraCalibration, spec: ChessboardSpec) -> Self {
        Self {
            calibration,
            spec,
            min_speed_span_px_s: 600.0,
            prev: None,
            samples: Vec::new(),
        }
    }

    pub fn samples(&self) -> usize {
        self.samples.len()
    }

    /// Fastest minus slowest horizontal velocity seen so far, px/s.
    pub fn speed_span(&self) -> f64 {
        let (lo, hi) = self
            .samples
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s.0), hi.max(s.0))
            });
        if self.samples.is_empty() {
            0.0
        } else {
            hi - lo
        }
    }

    pub fn feed(&mut self, d: &Detection) -> SweepVerdict {
        if !d.is_complete(&self.spec) {
            self.prev = None;
            return SweepVerdict::Incomplete;
        }
        let Some(prev) = self.prev.replace(d.clone()) else {
            return SweepVerdict::NoPair;
        };
        if d.sequence.wrapping_sub(prev.sequence) != 1 || d.timestamp <= prev.timestamp {
            return SweepVerdict::NoPair;
        }
        let dt = (d.timestamp - prev.timestamp).as_secs_f64();

        let und = |c: &super::Corner| {
            self.calibration
                .undistort_pixel(c.px[0] as f64, c.px[1] as f64)
        };
        let mut dx = Vec::new();
        for c in &d.corners {
            if let Some(p) = prev.corners.iter().find(|p| p.grid == c.grid) {
                dx.push(und(c).0 - und(p).0);
            }
        }
        if dx.len() < 8 {
            return SweepVerdict::NoPair;
        }
        let vx = median(&mut dx) / dt;

        let square = self.spec.square_mm as f64;
        let board: Vec<[f64; 2]> = d
            .corners
            .iter()
            .map(|c| [c.grid.0 as f64 * square, c.grid.1 as f64 * square])
            .collect();
        let img: Vec<[f64; 2]> = d
            .corners
            .iter()
            .map(|c| {
                let p = und(c);
                [p.0, p.1]
            })
            .collect();
        let Some(shear) = horizontal_shear(&board, &img) else {
            return SweepVerdict::NoPair;
        };
        self.samples.push((vx, shear));
        SweepVerdict::Accepted {
            velocity_px_s: vx,
            shear,
        }
    }

    /// Regression of shear against velocity. `None` until the sweep covered enough speed range.
    pub fn estimate(&self) -> Option<RollingShutter> {
        if self.samples.len() < 6 || self.speed_span() < self.min_speed_span_px_s {
            return None;
        }
        let n = self.samples.len() as f64;
        let mv = self.samples.iter().map(|s| s.0).sum::<f64>() / n;
        let mg = self.samples.iter().map(|s| s.1).sum::<f64>() / n;
        let sxx = self.samples.iter().map(|s| (s.0 - mv).powi(2)).sum::<f64>();
        let sxy = self
            .samples
            .iter()
            .map(|s| (s.0 - mv) * (s.1 - mg))
            .sum::<f64>();
        if sxx <= 0.0 {
            return None;
        }
        let tau = sxy / sxx;
        let g0 = mg - tau * mv;
        // Residual shear converted to a per-sample τ error, weighted by velocity.
        let resid = self
            .samples
            .iter()
            .map(|s| (s.1 - g0 - tau * s.0).powi(2))
            .sum::<f64>()
            / n;
        let spread = (resid / (sxx / n)).sqrt();
        Some(RollingShutter {
            line_delay_us: tau * 1e6,
            readout_ms: tau * self.calibration.height as f64 * 1e3,
            samples: self.samples.len(),
            spread_us: spread * 1e6,
        })
    }
}

/// Affine least-squares fit `x = a·X + b·Y + c`, `y = d·X + e·Y + f` of image against board, then
/// the horizontal shear per image row: how much `x` shifts per pixel of `y` along the board's
/// column direction, `b / e`. Under a rolling shutter this gains `v_x · τ`.
fn horizontal_shear(board: &[[f64; 2]], img: &[[f64; 2]]) -> Option<f64> {
    let n = board.len();
    if n < 4 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(n, 3);
    let mut bx = nalgebra::DVector::<f64>::zeros(n);
    let mut by = nalgebra::DVector::<f64>::zeros(n);
    for (i, (b, p)) in board.iter().zip(img).enumerate() {
        a[(i, 0)] = b[0];
        a[(i, 1)] = b[1];
        a[(i, 2)] = 1.0;
        bx[i] = p[0];
        by[i] = p[1];
    }
    let svd = a.clone().svd(true, true);
    let cx = svd.solve(&bx, 1e-12).ok()?;
    let cy = svd.solve(&by, 1e-12).ok()?;
    let (b_, e_) = (cx[1], cy[1]);
    if e_.abs() < 1e-9 {
        return None;
    }
    Some(b_ / e_)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Homography `dst ≈ H · src` by the normalised DLT.
pub fn homography(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Matrix3<f64>> {
    if src.len() < 4 || src.len() != dst.len() {
        return None;
    }
    let (ts, s_n) = normalize(src);
    let (td, d_n) = normalize(dst);
    let mut a = DMatrix::<f64>::zeros(2 * src.len(), 9);
    for (i, (s, d)) in s_n.iter().zip(&d_n).enumerate() {
        let (x, y, u, v) = (s[0], s[1], d[0], d[1]);
        a.set_row(
            2 * i,
            &nalgebra::RowDVector::from_row_slice(&[-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]),
        );
        a.set_row(
            2 * i + 1,
            &nalgebra::RowDVector::from_row_slice(&[0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]),
        );
    }
    let svd = a.svd(false, true);
    let vt = svd.v_t?;
    let hrow: Vec<f64> = vt.row(vt.nrows() - 1).iter().copied().collect();
    let hn = Matrix3::from_row_slice(&hrow);
    let h = td.try_inverse()? * hn * ts;
    Some(h / h[(2, 2)])
}

/// Hartley normalisation: translate to the centroid and scale to mean distance √2.
fn normalize(pts: &[[f64; 2]]) -> (Matrix3<f64>, Vec<[f64; 2]>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let mean_d = pts
        .iter()
        .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
        .sum::<f64>()
        / n;
    let s = if mean_d > 0.0 {
        2f64.sqrt() / mean_d
    } else {
        1.0
    };
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    (
        t,
        pts.iter()
            .map(|p| [s * (p[0] - cx), s * (p[1] - cy)])
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn homography_recovers_a_known_map() {
        let h_true = Matrix3::new(1.2, 0.1, 30.0, -0.05, 0.9, 12.0, 1e-4, -2e-4, 1.0);
        let src: Vec<[f64; 2]> = (0..6)
            .flat_map(|j| (0..9).map(move |i| [i as f64 * 20.0, j as f64 * 20.0]))
            .collect();
        let dst: Vec<[f64; 2]> = src
            .iter()
            .map(|p| {
                let q = h_true * Vector3::new(p[0], p[1], 1.0);
                [q.x / q.z, q.y / q.z]
            })
            .collect();
        let h = homography(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let q = h * Vector3::new(s[0], s[1], 1.0);
            assert!((q.x / q.z - d[0]).abs() < 1e-6 && (q.y / q.z - d[1]).abs() < 1e-6);
        }
    }

    fn calib() -> CameraCalibration {
        CameraCalibration {
            width: 1280,
            height: 720,
            fx: 1000.0,
            fy: 1000.0,
            cx: 640.0,
            cy: 360.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            reproj_error_px: 0.0,
            views: 0,
            board: ChessboardSpec::default(),
            rolling_shutter: None,
        }
    }

    /// A fronto-parallel board swept back and forth under a rolling shutter with a known line
    /// delay; velocity varies sinusoidally so the shear-vs-velocity regression is observable.
    #[test]
    fn line_delay_is_recovered_from_a_sweep() {
        let spec = ChessboardSpec::default();
        let tau = 30e-6; // 30 µs per row
        let fps = 30.0;
        let mut rs = RollingShutterCalibrator::new(calib(), spec);
        let mut accepted = 0;
        for f in 0..40u32 {
            let t = f as f64 / fps;
            let x0 = 500.0 + 250.0 * (2.0 * std::f64::consts::PI * 0.5 * t).sin();
            let vx = 250.0
                * 2.0
                * std::f64::consts::PI
                * 0.5
                * (2.0 * std::f64::consts::PI * 0.5 * t).cos();
            let y_mean = 360.0;
            // A small static in-plane rotation, which the regression intercept must absorb.
            let rot = 0.05f64;
            let corners = (0..6)
                .flat_map(|j| (0..9).map(move |i| (i, j)))
                .map(|(i, j)| {
                    let bx = i as f64 * 40.0 - 160.0;
                    let by = j as f64 * 40.0 - 100.0;
                    let x_true = x0 + bx * rot.cos() - by * rot.sin();
                    let y = y_mean + bx * rot.sin() + by * rot.cos();
                    let x = x_true + vx * tau * (y - y_mean);
                    super::super::Corner {
                        grid: (i, j),
                        px: [x as f32, y as f32],
                    }
                })
                .collect();
            let d = Detection {
                corners,
                cell_px: 40.0,
                width: 1280,
                height: 720,
                sequence: f,
                timestamp: std::time::Duration::from_secs_f64(t),
            };
            if matches!(rs.feed(&d), SweepVerdict::Accepted { .. }) {
                accepted += 1;
            }
        }
        assert!(accepted > 30, "accepted {accepted}");
        let est = rs.estimate().expect("estimate");
        assert!((est.line_delay_us - 30.0).abs() < 1.5, "got {est:?}");
    }
}
