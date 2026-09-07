//! Picking a capture mode.

use v4l::FourCC;

use crate::capture::{CaptureMode, FOURCC_MJPG, FOURCC_UYVY, FOURCC_YUYV};

/// Below this a feed reads as a slideshow.
pub const SMOOTH_FPS: f32 = 24.0;

/// What the caller wants. Any field may be left unspecified.
#[derive(Debug, Clone, Copy, Default)]
pub struct Want {
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    /// Restrict to one pixel format.
    pub fourcc: Option<FourCC>,
}

/// A chosen mode and the rate to request.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    pub fourcc: FourCC,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    /// Raw pixels the GPU can read without a decoder (the zero-copy path).
    pub raw: bool,
}

pub fn is_raw(fourcc: FourCC) -> bool {
    fourcc == FOURCC_YUYV || fourcc == FOURCC_UYVY
}

pub fn is_supported(fourcc: FourCC) -> bool {
    is_raw(fourcc) || fourcc == FOURCC_MJPG
}

/// Choose the mode that best matches `want`.
///
/// Rules, in order:
/// 1. Only formats this crate's consumers can display (raw 4:2:2 or MJPEG).
/// 2. At the requested size, prefer a raw mode that reaches the requested rate, then MJPEG that
///    reaches it, then whichever is fastest (raw wins ties). Raw is preferred because it needs
///    no decoder and can be read by the GPU in place.
/// 3. With no size given: the largest mode that reaches the requested rate (or [`SMOOTH_FPS`]),
///    raw preferred at equal size; failing that, the fastest mode at the largest size.
///
/// The returned rate is the highest offered rate not above the requested one, or the mode's
/// fastest when nothing is requested or the request is unreachable.
pub fn choose_mode(modes: &[CaptureMode], want: Want) -> Option<Choice> {
    let usable: Vec<&CaptureMode> = modes
        .iter()
        .filter(|m| {
            is_supported(m.fourcc) && want.fourcc.is_none_or(|f| f == m.fourcc) && !m.fps.is_empty()
        })
        .collect();
    let target = want.fps.unwrap_or(SMOOTH_FPS);
    let max_fps = |m: &CaptureMode| m.fps.iter().copied().fold(0.0_f32, f32::max);
    let area = |m: &CaptureMode| (m.width as u64) * (m.height as u64);
    // Sort key: reaching the target rate first. Among modes that reach it, raw beats compressed;
    // among modes that do not, the fastest wins and raw only breaks ties.
    let score = |m: &&CaptureMode| {
        let reaches = max_fps(m) >= target;
        let fps = (max_fps(m) * 1000.0) as u64;
        let raw = is_raw(m.fourcc) as u64;
        if reaches {
            (true, raw, fps)
        } else {
            (false, fps, raw)
        }
    };

    let size_matches: Vec<&CaptureMode> = usable
        .iter()
        .copied()
        .filter(|m| {
            want.width.is_none_or(|w| w == m.width) && want.height.is_none_or(|h| h == m.height)
        })
        .collect();

    let picked = if want.width.is_some() || want.height.is_some() {
        size_matches.iter().copied().max_by_key(score)
    } else {
        let smooth: Vec<&CaptureMode> = usable
            .iter()
            .copied()
            .filter(|m| max_fps(m) >= target)
            .collect();
        if smooth.is_empty() {
            usable
                .iter()
                .copied()
                .max_by_key(|m| ((max_fps(m) * 1000.0) as u64, area(m), is_raw(m.fourcc)))
        } else {
            smooth
                .iter()
                .copied()
                .max_by_key(|m| (area(m), is_raw(m.fourcc), (max_fps(m) * 1000.0) as u64))
        }
    }?;

    let fps = match want.fps {
        Some(f) => picked
            .fps
            .iter()
            .copied()
            .filter(|r| *r <= f + 0.01)
            .fold(f32::NAN, f32::max),
        None => f32::NAN,
    };
    let fps = if fps.is_nan() { max_fps(picked) } else { fps };
    Some(Choice {
        fourcc: picked.fourcc,
        width: picked.width,
        height: picked.height,
        fps,
        raw: is_raw(picked.fourcc),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(fourcc: FourCC, w: u32, h: u32, fps: &[f32]) -> CaptureMode {
        CaptureMode {
            fourcc,
            description: String::new(),
            width: w,
            height: h,
            fps: fps.to_vec(),
        }
    }

    fn brio_usb2() -> Vec<CaptureMode> {
        vec![
            mode(FOURCC_YUYV, 848, 480, &[30.0, 24.0, 15.0]),
            mode(FOURCC_YUYV, 1280, 720, &[10.0, 8.0, 5.0]),
            mode(FOURCC_YUYV, 1920, 1080, &[5.0]),
            mode(FOURCC_MJPG, 1280, 720, &[60.0, 30.0, 24.0, 15.0]),
            mode(FOURCC_MJPG, 1920, 1080, &[30.0, 24.0, 15.0]),
        ]
    }

    fn brio_usb3() -> Vec<CaptureMode> {
        vec![
            mode(FOURCC_YUYV, 1280, 720, &[30.0, 24.0, 15.0]),
            mode(FOURCC_YUYV, 1920, 1080, &[30.0, 24.0, 15.0]),
            mode(FOURCC_MJPG, 1280, 720, &[90.0, 60.0, 30.0, 24.0]),
            mode(FOURCC_MJPG, 1920, 1080, &[60.0, 30.0, 24.0]),
            mode(FOURCC_MJPG, 3840, 2160, &[30.0, 24.0]),
        ]
    }

    #[test]
    fn hd60_on_usb2_needs_mjpeg() {
        let c = choose_mode(
            &brio_usb2(),
            Want {
                width: Some(1280),
                height: Some(720),
                fps: Some(60.0),
                fourcc: None,
            },
        )
        .unwrap();
        assert_eq!((c.fourcc, c.fps, c.raw), (FOURCC_MJPG, 60.0, false));
    }

    #[test]
    fn raw_wins_when_it_reaches_the_rate() {
        let c = choose_mode(
            &brio_usb3(),
            Want {
                width: Some(1920),
                height: Some(1080),
                fps: Some(30.0),
                fourcc: None,
            },
        )
        .unwrap();
        assert_eq!((c.fourcc, c.fps, c.raw), (FOURCC_YUYV, 30.0, true));
    }

    #[test]
    fn rate_is_the_highest_offered_not_above_the_request() {
        let c = choose_mode(
            &brio_usb3(),
            Want {
                width: Some(1280),
                height: Some(720),
                fps: Some(75.0),
                fourcc: None,
            },
        )
        .unwrap();
        assert_eq!((c.fourcc, c.fps), (FOURCC_MJPG, 60.0));
    }

    #[test]
    fn unreachable_rate_takes_the_fastest_available() {
        let c = choose_mode(
            &brio_usb2(),
            Want {
                width: Some(1920),
                height: Some(1080),
                fps: Some(60.0),
                fourcc: None,
            },
        )
        .unwrap();
        assert_eq!((c.fourcc, c.fps), (FOURCC_MJPG, 30.0));
    }

    #[test]
    fn no_size_picks_the_largest_smooth_mode_preferring_raw() {
        let c = choose_mode(&brio_usb3(), Want::default()).unwrap();
        // 4K MJPEG reaches 24 fps and is largest.
        assert_eq!((c.width, c.fourcc), (3840, FOURCC_MJPG));
        let c = choose_mode(
            &brio_usb3(),
            Want {
                fps: Some(30.0),
                fourcc: Some(FOURCC_YUYV),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((c.width, c.fourcc, c.fps), (1920, FOURCC_YUYV, 30.0));
    }

    #[test]
    fn a_slow_big_mode_loses_to_a_smaller_smooth_one() {
        let modes = vec![
            mode(FOURCC_YUYV, 1920, 1080, &[5.0]),
            mode(FOURCC_YUYV, 1280, 720, &[10.0]),
            mode(FOURCC_YUYV, 800, 448, &[30.0]),
        ];
        let c = choose_mode(&modes, Want::default()).unwrap();
        assert_eq!((c.width, c.fps), (800, 30.0));
    }

    #[test]
    fn nothing_usable_chooses_nothing() {
        let modes = vec![mode(FourCC { repr: *b"H264" }, 1920, 1080, &[30.0])];
        assert!(choose_mode(&modes, Want::default()).is_none());
    }
}
