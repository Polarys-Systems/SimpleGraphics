//! CPU preparation for `texture_atlas.comp`.

use crate::simple_ttf::{Glyph, PathCommand, Vec2};
use std::io::{self, Error, ErrorKind};

/// Exact std430 array stride: 32 bytes. Y increases downward.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct GpuQuadratic {
    pub p0_p1: [f32; 4],
    pub p2_winding: [f32; 4],
}

/// The raster fields in the compute shader's push-constant block.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct RasterJob {
    pub rect: [u32; 4],
    pub segment_range: [u32; 4],
    pub mapping: [f32; 4],
}

const _: () = {
    assert!(std::mem::size_of::<GpuQuadratic>() == 32);
    assert!(std::mem::offset_of!(GpuQuadratic, p2_winding) == 16);
    assert!(std::mem::size_of::<RasterJob>() == 48);
    assert!(std::mem::offset_of!(RasterJob, segment_range) == 16);
    assert!(std::mem::offset_of!(RasterJob, mapping) == 32);
};

impl RasterJob {
    pub fn workgroups(&self) -> [u32; 3] {
        [self.rect[2].div_ceil(8), self.rect[3].div_ceil(8), 1]
    }

    pub fn to_le_bytes(self) -> [u8; 48] {
        let mut bytes = [0; 48];
        for (index, value) in self
            .rect
            .iter()
            .chain(self.segment_range.iter())
            .enumerate()
        {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (index, value) in self.mapping.iter().enumerate() {
            bytes[32 + index * 4..36 + index * 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug)]
pub struct PreparedGlyph {
    pub segments: Vec<GpuQuadratic>,
    /// Conservative bounds of the uploaded controls, in Y-down font units.
    pub bounds: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct RasterPlacement {
    /// Padded bitmap offset from an integer-positioned pen, in pixels.
    pub origin: [i32; 2],
    pub size: [u32; 2],
    pub scale: f32,
    pub phase: [f32; 2],
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidInput, message)
}

type Point = [f64; 2];

fn font_point(point: Vec2) -> io::Result<Point> {
    if !point.x.is_finite() || !point.y.is_finite() {
        return Err(invalid("non-finite glyph coordinate"));
    }
    Ok([f64::from(point.x), -f64::from(point.y)])
}

fn lerp(a: Point, b: Point, t: f64) -> Point {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn emit_monotonic(out: &mut Vec<GpuQuadratic>, a: Point, b: Point, c: Point) {
    let (a, c, direction) = if a[1] <= c[1] {
        (a, c, 1.0)
    } else {
        (c, a, -1.0)
    };
    let y0 = a[1] as f32;
    let y2 = c[1] as f32;
    if y0 == y2 {
        return;
    }
    let y1 = (b[1] as f32).clamp(y0, y2);
    out.push(GpuQuadratic {
        p0_p1: [a[0] as f32, y0, b[0] as f32, y1],
        p2_winding: [c[0] as f32, y2, direction, 0.0],
    });
}

fn append_quadratic(out: &mut Vec<GpuQuadratic>, a: Point, b: Point, c: Point) {
    let denominator = a[1] - 2.0 * b[1] + c[1];
    if denominator != 0.0 {
        let t = (a[1] - b[1]) / denominator;
        if t > 0.0 && t < 1.0 {
            let ab = lerp(a, b, t);
            let bc = lerp(b, c, t);
            let middle = lerp(ab, bc, t);
            emit_monotonic(out, a, ab, middle);
            emit_monotonic(out, middle, bc, c);
            return;
        }
    }
    emit_monotonic(out, a, b, c);
}

/// Converts already-closed TrueType contours to monotonic quadratic segments.
pub fn prepare_glyph(glyph: &Glyph) -> io::Result<PreparedGlyph> {
    let commands = glyph.path_commands();
    let mut segments = Vec::with_capacity(commands.len());
    let mut current: Option<Point> = None;
    for command in commands {
        match command {
            PathCommand::MoveTo(point) => current = Some(font_point(point)?),
            PathCommand::LineTo(point) => {
                let a = current.ok_or_else(|| invalid("line without a starting point"))?;
                let c = font_point(point)?;
                append_quadratic(&mut segments, a, lerp(a, c, 0.5), c);
                current = Some(c);
            }
            PathCommand::QuadTo { control, end } => {
                let a = current.ok_or_else(|| invalid("quadratic without a starting point"))?;
                let b = font_point(control)?;
                let c = font_point(end)?;
                append_quadratic(&mut segments, a, b, c);
                current = Some(c);
            }
        }
    }

    let mut bounds = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for quadratic in &segments {
        for point in [
            [quadratic.p0_p1[0], quadratic.p0_p1[1]],
            [quadratic.p0_p1[2], quadratic.p0_p1[3]],
            [quadratic.p2_winding[0], quadratic.p2_winding[1]],
        ] {
            bounds[0] = bounds[0].min(point[0]);
            bounds[1] = bounds[1].min(point[1]);
            bounds[2] = bounds[2].max(point[0]);
            bounds[3] = bounds[3].max(point[1]);
        }
    }
    if segments.is_empty() {
        bounds = [0.0; 4];
    }
    Ok(PreparedGlyph { segments, bounds })
}

impl PreparedGlyph {
    pub fn placement(
        &self,
        ppem: f32,
        units_per_em: u16,
        phase: [f32; 2],
    ) -> io::Result<Option<RasterPlacement>> {
        if !ppem.is_finite()
            || ppem <= 0.0
            || units_per_em == 0
            || phase
                .iter()
                .any(|phase| !phase.is_finite() || *phase < 0.0 || *phase >= 1.0)
        {
            return Err(invalid("invalid ppem, units_per_em, or fractional phase"));
        }
        if self.segments.is_empty() {
            return Ok(None);
        }
        let scale = ppem / f32::from(units_per_em);
        if scale <= 0.0 || !scale.is_finite() || !scale.recip().is_finite() {
            return Err(invalid("invalid raster scale"));
        }
        let mut origin = [0; 2];
        let mut size = [0; 2];
        for axis in 0..2 {
            let low = (f64::from(self.bounds[axis]) * f64::from(scale) + f64::from(phase[axis]))
                .floor()
                - 1.0;
            let high = (f64::from(self.bounds[axis + 2]) * f64::from(scale)
                + f64::from(phase[axis]))
            .ceil()
                + 1.0;
            if !low.is_finite()
                || !high.is_finite()
                || low < -1_048_576.0
                || high > 1_048_576.0
                || high <= low
            {
                return Err(invalid("glyph raster bounds exceed supported range"));
            }
            origin[axis] = low as i32;
            size[axis] = (high - low) as u32;
        }
        Ok(Some(RasterPlacement {
            origin,
            size,
            scale,
            phase,
        }))
    }
}

impl RasterPlacement {
    pub fn job(
        &self,
        atlas_xy: [u32; 2],
        atlas_size: [u32; 2],
        first_segment: u32,
        segment_count: u32,
        total_uploaded_segments: u32,
    ) -> io::Result<RasterJob> {
        for axis in 0..2 {
            if self.size[axis] == 0
                || atlas_size[axis] > i32::MAX as u32
                || atlas_xy[axis] > atlas_size[axis]
                || self.size[axis] > atlas_size[axis] - atlas_xy[axis]
            {
                return Err(invalid("atlas rectangle is outside the image"));
            }
        }
        if first_segment > total_uploaded_segments
            || segment_count > total_uploaded_segments - first_segment
        {
            return Err(invalid("curve range is outside the uploaded buffer"));
        }
        Ok(RasterJob {
            rect: [atlas_xy[0], atlas_xy[1], self.size[0], self.size[1]],
            segment_range: [first_segment, segment_count, total_uploaded_segments, 0],
            mapping: [
                self.scale.recip(),
                self.origin[0] as f32 - self.phase[0],
                self.origin[1] as f32 - self.phase[1],
                0.0,
            ],
        })
    }
}

/// Serialize once when adding geometry to the GPU outline cache.
pub fn encode_segments(segments: &[GpuQuadratic]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(segments));
    for segment in segments {
        for value in segment.p0_p1.iter().chain(segment.p2_winding.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simple_ttf::{Contour, GlyphBounds, GlyphPoint};

    #[test]
    fn preparation_splits_curves_and_matches_gpu_layout() {
        let glyph = Glyph {
            bounds: GlyphBounds {
                x_min: 0,
                y_min: 0,
                x_max: 100,
                y_max: 100,
            },
            contours: vec![Contour {
                points: vec![
                    GlyphPoint {
                        x: 0.0,
                        y: 0.0,
                        on_curve: true,
                    },
                    GlyphPoint {
                        x: 50.0,
                        y: 100.0,
                        on_curve: false,
                    },
                    GlyphPoint {
                        x: 100.0,
                        y: 0.0,
                        on_curve: true,
                    },
                ],
            }],
        };

        let prepared = prepare_glyph(&glyph).unwrap();
        assert!(prepared.segments.len() >= 2);
        assert_eq!(
            encode_segments(&prepared.segments).len(),
            prepared.segments.len() * 32
        );
        assert!(prepared.segments.iter().all(|segment| {
            segment.p0_p1[1] <= segment.p0_p1[3] && segment.p0_p1[3] <= segment.p2_winding[1]
        }));
    }
}
