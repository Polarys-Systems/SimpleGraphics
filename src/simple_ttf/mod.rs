//! Minimal, bounds-checked TrueType outline parser.
//!
//! The parser implements the tables needed by a basic text renderer:
//!
//! ```text
//! Unicode code point -> cmap -> glyph ID -> loca -> glyf outline
//!                                      \-> hmtx horizontal metrics
//! ```
//!
//! It supports Unicode `cmap` formats 4 and 12, both `loca` encodings, simple
//! and composite `glyf` outlines, and horizontal metrics from `hhea`/`hmtx`.
//! Coordinates and metrics are returned in font units. Multiply them by
//! `pixel_height / units_per_em` before rendering at a desired pixel height.
//!
//! Composite components are flattened in font space, including nested transforms
//! and attachment by explicit contour point. Phantom-point attachment is rejected
//! as unsupported. Pixel-grid rounding and hint instructions are not executed.
//! `horizontal_metrics()` returns raw `hmtx` values, not `USE_MY_METRICS` resolution.
//!
//! CFF outlines, font variations, hint execution, kerning, shaping, and
//! rasterization are intentionally outside this module's current scope.
//!
//! # Basic use
//!
//! ```ignore
//! let font = TTFParser::read_from("font.ttf")?;
//! let glyph_id = font.glyph_index('A' as u32);
//! let metrics = font.horizontal_metrics(glyph_id).unwrap();
//! if let Some(glyph) = font.load_glyph(glyph_id)? {
//!     for command in glyph.path_commands() {
//!         // Feed commands to a quadratic-path rasterizer.
//!     }
//! }
//! # Ok::<(), std::io::Error>(())
//! ```

use std::collections::HashMap;
use std::io::{self, Error, ErrorKind};
use std::ops::Range;

// Simple-glyph point flags from the OpenType glyf table specification.
const ON_CURVE_POINT: u8 = 0x01;
const X_SHORT_VECTOR: u8 = 0x02;
const Y_SHORT_VECTOR: u8 = 0x04;
const REPEAT_FLAG: u8 = 0x08;
const X_IS_SAME_OR_POSITIVE_X_SHORT_VECTOR: u8 = 0x10;
const Y_IS_SAME_OR_POSITIVE_Y_SHORT_VECTOR: u8 = 0x20;

// Composite-glyph flags from the OpenType glyf specification.
const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
const ARGS_ARE_XY_VALUES: u16 = 0x0002;
const WE_HAVE_A_SCALE: u16 = 0x0008;
const MORE_COMPONENTS: u16 = 0x0020;
const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;
const WE_HAVE_INSTRUCTIONS: u16 = 0x0100;
const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
const UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;

// Implementation limits, not limits imposed by the font format. Bound both
// recursion and total expansion: a shallow component graph can still expand
// exponentially when it references the same child more than once.
const MAX_GLYPH_LOAD_DEPTH: usize = 32;
const MAX_GLYPH_LOAD_VISITS: usize = 4096;
const MAX_GLYPH_LOAD_POINTS: usize = 1 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Values from `head` needed to scale outlines and decode `loca`.
pub struct HeadData {
    /// Number of font units in one em, normally used as the outline scale denominator.
    pub units_per_em: u16,
    /// `0` for short `loca` offsets and `1` for long offsets.
    pub index_to_loc_format: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Basic metadata from the `maxp` table.
pub struct MaxpTable {
    /// Raw 16.16 fixed-point `maxp` version.
    pub version: u32,
    /// Number of glyphs in the font, including glyph zero (`.notdef`).
    pub num_glyphs: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Vertical line metrics and `hmtx` layout information from `hhea`.
pub struct HheaTable {
    /// Recommended distance from the baseline to the highest line extent, in font units.
    pub ascender: i16,
    /// Recommended distance from the baseline to the lowest line extent, in font units.
    pub descender: i16,
    /// Recommended additional spacing between lines, in font units.
    pub line_gap: i16,
    /// Number of full `(advance_width, left_side_bearing)` records in `hmtx`.
    pub number_of_h_metrics: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Horizontal layout metrics for one glyph, expressed in font units.
pub struct HorizontalMetrics {
    /// Distance by which a horizontal text pen advances after this glyph.
    pub advance_width: u16,
    /// Horizontal distance from the pen position to the glyph's left bound.
    pub left_side_bearing: i16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Two-dimensional point in font space.
pub struct Vec2 {
    /// Horizontal coordinate, increasing to the right.
    pub x: f32,
    /// Vertical coordinate, increasing upwards in TrueType font space.
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A decoded point from a simple TrueType contour.
pub struct GlyphPoint {
    /// Horizontal coordinate in font units.
    pub x: f32,
    /// Vertical coordinate in font units.
    pub y: f32,
    /// Whether this is a point on the curve rather than a quadratic control point.
    pub on_curve: bool,
}

impl GlyphPoint {
    fn midpoint(self, other: Self) -> Self {
        Self {
            x: (self.x + other.x) * 0.5,
            y: (self.y + other.y) * 0.5,
            on_curve: true,
        }
    }

    fn position(self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// One closed outline contour in the point order stored by TrueType.
pub struct Contour {
    /// Explicit points from `glyf`; implied on-curve points are not stored here.
    pub points: Vec<GlyphPoint>,
}

impl Contour {
    /// Converts this closed contour into renderer-independent path commands.
    ///
    /// TrueType permits two consecutive off-curve points. In that case this
    /// method inserts their midpoint as an implied on-curve point and emits two
    /// quadratic segments. The returned command sequence includes the final
    /// segment back to its initial [`PathCommand::MoveTo`] position.
    pub fn path_commands(&self) -> Vec<PathCommand> {
        if self.points.is_empty() {
            return Vec::new();
        }

        // Materialize implied points so the command pass never sees two adjacent controls.
        let mut expanded = Vec::with_capacity(self.points.len() * 2);
        for index in 0..self.points.len() {
            let point = self.points[index];
            let next = self.points[(index + 1) % self.points.len()];
            expanded.push(point);
            if !point.on_curve && !next.on_curve {
                expanded.push(point.midpoint(next));
            }
        }

        let Some(start_index) = expanded.iter().position(|point| point.on_curve) else {
            return Vec::new();
        };
        let start = expanded[start_index];
        let mut commands = vec![PathCommand::MoveTo(start.position())];
        let mut control: Option<GlyphPoint> = None;

        // Include the start point once more to emit the segment that closes the contour.
        for step in 1..=expanded.len() {
            let point = expanded[(start_index + step) % expanded.len()];
            if point.on_curve {
                if let Some(control) = control.take() {
                    commands.push(PathCommand::QuadTo {
                        control: control.position(),
                        end: point.position(),
                    });
                } else {
                    commands.push(PathCommand::LineTo(point.position()));
                }
            } else {
                control = Some(point);
            }
        }

        commands
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Drawing operation produced from a TrueType outline.
///
/// Each contour starts with [`PathCommand::MoveTo`] and is closed by its final
/// line or quadratic command. A separate close-path operation is therefore not
/// required.
pub enum PathCommand {
    /// Starts a new contour at the supplied point.
    MoveTo(Vec2),
    /// Adds a straight segment from the current point.
    LineTo(Vec2),
    /// Adds a quadratic Bezier segment from the current point.
    QuadTo {
        /// Quadratic control point.
        control: Vec2,
        /// On-curve endpoint of the segment.
        end: Vec2,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Bounding box declared by a glyph, in font units.
pub struct GlyphBounds {
    /// Minimum horizontal coordinate.
    pub x_min: i16,
    /// Minimum vertical coordinate.
    pub y_min: i16,
    /// Maximum horizontal coordinate.
    pub x_max: i16,
    /// Maximum vertical coordinate.
    pub y_max: i16,
}

#[derive(Debug, Clone, PartialEq)]
/// Decoded geometry of a simple or flattened composite TrueType glyph.
pub struct Glyph {
    /// Bounding box from the glyph header.
    pub bounds: GlyphBounds,
    /// Closed contours that make up the glyph outline.
    pub contours: Vec<Contour>,
}

impl Glyph {
    /// Converts every contour to path commands, preserving contour order.
    ///
    /// Every contour contributes its own [`PathCommand::MoveTo`], so consumers
    /// can distinguish independent outer shapes and holes in the flat result.
    pub fn path_commands(&self) -> Vec<PathCommand> {
        self.contours
            .iter()
            .flat_map(Contour::path_commands)
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct TableRecord {
    offset: usize,
    length: usize,
}

#[derive(Debug)]
enum CmapSubtable {
    Format4(CmapFormat4),
    Format12(Vec<CmapGroup>),
}

#[derive(Debug)]
struct CmapFormat4 {
    data: Vec<u8>,
    seg_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct CmapGroup {
    start_char_code: u32,
    end_char_code: u32,
    start_glyph_id: u32,
}

/// Parsed TrueType font data required for basic outline rendering and layout.
///
/// Construction eagerly validates table bounds and decodes character maps,
/// glyph locations, and horizontal metrics. Glyph outlines themselves are
/// decoded lazily by [`TTFParser::load_glyph`].
pub struct TTFParser {
    font_data: Vec<u8>,
    font_tables: HashMap<[u8; 4], TableRecord>,
    head_data: HeadData,
    maxp_table: MaxpTable,
    hhea_table: HheaTable,
    cmaps: Vec<CmapSubtable>,
    glyph_offsets: Vec<u32>,
    horizontal_metrics: Vec<HorizontalMetrics>,
}

impl TTFParser {
    /// Reads and parses a TrueType font from a filesystem path.
    ///
    /// # Errors
    ///
    /// Returns the underlying I/O error when the file cannot be read. Malformed
    /// fonts, unsupported outline formats, and missing required tables produce
    /// [`io::ErrorKind::InvalidData`].
    pub fn read_from(path: &str) -> io::Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Parses an owned TrueType font buffer.
    ///
    /// Keeping ownership of the buffer allows [`TTFParser::load_glyph`] to
    /// decode outlines on demand without lifetime parameters or another copy.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidData`] for invalid table relationships
    /// or unsupported TrueType data, and [`io::ErrorKind::UnexpectedEof`] when
    /// a declared table or structure is truncated.
    pub fn from_bytes(font_data: Vec<u8>) -> io::Result<Self> {
        let font_tables = parse_table_directory(&font_data)?;
        let head_data = parse_head(table_data(&font_data, &font_tables, *b"head")?)?;
        let maxp_table = parse_maxp(table_data(&font_data, &font_tables, *b"maxp")?)?;
        let hhea_table = parse_hhea(table_data(&font_data, &font_tables, *b"hhea")?)?;
        let cmaps = parse_cmap(table_data(&font_data, &font_tables, *b"cmap")?)?;
        let glyph_offsets = parse_loca(
            table_data(&font_data, &font_tables, *b"loca")?,
            head_data.index_to_loc_format,
            maxp_table.num_glyphs,
        )?;
        let glyf_length = table_record(&font_tables, *b"glyf")?.length;
        validate_loca(&glyph_offsets, glyf_length)?;
        let horizontal_metrics = parse_hmtx(
            table_data(&font_data, &font_tables, *b"hmtx")?,
            maxp_table.num_glyphs,
            hhea_table.number_of_h_metrics,
        )?;

        Ok(Self {
            font_data,
            font_tables,
            head_data,
            maxp_table,
            hhea_table,
            cmaps,
            glyph_offsets,
            horizontal_metrics,
        })
    }

    /// Compatibility no-op for the original staged parser API.
    ///
    /// [`TTFParser::read_from`] and [`TTFParser::from_bytes`] now complete all
    /// parsing before returning, so callers do not need to invoke this method.
    pub fn parse_information(&mut self) -> &mut Self {
        self
    }

    /// Returns the parsed values required from the `head` table.
    pub fn head(&self) -> HeadData {
        self.head_data
    }

    /// Returns the basic `maxp` table metadata.
    pub fn maxp(&self) -> MaxpTable {
        self.maxp_table
    }

    /// Returns horizontal-header metrics from `hhea`.
    pub fn hhea(&self) -> HheaTable {
        self.hhea_table
    }

    /// Returns the number of font units per em from `head`.
    ///
    /// A scale suitable for rendering at `pixel_height` is
    /// `pixel_height / units_per_em() as f32`.
    pub fn units_per_em(&self) -> u16 {
        self.head_data.units_per_em
    }

    /// Returns the total number of glyphs declared by `maxp`.
    pub fn num_glyphs(&self) -> u16 {
        self.maxp_table.num_glyphs
    }

    /// Maps a Unicode code point to a glyph ID using `cmap`.
    ///
    /// Format 12 maps are preferred over format 4 maps when both are present.
    /// Returns glyph zero (`.notdef`) when the code point is invalid, unmapped,
    /// or maps outside the font's declared glyph count.
    pub fn glyph_index(&self, codepoint: u32) -> u16 {
        if codepoint > 0x10ffff {
            return 0;
        }

        self.cmaps
            .iter()
            .find_map(|cmap| cmap.glyph_index(codepoint))
            .filter(|glyph_id| *glyph_id < self.maxp_table.num_glyphs)
            .unwrap_or(0)
    }

    /// Returns the glyph's absolute byte range within the original font data.
    ///
    /// Adjacent equal `loca` offsets produce an empty range, which represents a
    /// glyph without outline data, commonly a space. Returns `None` only when
    /// `glyph_id` is outside `0..num_glyphs()`.
    pub fn glyph_range(&self, glyph_id: u16) -> Option<Range<usize>> {
        let index = usize::from(glyph_id);
        if index >= usize::from(self.maxp_table.num_glyphs) {
            return None;
        }

        let glyf = self.font_tables.get(b"glyf")?;
        let start = glyf
            .offset
            .checked_add(self.glyph_offsets[index] as usize)?;
        let end = glyf
            .offset
            .checked_add(self.glyph_offsets[index + 1] as usize)?;
        Some(start..end)
    }

    /// Decodes a simple or composite TrueType glyph from `glyf`.
    ///
    /// Composite children are recursively decoded, transformed, and appended in
    /// font order. Explicit point indices are resolved before path conversion
    /// inserts implied points. Results remain unhinted, in font units.
    ///
    /// `ROUND_XY_TO_GRID` is intentionally ignored: pixel-grid rounding requires
    /// a device scale and belongs to a future hinted loader. Instruction bytes
    /// are bounds-checked but not executed. `USE_MY_METRICS` does not change this
    /// geometry-only result or the raw values returned by `horizontal_metrics`.
    ///
    /// Returns `Ok(None)` when `loca` declares an empty range. A non-empty
    /// zero-contour glyph is represented by `Some(Glyph)` with no contours.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidData`] for an out-of-range glyph ID,
    /// malformed components, cycles, or an exceeded expansion limit. Returns
    /// [`io::ErrorKind::Unsupported`] for attachment using phantom points, and
    /// [`io::ErrorKind::UnexpectedEof`] for truncated data.
    pub fn load_glyph(&self, glyph_id: u16) -> io::Result<Option<Glyph>> {
        let mut active = Vec::with_capacity(MAX_GLYPH_LOAD_DEPTH);
        let mut budget = GlyphLoadBudget {
            visits: MAX_GLYPH_LOAD_VISITS,
            points: MAX_GLYPH_LOAD_POINTS,
        };
        self.load_glyph_inner(glyph_id, &mut active, &mut budget)
    }

    fn load_glyph_inner(
        &self,
        glyph_id: u16,
        active: &mut Vec<u16>,
        budget: &mut GlyphLoadBudget,
    ) -> io::Result<Option<Glyph>> {
        if active.len() >= MAX_GLYPH_LOAD_DEPTH {
            return Err(invalid_data("composite glyph nesting exceeds loader limit"));
        }
        if active.contains(&glyph_id) {
            return Err(invalid_data("cyclic composite glyph reference"));
        }
        budget.visits = budget.visits.checked_sub(1)
            .ok_or_else(|| invalid_data("glyph expansion exceeds visit limit"))?;
        let range = self
            .glyph_range(glyph_id)
            .ok_or_else(|| invalid_data("glyph ID is outside maxp.numGlyphs"))?;
        if range.is_empty() {
            return Ok(None);
        }
        let data = &self.font_data[range];
        require(data, 0, 10, "glyf header")?;
        if read_i16(data, 0)? >= 0 {
            let glyph = parse_simple_glyph(data)?;
            if let Some(glyph) = &glyph {
                let count: usize = glyph.contours.iter().map(|c| c.points.len()).sum();
                budget.points = budget.points.checked_sub(count)
                    .ok_or_else(|| invalid_data("glyph expansion exceeds point limit"))?;
            }
            return Ok(glyph);
        }

        active.push(glyph_id);
        let result = self.load_composite_glyph(data, active, budget);
        active.pop();
        result.map(Some)
    }

    fn load_composite_glyph(
        &self,
        data: &[u8],
        active: &mut Vec<u16>,
        budget: &mut GlyphLoadBudget,
    ) -> io::Result<Glyph> {
        let mut glyph = Glyph {
            bounds: GlyphBounds {
                x_min: read_i16(data, 2)?,
                y_min: read_i16(data, 4)?,
                x_max: read_i16(data, 6)?,
                y_max: read_i16(data, 8)?,
            },
            contours: Vec::new(),
        };
        let mut reader = GlyphReader { data, cursor: 10 };
        let mut has_instructions = false;
        let mut first_component = true;
        loop {
            let component = GlyphComponent::read(&mut reader)?;
            has_instructions |= component.flags & WE_HAVE_INSTRUCTIONS != 0;
            if first_component && matches!(component.placement, ComponentPlacement::Points(..)) {
                return Err(invalid_data("first composite component must use XY offsets"));
            }
            first_component = false;
            let child = self.load_glyph_inner(component.glyph_id, active, budget)?;
            let mut contours = child.map(|g| g.contours).unwrap_or_default();

            let offset = match component.placement {
                ComponentPlacement::Offset(offset) => {
                    let offset_flags = component.flags
                        & (SCALED_COMPONENT_OFFSET | UNSCALED_COMPONENT_OFFSET);
                    // Neither flag, or both flags: use the recommended default,
                    // an unscaled offset. Ignore these flags for point attachment.
                    if offset_flags == SCALED_COMPONENT_OFFSET {
                        component.transform.apply(offset)
                    } else {
                        offset
                    }
                }
                ComponentPlacement::Points(parent_index, child_index) => {
                    let parent = explicit_point(&glyph.contours, parent_index)?;
                    let child = component.transform.apply(explicit_point(&contours, child_index)?);
                    Vec2 { x: parent.x - child.x, y: parent.y - child.y }
                }
            };
            for contour in &mut contours {
                for point in &mut contour.points {
                    let transformed = component.transform.apply(point.position());
                    point.x = transformed.x + offset.x;
                    point.y = transformed.y + offset.y;
                }
            }
            // Move contour allocations; do not clone all of the child's points.
            glyph.contours.append(&mut contours);
            if component.flags & MORE_COMPONENTS == 0 {
                break;
            }
        }
        // The flag may occur on any component, not just the final one.
        if has_instructions {
            let length = usize::from(reader.u16()?);
            require(data, reader.cursor, length, "composite glyph instructions")?;
        }
        Ok(glyph)
    }

    /// Returns horizontal metrics for a glyph ID.
    ///
    /// The `hmtx` rule that trailing glyphs reuse the last full advance width
    /// is already applied. Returns `None` for an out-of-range glyph ID.
    /// These are raw table values; composite `USE_MY_METRICS` and hinting are
    /// not resolved by this method.
    pub fn horizontal_metrics(&self, glyph_id: u16) -> Option<HorizontalMetrics> {
        self.horizontal_metrics.get(usize::from(glyph_id)).copied()
    }
}

impl CmapSubtable {
    fn glyph_index(&self, codepoint: u32) -> Option<u16> {
        match self {
            Self::Format4(cmap) => cmap.glyph_index(codepoint),
            Self::Format12(groups) => {
                // Format 12 groups are sorted, non-overlapping code point ranges.
                let group_index = groups
                    .binary_search_by(|group| {
                        if codepoint < group.start_char_code {
                            std::cmp::Ordering::Greater
                        } else if codepoint > group.end_char_code {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
                    .ok()?;
                let group = groups[group_index];
                let glyph_id = group
                    .start_glyph_id
                    .checked_add(codepoint - group.start_char_code)?;
                u16::try_from(glyph_id)
                    .ok()
                    .filter(|glyph_id| *glyph_id != 0)
            }
        }
    }
}

impl CmapFormat4 {
    fn glyph_index(&self, codepoint: u32) -> Option<u16> {
        let codepoint = u16::try_from(codepoint).ok()?;
        let end_codes_offset = 14;
        let start_codes_offset = end_codes_offset + self.seg_count * 2 + 2;
        let deltas_offset = start_codes_offset + self.seg_count * 2;
        let range_offsets_offset = deltas_offset + self.seg_count * 2;

        let mut matching_segment = None;
        for segment in 0..self.seg_count {
            if read_u16(&self.data, end_codes_offset + segment * 2).ok()? >= codepoint {
                matching_segment = Some(segment);
                break;
            }
        }
        let segment = matching_segment?;
        let start_code = read_u16(&self.data, start_codes_offset + segment * 2).ok()?;
        if codepoint < start_code {
            return None;
        }

        let delta = read_i16(&self.data, deltas_offset + segment * 2).ok()?;
        let range_offset_position = range_offsets_offset + segment * 2;
        let range_offset = read_u16(&self.data, range_offset_position).ok()?;
        let glyph_id = if range_offset == 0 {
            codepoint.wrapping_add_signed(delta)
        } else {
            // idRangeOffset is relative to the location of its own array entry,
            // not to the beginning of the subtable.
            let glyph_position = range_offset_position
                .checked_add(usize::from(range_offset))?
                .checked_add(usize::from(codepoint - start_code) * 2)?;
            let glyph_id = read_u16(&self.data, glyph_position).ok()?;
            if glyph_id == 0 {
                return None;
            }
            glyph_id.wrapping_add_signed(delta)
        };

        (glyph_id != 0).then_some(glyph_id)
    }
}

fn parse_table_directory(data: &[u8]) -> io::Result<HashMap<[u8; 4], TableRecord>> {
    require(data, 0, 12, "offset table")?;
    let sfnt_version = read_u32(data, 0)?;
    if sfnt_version != 0x0001_0000 && sfnt_version != u32::from_be_bytes(*b"true") {
        return Err(invalid_data("font does not contain TrueType outlines"));
    }

    let num_tables = usize::from(read_u16(data, 4)?);
    require(data, 12, num_tables.saturating_mul(16), "table directory")?;
    let mut tables = HashMap::with_capacity(num_tables);
    for index in 0..num_tables {
        let offset = 12 + index * 16;
        let tag = [
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ];
        let table_offset = read_u32(data, offset + 8)? as usize;
        let length = read_u32(data, offset + 12)? as usize;
        require(data, table_offset, length, "font table")?;
        if tables
            .insert(
                tag,
                TableRecord {
                    offset: table_offset,
                    length,
                },
            )
            .is_some()
        {
            return Err(invalid_data("duplicate table tag"));
        }
    }
    Ok(tables)
}

fn table_record(tables: &HashMap<[u8; 4], TableRecord>, tag: [u8; 4]) -> io::Result<TableRecord> {
    tables.get(&tag).copied().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidData,
            format!(
                "required table {:?} is missing",
                String::from_utf8_lossy(&tag)
            ),
        )
    })
}

fn table_data<'a>(
    data: &'a [u8],
    tables: &HashMap<[u8; 4], TableRecord>,
    tag: [u8; 4],
) -> io::Result<&'a [u8]> {
    let record = table_record(tables, tag)?;
    Ok(&data[record.offset..record.offset + record.length])
}

fn parse_head(data: &[u8]) -> io::Result<HeadData> {
    require(data, 0, 54, "head table")?;
    if read_u32(data, 12)? != 0x5f0f_3cf5 {
        return Err(invalid_data("invalid head magic number"));
    }
    let units_per_em = read_u16(data, 18)?;
    if !(16..=16384).contains(&units_per_em) {
        return Err(invalid_data("head.unitsPerEm is outside the valid range"));
    }
    let index_to_loc_format = read_i16(data, 50)?;
    if !matches!(index_to_loc_format, 0 | 1) {
        return Err(invalid_data("unsupported head.indexToLocFormat"));
    }
    Ok(HeadData {
        units_per_em,
        index_to_loc_format,
    })
}

fn parse_maxp(data: &[u8]) -> io::Result<MaxpTable> {
    require(data, 0, 6, "maxp table")?;
    let table = MaxpTable {
        version: read_u32(data, 0)?,
        num_glyphs: read_u16(data, 4)?,
    };
    if table.num_glyphs == 0 {
        return Err(invalid_data("maxp.numGlyphs is zero"));
    }
    Ok(table)
}

fn parse_hhea(data: &[u8]) -> io::Result<HheaTable> {
    require(data, 0, 36, "hhea table")?;
    let table = HheaTable {
        ascender: read_i16(data, 4)?,
        descender: read_i16(data, 6)?,
        line_gap: read_i16(data, 8)?,
        number_of_h_metrics: read_u16(data, 34)?,
    };
    if table.number_of_h_metrics == 0 {
        return Err(invalid_data("hhea.numberOfHMetrics is zero"));
    }
    Ok(table)
}

fn parse_cmap(data: &[u8]) -> io::Result<Vec<CmapSubtable>> {
    require(data, 0, 4, "cmap table")?;
    if read_u16(data, 0)? != 0 {
        return Err(invalid_data("unsupported cmap version"));
    }
    let num_tables = usize::from(read_u16(data, 2)?);
    require(
        data,
        4,
        num_tables.saturating_mul(8),
        "cmap encoding records",
    )?;

    let mut records = Vec::new();
    for index in 0..num_tables {
        let record_offset = 4 + index * 8;
        let platform_id = read_u16(data, record_offset)?;
        let encoding_id = read_u16(data, record_offset + 2)?;
        let subtable_offset = read_u32(data, record_offset + 4)? as usize;
        require(data, subtable_offset, 2, "cmap subtable")?;
        let format = read_u16(data, subtable_offset)?;
        let priority = cmap_priority(platform_id, encoding_id, format);
        if priority > 0 {
            records.push((priority, subtable_offset, format));
        }
    }
    // Prefer format 12 because it covers supplementary Unicode planes. Keep
    // format 4 as a fallback for fonts whose subtables differ.
    records.sort_by_key(|record| std::cmp::Reverse(record.0));
    records.dedup_by_key(|record| record.1);

    let mut cmaps = Vec::new();
    for (_, offset, format) in records {
        let cmap = match format {
            4 => CmapSubtable::Format4(parse_cmap_format4(&data[offset..])?),
            12 => CmapSubtable::Format12(parse_cmap_format12(&data[offset..])?),
            _ => continue,
        };
        cmaps.push(cmap);
    }
    if cmaps.is_empty() {
        return Err(invalid_data("no supported Unicode cmap subtable"));
    }
    Ok(cmaps)
}

fn cmap_priority(platform_id: u16, encoding_id: u16, format: u16) -> u8 {
    let unicode_encoding = platform_id == 0 || (platform_id == 3 && matches!(encoding_id, 1 | 10));
    if !unicode_encoding {
        return 0;
    }
    match format {
        12 => 2,
        4 => 1,
        _ => 0,
    }
}

fn parse_cmap_format4(data: &[u8]) -> io::Result<CmapFormat4> {
    require(data, 0, 16, "cmap format 4 header")?;
    let length = usize::from(read_u16(data, 2)?);
    require(data, 0, length, "cmap format 4")?;
    let seg_count_x2 = read_u16(data, 6)?;
    if seg_count_x2 == 0 || seg_count_x2 % 2 != 0 {
        return Err(invalid_data("invalid cmap format 4 segment count"));
    }
    let seg_count = usize::from(seg_count_x2 / 2);
    let arrays_length = seg_count
        .checked_mul(8)
        .and_then(|length| length.checked_add(16))
        .ok_or_else(|| invalid_data("cmap format 4 length overflow"))?;
    if arrays_length > length {
        return Err(invalid_data("truncated cmap format 4 arrays"));
    }

    let end_codes_offset = 14;
    let start_codes_offset = end_codes_offset + seg_count * 2 + 2;
    let mut previous_end = None;
    for segment in 0..seg_count {
        let end = read_u16(data, end_codes_offset + segment * 2)?;
        let start = read_u16(data, start_codes_offset + segment * 2)?;
        if start > end || previous_end.is_some_and(|previous| end <= previous) {
            return Err(invalid_data("invalid cmap format 4 segments"));
        }
        previous_end = Some(end);
    }

    Ok(CmapFormat4 {
        data: data[..length].to_vec(),
        seg_count,
    })
}

fn parse_cmap_format12(data: &[u8]) -> io::Result<Vec<CmapGroup>> {
    require(data, 0, 16, "cmap format 12 header")?;
    let length = read_u32(data, 4)? as usize;
    require(data, 0, length, "cmap format 12")?;
    let num_groups = read_u32(data, 12)? as usize;
    let required = num_groups
        .checked_mul(12)
        .and_then(|length| length.checked_add(16))
        .ok_or_else(|| invalid_data("cmap format 12 length overflow"))?;
    if required > length {
        return Err(invalid_data("truncated cmap format 12 groups"));
    }

    let mut groups = Vec::with_capacity(num_groups);
    for index in 0..num_groups {
        let offset = 16 + index * 12;
        let group = CmapGroup {
            start_char_code: read_u32(data, offset)?,
            end_char_code: read_u32(data, offset + 4)?,
            start_glyph_id: read_u32(data, offset + 8)?,
        };
        if group.start_char_code > group.end_char_code
            || group.end_char_code > 0x10ffff
            || groups
                .last()
                .is_some_and(|previous: &CmapGroup| previous.end_char_code >= group.start_char_code)
        {
            return Err(invalid_data("invalid cmap format 12 groups"));
        }
        groups.push(group);
    }
    Ok(groups)
}

fn parse_loca(data: &[u8], format: i16, num_glyphs: u16) -> io::Result<Vec<u32>> {
    // The extra entry supplies the end offset for the final glyph.
    let count = usize::from(num_glyphs) + 1;
    let entry_size = if format == 0 { 2 } else { 4 };
    require(data, 0, count.saturating_mul(entry_size), "loca table")?;
    let mut offsets = Vec::with_capacity(count);
    for index in 0..count {
        let offset = if format == 0 {
            // Short loca entries store half-offsets to fit an aligned byte offset in u16.
            u32::from(read_u16(data, index * 2)?) * 2
        } else {
            read_u32(data, index * 4)?
        };
        offsets.push(offset);
    }
    Ok(offsets)
}

fn validate_loca(offsets: &[u32], glyf_length: usize) -> io::Result<()> {
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(invalid_data("loca offsets are not ordered"));
    }
    if offsets.last().copied().unwrap_or(0) as usize > glyf_length {
        return Err(invalid_data("loca offset is outside the glyf table"));
    }
    Ok(())
}

fn parse_hmtx(
    data: &[u8],
    num_glyphs: u16,
    number_of_h_metrics: u16,
) -> io::Result<Vec<HorizontalMetrics>> {
    if number_of_h_metrics > num_glyphs {
        return Err(invalid_data("hhea.numberOfHMetrics exceeds maxp.numGlyphs"));
    }
    let long_count = usize::from(number_of_h_metrics);
    let glyph_count = usize::from(num_glyphs);
    let required = long_count
        .checked_mul(4)
        .and_then(|length| length.checked_add((glyph_count - long_count) * 2))
        .ok_or_else(|| invalid_data("hmtx length overflow"))?;
    require(data, 0, required, "hmtx table")?;

    let mut metrics = Vec::with_capacity(glyph_count);
    for index in 0..long_count {
        metrics.push(HorizontalMetrics {
            advance_width: read_u16(data, index * 4)?,
            left_side_bearing: read_i16(data, index * 4 + 2)?,
        });
    }
    let last_advance_width = metrics
        .last()
        .ok_or_else(|| invalid_data("hmtx has no long horizontal metrics"))?
        .advance_width;
    // hmtx omits advance widths after numberOfHMetrics; all remaining glyphs
    // inherit the final full record's width and store only a side bearing.
    for index in long_count..glyph_count {
        metrics.push(HorizontalMetrics {
            advance_width: last_advance_width,
            left_side_bearing: read_i16(data, long_count * 4 + (index - long_count) * 2)?,
        });
    }
    Ok(metrics)
}

struct GlyphLoadBudget {
    visits: usize,
    points: usize,
}

/// Matrix coefficients in the order stored by glyf: xx, yx, xy, yy.
#[derive(Clone, Copy)]
struct ComponentTransform {
    xx: f32,
    yx: f32,
    xy: f32,
    yy: f32,
}

impl ComponentTransform {
    const IDENTITY: Self = Self { xx: 1.0, yx: 0.0, xy: 0.0, yy: 1.0 };

    fn apply(self, point: Vec2) -> Vec2 {
        Vec2 {
            x: self.xx * point.x + self.xy * point.y,
            y: self.yx * point.x + self.yy * point.y,
        }
    }
}

#[derive(Clone, Copy)]
enum ComponentPlacement {
    Offset(Vec2),
    Points(u16, u16),
}

struct GlyphComponent {
    flags: u16,
    glyph_id: u16,
    placement: ComponentPlacement,
    transform: ComponentTransform,
}

impl GlyphComponent {
    fn read(reader: &mut GlyphReader<'_>) -> io::Result<Self> {
        let flags = reader.u16()?;
        let glyph_id = reader.u16()?;
        let words = flags & ARG_1_AND_2_ARE_WORDS != 0;
        let placement = if flags & ARGS_ARE_XY_VALUES != 0 {
            // XY arguments are signed; point indices below are unsigned.
            let (x, y) = if words {
                (f32::from(reader.i16()?), f32::from(reader.i16()?))
            } else {
                (f32::from(reader.u8()? as i8), f32::from(reader.u8()? as i8))
            };
            ComponentPlacement::Offset(Vec2 { x, y })
        } else {
            let (parent, child) = if words {
                (reader.u16()?, reader.u16()?)
            } else {
                (u16::from(reader.u8()?), u16::from(reader.u8()?))
            };
            ComponentPlacement::Points(parent, child)
        };

        let transform_flags = flags
            & (WE_HAVE_A_SCALE | WE_HAVE_AN_X_AND_Y_SCALE | WE_HAVE_A_TWO_BY_TWO);
        if transform_flags.count_ones() > 1 {
            return Err(invalid_data("conflicting composite transform flags"));
        }
        let transform = match transform_flags {
            WE_HAVE_A_SCALE => {
                let scale = reader.f2dot14()?;
                ComponentTransform { xx: scale, yy: scale, ..ComponentTransform::IDENTITY }
            }
            WE_HAVE_AN_X_AND_Y_SCALE => ComponentTransform {
                xx: reader.f2dot14()?,
                yy: reader.f2dot14()?,
                ..ComponentTransform::IDENTITY
            },
            WE_HAVE_A_TWO_BY_TWO => ComponentTransform {
                xx: reader.f2dot14()?,
                yx: reader.f2dot14()?,
                xy: reader.f2dot14()?,
                yy: reader.f2dot14()?,
            },
            _ => ComponentTransform::IDENTITY,
        };
        Ok(Self { flags, glyph_id, placement, transform })
    }
}

struct GlyphReader<'a> {
    data: &'a [u8],
    cursor: usize,
}

impl GlyphReader<'_> {
    fn u8(&mut self) -> io::Result<u8> {
        let value = read_u8(self.data, self.cursor)?;
        self.cursor += 1;
        Ok(value)
    }

    fn u16(&mut self) -> io::Result<u16> {
        let value = read_u16(self.data, self.cursor)?;
        self.cursor += 2;
        Ok(value)
    }

    fn i16(&mut self) -> io::Result<i16> {
        Ok(self.u16()? as i16)
    }

    fn f2dot14(&mut self) -> io::Result<f32> {
        Ok(f32::from(self.i16()?) / 16384.0)
    }
}

fn explicit_point(contours: &[Contour], index: u16) -> io::Result<Vec2> {
    let mut remaining = usize::from(index);
    for contour in contours {
        if let Some(point) = contour.points.get(remaining) {
            return Ok(point.position());
        }
        remaining -= contour.points.len();
    }
    // TrueType has four phantom points beyond the explicit contour points.
    // Supporting them requires a metric-aware loader (including vertical metrics).
    if remaining < 4 {
        Err(Error::new(ErrorKind::Unsupported,
            "composite attachment to phantom points is not supported"))
    } else {
        Err(invalid_data("composite attachment point is outside the glyph"))
    }
}

fn parse_simple_glyph(data: &[u8]) -> io::Result<Option<Glyph>> {
    require(data, 0, 10, "glyf header")?;
    let number_of_contours = read_i16(data, 0)?;
    if number_of_contours < 0 {
        return Err(invalid_data("composite passed to simple glyph decoder"));
    }
    let bounds = GlyphBounds {
        x_min: read_i16(data, 2)?,
        y_min: read_i16(data, 4)?,
        x_max: read_i16(data, 6)?,
        y_max: read_i16(data, 8)?,
    };
    if number_of_contours == 0 {
        return Ok(Some(Glyph {
            bounds,
            contours: Vec::new(),
        }));
    }

    let contour_count = number_of_contours as usize;
    require(
        data,
        10,
        contour_count.saturating_mul(2),
        "glyf contour endpoints",
    )?;
    let mut end_points = Vec::with_capacity(contour_count);
    for index in 0..contour_count {
        let end_point = read_u16(data, 10 + index * 2)?;
        if end_points
            .last()
            .is_some_and(|previous| *previous >= end_point)
        {
            return Err(invalid_data("glyf contour endpoints are not ordered"));
        }
        end_points.push(end_point);
    }
    let point_count = usize::from(end_points[contour_count - 1]) + 1;

    let instruction_length_offset = 10 + contour_count * 2;
    let instruction_length = usize::from(read_u16(data, instruction_length_offset)?);
    let mut cursor = instruction_length_offset + 2;
    require(data, cursor, instruction_length, "glyf instructions")?;
    cursor += instruction_length;

    // Flags are run-length encoded independently of the coordinate streams.
    let mut flags = Vec::with_capacity(point_count);
    while flags.len() < point_count {
        let flag = read_u8(data, cursor)?;
        cursor += 1;
        flags.push(flag);
        if flag & REPEAT_FLAG != 0 {
            let repeat_count = usize::from(read_u8(data, cursor)?);
            cursor += 1;
            if flags.len() + repeat_count > point_count {
                return Err(invalid_data("glyf flag repeat exceeds point count"));
            }
            flags.extend(std::iter::repeat_n(flag, repeat_count));
        }
    }

    // Coordinates are deltas. TrueType stores every x delta first, followed by
    // every y delta, rather than interleaving x/y values per point.
    let mut x_coordinates = Vec::with_capacity(point_count);
    let mut x = 0_i32;
    for flag in &flags {
        let delta = read_coordinate_delta(
            data,
            &mut cursor,
            *flag,
            X_SHORT_VECTOR,
            X_IS_SAME_OR_POSITIVE_X_SHORT_VECTOR,
        )?;
        x = x
            .checked_add(delta)
            .ok_or_else(|| invalid_data("glyf x coordinate overflow"))?;
        x_coordinates.push(x);
    }

    let mut points = Vec::with_capacity(point_count);
    let mut y = 0_i32;
    for (index, flag) in flags.iter().enumerate() {
        let delta = read_coordinate_delta(
            data,
            &mut cursor,
            *flag,
            Y_SHORT_VECTOR,
            Y_IS_SAME_OR_POSITIVE_Y_SHORT_VECTOR,
        )?;
        y = y
            .checked_add(delta)
            .ok_or_else(|| invalid_data("glyf y coordinate overflow"))?;
        points.push(GlyphPoint {
            x: x_coordinates[index] as f32,
            y: y as f32,
            on_curve: flag & ON_CURVE_POINT != 0,
        });
    }

    let mut contours = Vec::with_capacity(contour_count);
    let mut start = 0;
    for end in end_points {
        let end = usize::from(end) + 1;
        contours.push(Contour {
            points: points[start..end].to_vec(),
        });
        start = end;
    }
    Ok(Some(Glyph { bounds, contours }))
}

fn read_coordinate_delta(
    data: &[u8],
    cursor: &mut usize,
    flag: u8,
    short_mask: u8,
    same_or_positive_mask: u8,
) -> io::Result<i32> {
    if flag & short_mask != 0 {
        let magnitude = i32::from(read_u8(data, *cursor)?);
        *cursor += 1;
        Ok(if flag & same_or_positive_mask != 0 {
            magnitude
        } else {
            -magnitude
        })
    } else if flag & same_or_positive_mask != 0 {
        Ok(0)
    } else {
        let delta = i32::from(read_i16(data, *cursor)?);
        *cursor += 2;
        Ok(delta)
    }
}

fn require(data: &[u8], offset: usize, length: usize, context: &'static str) -> io::Result<()> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| invalid_data("font data range overflow"))?;
    if end > data.len() {
        return Err(Error::new(
            ErrorKind::UnexpectedEof,
            format!("truncated {context}"),
        ));
    }
    Ok(())
}

fn read_u8(data: &[u8], offset: usize) -> io::Result<u8> {
    require(data, offset, 1, "u8")?;
    Ok(data[offset])
}

fn read_u16(data: &[u8], offset: usize) -> io::Result<u16> {
    require(data, offset, 2, "u16")?;
    Ok(u16::from_be_bytes([data[offset], data[offset + 1]]))
}

fn read_i16(data: &[u8], offset: usize) -> io::Result<i16> {
    require(data, offset, 2, "i16")?;
    Ok(i16::from_be_bytes([data[offset], data[offset + 1]]))
}

fn read_u32(data: &[u8], offset: usize) -> io::Result<u32> {
    require(data, offset, 4, "u32")?;
    Ok(u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn invalid_data(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syne() -> TTFParser {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/data/Syne/static/Syne-Regular.ttf"
        );
        TTFParser::read_from(path).unwrap()
    }

    #[test]
    fn parses_required_tables_and_maps_bmp_characters() {
        let font = syne();
        let glyph_id = font.glyph_index('A' as u32);

        assert_eq!(font.units_per_em(), 1000);
        assert!(font.num_glyphs() > 1);
        assert_ne!(glyph_id, 0);
        assert_eq!(font.glyph_index(0x11_0000), 0);
        assert!(
            font.glyph_range(glyph_id)
                .is_some_and(|range| !range.is_empty())
        );
    }

    #[test]
    fn parses_simple_glyph_points_and_paths() {
        let font = syne();
        let glyph = font
            .load_glyph(font.glyph_index('S' as u32))
            .unwrap()
            .unwrap();

        assert!(!glyph.contours.is_empty());
        assert!(
            glyph
                .contours
                .iter()
                .all(|contour| !contour.points.is_empty())
        );
        assert!(matches!(
            glyph.path_commands().first(),
            Some(PathCommand::MoveTo(_))
        ));
        assert!(
            glyph
                .path_commands()
                .iter()
                .any(|command| matches!(command, PathCommand::QuadTo { .. }))
        );
    }

    #[test]
    fn reads_horizontal_metrics() {
        let font = syne();
        let glyph_id = font.glyph_index('A' as u32);
        let metrics = font.horizontal_metrics(glyph_id).unwrap();

        assert!(metrics.advance_width > 0);
        assert!(font.hhea().number_of_h_metrics <= font.num_glyphs());
        assert!(font.horizontal_metrics(font.num_glyphs()).is_none());
    }

    #[test]
    fn inserts_implied_points_between_off_curve_points() {
        let contour = Contour {
            points: vec![
                GlyphPoint {
                    x: 0.0,
                    y: 0.0,
                    on_curve: true,
                },
                GlyphPoint {
                    x: 10.0,
                    y: 10.0,
                    on_curve: false,
                },
                GlyphPoint {
                    x: 20.0,
                    y: 10.0,
                    on_curve: false,
                },
                GlyphPoint {
                    x: 30.0,
                    y: 0.0,
                    on_curve: true,
                },
            ],
        };

        let commands = contour.path_commands();
        assert_eq!(
            commands[1],
            PathCommand::QuadTo {
                control: Vec2 { x: 10.0, y: 10.0 },
                end: Vec2 { x: 15.0, y: 10.0 },
            }
        );
        assert_eq!(
            commands[2],
            PathCommand::QuadTo {
                control: Vec2 { x: 20.0, y: 10.0 },
                end: Vec2 { x: 30.0, y: 0.0 },
            }
        );
    }

    #[test]
    fn format_4_applies_glyph_array_offsets_and_delta() {
        let data = [
            0x00, 0x04, 0x00, 0x22, 0x00, 0x00, 0x00, 0x04, 0x00, 0x04, 0x00, 0x01, 0x00, 0x00,
            0x00, 0x41, 0xff, 0xff, 0x00, 0x00, 0x00, 0x41, 0xff, 0xff, 0x00, 0x02, 0x00, 0x01,
            0x00, 0x04, 0x00, 0x00, 0x00, 0x05,
        ];
        let cmap = parse_cmap_format4(&data).unwrap();

        assert_eq!(cmap.glyph_index('A' as u32), Some(7));
        assert_eq!(cmap.glyph_index('B' as u32), None);
    }

    #[test]
    fn hmtx_reuses_the_last_advance_width() {
        let data = [0x01, 0xf4, 0x00, 0x0a, 0xff, 0xec, 0x00, 0x1e];
        let metrics = parse_hmtx(&data, 3, 1).unwrap();

        assert_eq!(metrics[0].advance_width, 500);
        assert_eq!(
            metrics[1],
            HorizontalMetrics {
                advance_width: 500,
                left_side_bearing: -20,
            }
        );
        assert_eq!(
            metrics[2],
            HorizontalMetrics {
                advance_width: 500,
                left_side_bearing: 30,
            }
        );
    }
}


#[cfg(test)]
mod composite_tests {
    use super::*;

    fn header(contours: i16) -> Vec<u8> {
        [contours, -100, -100, 1000, 1000]
            .into_iter().flat_map(i16::to_be_bytes).collect()
    }

    fn simple(points: &[(i16, i16)]) -> Vec<u8> {
        assert!(!points.is_empty());
        let mut data = header(1);
        data.extend_from_slice(&((points.len() - 1) as u16).to_be_bytes());
        data.extend_from_slice(&0_u16.to_be_bytes()); // No instructions.
        data.extend(std::iter::repeat_n(ON_CURVE_POINT, points.len()));
        for axis in 0..2 {
            let mut previous = 0_i16;
            for &(x, y) in points {
                let coordinate = if axis == 0 { x } else { y };
                data.extend_from_slice(&(coordinate - previous).to_be_bytes());
                previous = coordinate;
            }
        }
        data
    }

    fn triangle() -> Vec<u8> {
        simple(&[(0, 0), (10, 0), (0, 10)])
    }

    fn record(flags: u16, id: u16, args: &[u8], transform: &[i16]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&flags.to_be_bytes());
        data.extend_from_slice(&id.to_be_bytes());
        data.extend_from_slice(args);
        for value in transform {
            data.extend_from_slice(&value.to_be_bytes());
        }
        data
    }

    fn composite(records: &[Vec<u8>], instructions: Option<&[u8]>) -> Vec<u8> {
        let mut data = header(-1);
        for (index, record) in records.iter().enumerate() {
            let mut flags = u16::from_be_bytes([record[0], record[1]]);
            if index + 1 < records.len() {
                flags |= MORE_COMPONENTS;
            }
            data.extend_from_slice(&flags.to_be_bytes());
            data.extend_from_slice(&record[2..]);
        }
        if let Some(instructions) = instructions {
            data.extend_from_slice(&(instructions.len() as u16).to_be_bytes());
            data.extend_from_slice(instructions);
        }
        data
    }

    // Exercise the public loader with synthetic glyf/loca data; no external font
    // fixture or third-party crate is required by these tests.
    fn font(glyphs: &[Vec<u8>]) -> TTFParser {
        let mut bytes = Vec::new();
        let mut offsets = vec![0];
        for glyph in glyphs {
            bytes.extend_from_slice(glyph);
            offsets.push(bytes.len() as u32);
        }
        let mut tables = HashMap::new();
        tables.insert(*b"glyf", TableRecord { offset: 0, length: bytes.len() });
        TTFParser {
            font_data: bytes,
            font_tables: tables,
            head_data: HeadData { units_per_em: 1000, index_to_loc_format: 1 },
            maxp_table: MaxpTable { version: 0x0001_0000, num_glyphs: glyphs.len() as u16 },
            hhea_table: HheaTable {
                ascender: 800, descender: -200, line_gap: 0,
                number_of_h_metrics: glyphs.len() as u16,
            },
            cmaps: Vec::new(),
            glyph_offsets: offsets,
            horizontal_metrics: vec![HorizontalMetrics {
                advance_width: 500, left_side_bearing: 0,
            }; glyphs.len()],
        }
    }

    fn points(glyph: &Glyph) -> Vec<(f32, f32)> {
        glyph.contours.iter().flat_map(|c| c.points.iter())
            .map(|p| (p.x, p.y)).collect()
    }

    #[test]
    fn signed_byte_offsets_and_contour_closure() {
        let data = composite(&[record(ARGS_ARE_XY_VALUES, 0, &[254, 100], &[])], None);
        let parser = font(&[triangle(), data]);
        let glyph = parser.load_glyph(1).unwrap().unwrap();
        assert_eq!(points(&glyph), [(-2.0, 100.0), (8.0, 100.0), (-2.0, 110.0)]);
        assert_eq!(glyph.path_commands().last(), Some(&PathCommand::LineTo(Vec2 {
            x: -2.0, y: 100.0,
        })));
    }

    #[test]
    fn signed_word_offsets() {
        let args: Vec<u8> = [-300_i16, 400].into_iter().flat_map(i16::to_be_bytes).collect();
        let data = composite(&[record(ARGS_ARE_XY_VALUES | ARG_1_AND_2_ARE_WORDS,
            0, &args, &[])], None);
        let glyph = font(&[triangle(), data]).load_glyph(1).unwrap().unwrap();
        assert_eq!(points(&glyph)[0], (-300.0, 400.0));
    }

    #[test]
    fn uniform_nonuniform_and_full_matrix_transforms() {
        let cases = [
            (WE_HAVE_A_SCALE, vec![8192], vec![(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]),
            (WE_HAVE_AN_X_AND_Y_SCALE, vec![8192, 24576],
                vec![(0.0, 0.0), (5.0, 0.0), (0.0, 15.0)]),
            // Stored order is xx, yx, xy, yy. Unequal off-diagonals detect transposition.
            (WE_HAVE_A_TWO_BY_TWO, vec![16384, 8192, -4096, 24576],
                vec![(0.0, 0.0), (10.0, 5.0), (-2.5, 15.0)]),
        ];
        for (flags, transform, expected) in cases {
            let data = composite(&[record(flags | ARGS_ARE_XY_VALUES,
                0, &[0, 0], &transform)], None);
            let glyph = font(&[triangle(), data]).load_glyph(1).unwrap().unwrap();
            assert_eq!(points(&glyph), expected);
        }
    }

    #[test]
    fn offset_scaling_default_and_conflicting_flags() {
        for (offset_flags, expected) in [
            (0, (10.0, 20.0)),
            (UNSCALED_COMPONENT_OFFSET, (10.0, 20.0)),
            (SCALED_COMPONENT_OFFSET, (5.0, 10.0)),
            (SCALED_COMPONENT_OFFSET | UNSCALED_COMPONENT_OFFSET, (10.0, 20.0)),
        ] {
            let data = composite(&[record(ARGS_ARE_XY_VALUES | WE_HAVE_A_SCALE | offset_flags,
                0, &[10, 20], &[8192])], None);
            let glyph = font(&[triangle(), data]).load_glyph(1).unwrap().unwrap();
            assert_eq!(points(&glyph)[0], expected);
        }
    }

    #[test]
    fn nested_components_and_repeated_siblings_are_allowed() {
        let child = composite(&[record(ARGS_ARE_XY_VALUES, 0, &[10, 20], &[])], None);
        let parent = composite(&[
            record(ARGS_ARE_XY_VALUES | WE_HAVE_A_SCALE, 1, &[100, 0], &[8192]),
            record(ARGS_ARE_XY_VALUES, 1, &[0, 0], &[]),
        ], None);
        let glyph = font(&[triangle(), child, parent]).load_glyph(2).unwrap().unwrap();
        assert_eq!(points(&glyph), [
            (105.0, 10.0), (110.0, 10.0), (105.0, 15.0),
            (10.0, 20.0), (20.0, 20.0), (10.0, 30.0),
        ]);
    }

    #[test]
    fn point_attachment_after_transform() {
        let data = composite(&[
            record(ARGS_ARE_XY_VALUES, 0, &[100, 100], &[]),
            record(WE_HAVE_A_SCALE, 0, &[1, 2], &[8192]),
        ], None);
        let glyph = font(&[triangle(), data]).load_glyph(1).unwrap().unwrap();
        assert_eq!(points(&glyph)[3], (110.0, 95.0));
        assert_eq!(points(&glyph)[5], points(&glyph)[1]);
    }

    #[test]
    fn unsigned_point_indices_in_byte_and_word_forms() {
        let leaf = simple(&(0_i16..201).map(|x| (x, 0)).collect::<Vec<_>>());
        for (flags, args) in [(0, vec![200, 200]),
            (ARG_1_AND_2_ARE_WORDS, vec![0, 200, 0, 200])] {
            let data = composite(&[
                record(ARGS_ARE_XY_VALUES, 0, &[10, 20], &[]),
                record(flags, 0, &args, &[]),
            ], None);
            let glyph = font(&[leaf.clone(), data]).load_glyph(1).unwrap().unwrap();
            assert_eq!(points(&glyph)[401], (210.0, 20.0));
        }
    }

    #[test]
    fn attachment_indices_span_contours_and_exclude_implied_points() {
        let mut leaf = triangle();
        leaf[15] = 0;
        leaf[16] = 0; // Consecutive off-curve controls imply an extra path point.
        let child = composite(&[
            record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[]),
            record(ARGS_ARE_XY_VALUES, 0, &[100, 0], &[]),
        ], None);
        let parent = composite(&[
            record(ARGS_ARE_XY_VALUES, 1, &[0, 0], &[]),
            record(0, 0, &[4, 1], &[]),
        ], None);
        let glyph = font(&[leaf, child, parent]).load_glyph(2).unwrap().unwrap();
        assert_eq!(points(&glyph)[7], points(&glyph)[4]);
        assert!(!glyph.contours[2].points[1].on_curve);
    }

    #[test]
    fn instructions_on_nonfinal_component_are_checked() {
        let records = [
            record(ARGS_ARE_XY_VALUES | WE_HAVE_INSTRUCTIONS, 0, &[0, 0], &[]),
            record(ARGS_ARE_XY_VALUES, 0, &[10, 0], &[]),
        ];
        let data = composite(&records, Some(&[0, 1, 2]));
        assert!(font(&[triangle(), data.clone()]).load_glyph(1).is_ok());
        let mut truncated = data;
        truncated.pop();
        assert_eq!(font(&[triangle(), truncated]).load_glyph(1).unwrap_err().kind(),
            ErrorKind::UnexpectedEof);
    }

    #[test]
    fn malformed_components_and_phantom_points_are_reported() {
        let invalid_records = [
            record(ARGS_ARE_XY_VALUES, 99, &[0, 0], &[]),
            record(ARGS_ARE_XY_VALUES | WE_HAVE_A_SCALE | WE_HAVE_A_TWO_BY_TWO,
                0, &[0, 0], &[]),
            record(0, 0, &[0, 0], &[]), // First component cannot use point attachment.
        ];
        for record in invalid_records {
            let data = composite(&[record], None);
            assert_eq!(font(&[triangle(), data]).load_glyph(1).unwrap_err().kind(),
                ErrorKind::InvalidData);
        }
        for (index, expected) in [(3, ErrorKind::Unsupported), (100, ErrorKind::InvalidData)] {
            let data = composite(&[
                record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[]),
                record(0, 0, &[0, index], &[]),
            ], None);
            assert_eq!(font(&[triangle(), data]).load_glyph(1).unwrap_err().kind(), expected);
        }
    }

    #[test]
    fn truncated_component_records_are_rejected() {
        let data = composite(&[record(ARGS_ARE_XY_VALUES | WE_HAVE_A_TWO_BY_TWO,
            0, &[0, 0], &[16384, 0, 0, 16384])], None);
        for end in 10..data.len() {
            let truncated = data[..end].to_vec();
            assert_eq!(font(&[triangle(), truncated]).load_glyph(1).unwrap_err().kind(),
                ErrorKind::UnexpectedEof);
        }
    }

    #[test]
    fn empty_components_and_missing_glyph_data() {
        let data = composite(&[record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[])], None);
        let parser = font(&[Vec::new(), data]);
        assert!(parser.load_glyph(0).unwrap().is_none());
        assert!(parser.load_glyph(1).unwrap().unwrap().contours.is_empty());
    }

    #[test]
    fn cycles_and_depth_limits_are_rejected() {
        let direct = composite(&[record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[])], None);
        assert_eq!(font(&[direct]).load_glyph(0).unwrap_err().kind(), ErrorKind::InvalidData);
        let a = composite(&[record(ARGS_ARE_XY_VALUES, 1, &[0, 0], &[])], None);
        let b = composite(&[record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[])], None);
        assert_eq!(font(&[a, b]).load_glyph(0).unwrap_err().kind(), ErrorKind::InvalidData);
        let mut glyphs = vec![triangle()];
        for id in 0..MAX_GLYPH_LOAD_DEPTH as u16 {
            glyphs.push(composite(&[record(ARGS_ARE_XY_VALUES, id, &[0, 0], &[])], None));
        }
        assert!(font(&glyphs).load_glyph(MAX_GLYPH_LOAD_DEPTH as u16 - 1).is_ok());
        assert_eq!(font(&glyphs).load_glyph(MAX_GLYPH_LOAD_DEPTH as u16)
            .unwrap_err().kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn expansion_budgets_apply_across_siblings() {
        let data = composite(&[
            record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[]),
            record(ARGS_ARE_XY_VALUES, 0, &[0, 0], &[]),
        ], None);
        let parser = font(&[triangle(), data]);
        for (visits, points) in [(2, 100), (100, 5)] {
            let mut active = Vec::new();
            let mut budget = GlyphLoadBudget { visits, points };
            assert_eq!(parser.load_glyph_inner(1, &mut active, &mut budget)
                .unwrap_err().kind(), ErrorKind::InvalidData);
            assert!(active.is_empty());
        }
    }
}
