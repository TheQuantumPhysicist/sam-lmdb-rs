use std::error::Error as StdError;
use std::fmt;

/// The information that will be sent in a callback when a resize happens
#[derive(Debug, Clone)]
pub struct DatabaseResizeInfo {
    /// The size of the database before the resize
    pub old_size: u64,
    /// The size of the database after the resize (current size)
    pub new_size: u64,
    /// The estimated occupied size before the resize happened
    pub occupied_size_before_resize: u64,
}

const DEFAULT_MIN_MAP_SIZE_INCREASE: usize = 1 << 28; // 256 MB
const DEFAULT_MAX_MAP_SIZE_INCREASE: usize = 1 << 31; // 2 GB
const DEFAULT_RESIZE_RATIO: u32 = 100; // 100%, double the current storage
const DEFAULT_RESIZE_TRIGGER: ResizeTriggerFraction = ResizeTriggerFraction::from_literal(0.9); // 90% full, causes resize

/// How full the database may get before a resize is triggered, as a fraction of the map size.
///
/// - the value is a fraction in `[0, 1]`, so 90% full is `0.9`, not `90.0`
/// - the sibling setting `DatabaseResizeSettings::default_resize_ratio_percentage` is an
///   integer percent, which makes the two easy to confuse
/// - a percent-shaped value such as `90.0` would never be crossed by a real fill fraction, so
///   it would silently disable fill-triggered resizing and only surface much later as a
///   `MapFull` error, far from the setting that caused it
/// - this type exists so such a value is rejected where it is written, not where it hurts
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResizeTriggerFraction(f32);

impl ResizeTriggerFraction {
    /// Builds a trigger fraction from a value in `[0, 1]`, both ends included.
    ///
    /// - `0.0` triggers a resize as soon as anything at all is stored
    /// - `1.0` means a fill level alone never triggers a resize, because the fill fraction
    ///   cannot exceed the map size; resizes then only happen when a caller asks for headroom
    /// - anything outside `[0, 1]`, and NaN, is rejected
    pub const fn new(fraction: f32) -> Result<Self, InvalidResizeTriggerFraction> {
        if Self::is_in_range(fraction) {
            Ok(Self(fraction))
        } else {
            Err(InvalidResizeTriggerFraction {
                rejected: fraction,
            })
        }
    }

    /// The fill fraction, guaranteed to be within `[0, 1]`.
    pub const fn as_f32(&self) -> f32 {
        self.0
    }

    // - const path for literals written inside this crate, so an out-of-range default breaks
    //   the build rather than a user's process
    // - private on purpose: a public const constructor would panic on runtime values, which is
    //   what `new` already handles fallibly
    // - goes through `new` so the range check lives in exactly one place, and a literal that
    //   fails it turns into a compile error at the const site that wrote it
    const fn from_literal(fraction: f32) -> Self {
        match Self::new(fraction) {
            Ok(valid) => valid,
            Err(_) => panic!("lmdb: Resize trigger fraction must be within [0, 1]"),
        }
    }

    // NaN compares false against everything, so NaN lands outside the range, which is intended.
    const fn is_in_range(fraction: f32) -> bool {
        fraction >= 0. && fraction <= 1.
    }
}

/// The error returned when a value offered as a [`ResizeTriggerFraction`] is not in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidResizeTriggerFraction {
    rejected: f32,
}

impl InvalidResizeTriggerFraction {
    /// The value that was rejected.
    pub const fn rejected(&self) -> f32 {
        self.rejected
    }
}

impl fmt::Display for InvalidResizeTriggerFraction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "lmdb: Resize trigger must be a fraction within [0, 1], but {} was given. \
             This is a fraction and not a percent, so 90% full is 0.9",
            self.rejected
        )
    }
}

impl StdError for InvalidResizeTriggerFraction {}

/// Settings that control resizing the database
#[derive(Debug, Clone)]
pub struct DatabaseResizeSettings {
    /// The minimum amount to increase the size of the database by
    pub min_resize_step: usize,
    /// The maximum amount to increase the size of the database by
    pub max_resize_step: usize,
    /// When a resize is needed and no size is provided, this will be the size ratio to be added compared to the previous size
    /// 100 means 100%, which will double the map size
    pub default_resize_ratio_percentage: u32,
    /// When occupied_size/total_size crosses this fraction, a resize will be triggered
    pub resize_trigger_fraction: ResizeTriggerFraction,
}

impl Default for DatabaseResizeSettings {
    fn default() -> Self {
        Self::make_default()
    }
}

fn assert_unsigned<T: num::Unsigned>(_: T) {}

impl DatabaseResizeSettings {
    const fn make_default() -> Self {
        Self {
            min_resize_step: DEFAULT_MIN_MAP_SIZE_INCREASE,
            max_resize_step: DEFAULT_MAX_MAP_SIZE_INCREASE,
            default_resize_ratio_percentage: DEFAULT_RESIZE_RATIO,
            resize_trigger_fraction: DEFAULT_RESIZE_TRIGGER,
        }
    }

    // resize_trigger_fraction is absent below because ResizeTriggerFraction cannot be built
    // out of range, so its bounds are enforced by the type instead of by a check here.
    pub(crate) fn validate(&self) {
        // The check below assumes that the type is unsigned
        assert_unsigned(self.min_resize_step);
        assert!(self.min_resize_step != 0, "lmdb: Min step must be positive");

        // The check below assumes that the type is unsigned
        assert_unsigned(self.max_resize_step);
        assert!(self.max_resize_step != 0, "lmdb: Max step must be positive");

        assert!(self.min_resize_step <= self.max_resize_step, "lmdb: Min step must be <= max step");

        // The check below assumes that the type is unsigned
        assert_unsigned(self.default_resize_ratio_percentage);
        assert!(self.default_resize_ratio_percentage != 0, "lmdb: Resize ratio must be > 0");
    }
}

pub const DEFAULT_RESIZE_SETTINGS: DatabaseResizeSettings = DatabaseResizeSettings::make_default();

#[cfg(test)]
mod tests;
