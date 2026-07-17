use super::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// The bug this type exists to stop: 90.0 reads as "90 percent" but the setting is a fraction,
// so it would pass a lower-bound-only check and then never trigger a resize.
#[test]
fn rejects_percent_shaped_ninety() {
    let error = ResizeTriggerFraction::new(90.0).unwrap_err();
    assert_eq!(error.rejected(), 90.0);
}

#[test]
fn rejects_values_above_one() {
    for rejected in [1.5, 2.0, 90.0, 100.0, f32::MAX, f32::INFINITY] {
        assert!(ResizeTriggerFraction::new(rejected).is_err(), "{} must be rejected", rejected);
    }
}

// The smallest value that is still above 1.0, to pin the upper bound exactly.
#[test]
fn rejects_the_value_just_above_one() {
    let just_above_one = f32::from_bits(1.0_f32.to_bits() + 1);
    assert!(just_above_one > 1.0);
    assert!(ResizeTriggerFraction::new(just_above_one).is_err(), "{} must be rejected", just_above_one);
}

#[test]
fn rejects_negative_values() {
    for rejected in [-0.1, -1.0, -90.0, f32::MIN, f32::NEG_INFINITY, -f32::MIN_POSITIVE] {
        assert!(ResizeTriggerFraction::new(rejected).is_err(), "{} must be rejected", rejected);
    }
}

// The largest value that is still below 0.0, to pin the lower bound exactly.
// Stepping one bit up from sign-bit-set zero moves away from zero downward, which lands on the
// smallest negative subnormal; the fixed negative cases above only reach the smallest negative
// normal, so without this the lower edge itself is never tested.
#[test]
fn rejects_the_value_just_below_zero() {
    let just_below_zero = f32::from_bits((-0.0_f32).to_bits() + 1);
    assert!(just_below_zero < 0.0);
    assert!(ResizeTriggerFraction::new(just_below_zero).is_err(), "{} must be rejected", just_below_zero);
}

// NaN compares false against every bound, so it must fall out as rejected rather than sneak in.
#[test]
fn rejects_nan() {
    assert!(ResizeTriggerFraction::new(f32::NAN).is_err());
}

// Both ends are legal per the documented range, and each end has a meaning worth keeping:
// 0.0 resizes as soon as anything is stored, 1.0 leaves only headroom-requested resizes.
#[test]
fn accepts_inclusive_bounds() {
    assert_eq!(ResizeTriggerFraction::new(0.0).unwrap().as_f32(), 0.0);
    assert_eq!(ResizeTriggerFraction::new(1.0).unwrap().as_f32(), 1.0);
}

#[test]
fn accepts_typical_fractions() {
    for accepted in [0.01, 0.25, 0.5, 0.9, 0.99] {
        let fraction = ResizeTriggerFraction::new(accepted).unwrap();
        assert_eq!(fraction.as_f32(), accepted);
    }
}

// Negative zero equals zero under IEEE comparison, so it is in range.
// Bits are compared rather than values, because -0.0 == 0.0 is true, so an equality check
// would pass even if the stored value had lost the sign.
#[test]
fn accepts_negative_zero() {
    let stored = ResizeTriggerFraction::new(-0.0).unwrap().as_f32();
    assert_eq!(stored.to_bits(), (-0.0_f32).to_bits());
}

// The message has to name the value and the unit, because the whole failure mode is a
// reader assuming percent where a fraction is wanted.
#[test]
fn error_message_names_the_rejected_value_and_the_unit() {
    let message = ResizeTriggerFraction::new(90.0).unwrap_err().to_string();
    // - match the interpolated phrase, not a bare "90": the static text already reads
    //   "90% full", so contains("90") would still pass even if the rejected value were
    //   never interpolated, making the value half of this test a tautology
    assert!(message.contains("but 90 was given"), "{}", message);
    assert!(message.contains("fraction"), "{}", message);
}

#[test]
fn default_settings_trigger_at_ninety_percent_and_validate() {
    assert_eq!(DEFAULT_RESIZE_SETTINGS.resize_trigger_fraction.as_f32(), 0.9);
    DEFAULT_RESIZE_SETTINGS.validate();
    DatabaseResizeSettings::default().validate();
}

// The type has to hold up inside the settings struct it exists for, not only on its own.
#[test]
fn settings_carrying_a_valid_fraction_validate() {
    let settings = DatabaseResizeSettings {
        min_resize_step: 1 << 20,
        max_resize_step: 1 << 21,
        default_resize_ratio_percentage: 10,
        resize_trigger_fraction: ResizeTriggerFraction::new(0.75).unwrap(),
    };
    settings.validate();
    assert_eq!(settings.resize_trigger_fraction.as_f32(), 0.75);
}

#[test]
fn random_in_range_fractions_are_accepted() {
    // Seed is printed so a failing run can be replayed by hardcoding it.
    let seed: u64 = rand::random();
    println!("random_in_range_fractions_are_accepted seed: {}", seed);
    // Replaying a hardcoded seed is valid only while rand stays pinned to
    // 0.8; StdRng's algorithm is not stable across rand major versions.
    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..1000 {
        let fraction: f32 = rng.gen_range(0.0..=1.0);
        let built = ResizeTriggerFraction::new(fraction);
        assert!(built.is_ok(), "seed {}: {} is in range and must be accepted", seed, fraction);
        assert_eq!(built.unwrap().as_f32(), fraction, "seed {}: value must round-trip", seed);
    }
}

#[test]
fn random_out_of_range_fractions_are_rejected() {
    // Seed is printed so a failing run can be replayed by hardcoding it.
    let seed: u64 = rand::random();
    println!("random_out_of_range_fractions_are_rejected seed: {}", seed);
    // Replaying a hardcoded seed is valid only while rand stays pinned to
    // 0.8; StdRng's algorithm is not stable across rand major versions.
    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..1000 {
        let above: f32 = rng.gen_range(1.0001..1.0e6);
        assert!(ResizeTriggerFraction::new(above).is_err(), "seed {}: {} is above 1 and must be rejected", seed, above);

        let below: f32 = rng.gen_range(-1.0e6..-0.0001);
        assert!(ResizeTriggerFraction::new(below).is_err(), "seed {}: {} is below 0 and must be rejected", seed, below);
    }
}
