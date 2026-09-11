//! DfCM-compatible economic ISA wire boundary.
//!
//! This module intentionally does **not** own the economic opcode vocabulary.
//! The canonical registry is defined upstream and may be manufactured into
//! language-specific tables. `rust4pm` only preserves the wire identity needed
//! for process evidence and asks a caller-supplied [`AssignedRegistry`] whether
//! a non-sentinel byte is currently assigned.
//!
//! The central invariant is:
//!
//! `economic verb identity != event != plan != authority != economic value`.
//!
//! In particular, admitting or observing a byte never grants `DO` authority.

/// Explicit sentinel for unknown economic meaning.
pub const UNKNOWN: u8 = 0x00;

/// Escape byte for out-of-band extension identifiers.
pub const ESCAPE: u8 = 0xFF;

/// Stable structural class of an economic ISA byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EconomicByteClass {
    /// The explicit unknown sentinel (`0x00`).
    Unknown,
    /// A byte admitted as assigned by the supplied registry.
    Assigned,
    /// A non-sentinel byte not assigned by the supplied registry.
    Unassigned,
    /// The extension escape byte (`0xFF`).
    Escape,
}

/// Provider contract for the canonical assigned-byte set.
///
/// Implementations should be generated or adapted from the canonical registry;
/// this trait exists so `rust4pm` does not manufacture a second hand-maintained
/// semantic table.
pub trait AssignedRegistry {
    /// Return `true` only when `byte` is assigned by the admitted registry.
    fn is_assigned(&self, byte: u8) -> bool;
}

impl<F> AssignedRegistry for F
where
    F: Fn(u8) -> bool,
{
    fn is_assigned(&self, byte: u8) -> bool {
        self(byte)
    }
}

/// Wire frame preserving the one-byte core and optional extension identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EconomicFrame {
    /// Core opcode byte or [`ESCAPE`].
    pub opcode: u8,
    /// Out-of-band extension identifier, valid only when `opcode == ESCAPE`.
    pub extension_id: Option<u32>,
}

/// Admitted structural frame.
///
/// This type is suitable for process evidence. It carries no actuation token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdmittedEconomicFrame {
    /// Explicit unknown semantic identity.
    Unknown,
    /// Assigned canonical core byte.
    Core(u8),
    /// Extension identity selected through [`ESCAPE`].
    Extension(u32),
}

/// Named refusals at the economic wire boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EconomicWireRefusal {
    /// A byte is not assigned by the supplied canonical registry.
    UnassignedByte(u8),
    /// [`ESCAPE`] was present without an extension identifier.
    EscapeRequiresExtension,
    /// An extension identifier was attached to a non-escape byte.
    UnexpectedExtension,
}

/// Observation record that can be attached to process evidence without
/// pretending that semantic observation is authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EconomicObservation {
    /// Original wire byte.
    pub opcode: u8,
    /// Original extension identifier, if any.
    pub extension_id: Option<u32>,
    /// Registry-aware structural classification.
    pub class: EconomicByteClass,
}

/// Classify a byte using the caller-supplied canonical registry.
#[must_use]
pub fn classify_with_registry<R>(byte: u8, registry: &R) -> EconomicByteClass
where
    R: AssignedRegistry + ?Sized,
{
    match byte {
        UNKNOWN => EconomicByteClass::Unknown,
        ESCAPE => EconomicByteClass::Escape,
        other if registry.is_assigned(other) => EconomicByteClass::Assigned,
        _ => EconomicByteClass::Unassigned,
    }
}

/// Produce an observation while preserving the original wire coordinates.
#[must_use]
pub fn observe_with_registry<R>(frame: EconomicFrame, registry: &R) -> EconomicObservation
where
    R: AssignedRegistry + ?Sized,
{
    EconomicObservation {
        opcode: frame.opcode,
        extension_id: frame.extension_id,
        class: classify_with_registry(frame.opcode, registry),
    }
}

/// Admit a frame against the supplied canonical registry.
///
/// Admission validates only semantic wire structure. It does not authorize an
/// external effect, settlement, transfer, payment, or any other `DO` action.
pub fn admit_with_registry<R>(
    frame: EconomicFrame,
    registry: &R,
) -> Result<AdmittedEconomicFrame, EconomicWireRefusal>
where
    R: AssignedRegistry + ?Sized,
{
    match (frame.opcode, frame.extension_id) {
        (ESCAPE, Some(extension_id)) => Ok(AdmittedEconomicFrame::Extension(extension_id)),
        (ESCAPE, None) => Err(EconomicWireRefusal::EscapeRequiresExtension),
        (_, Some(_)) => Err(EconomicWireRefusal::UnexpectedExtension),
        (UNKNOWN, None) => Ok(AdmittedEconomicFrame::Unknown),
        (byte, None) if registry.is_assigned(byte) => Ok(AdmittedEconomicFrame::Core(byte)),
        (byte, None) => Err(EconomicWireRefusal::UnassignedByte(byte)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_registry(byte: u8) -> bool {
        matches!(byte, 0x01 | 0x20 | 0x41 | 0x61 | 0xA0 | 0xC0 | 0xE0)
    }

    #[test]
    fn preserves_all_four_dfcm_byte_states() {
        assert_eq!(
            classify_with_registry(UNKNOWN, &fixture_registry),
            EconomicByteClass::Unknown
        );
        assert_eq!(
            classify_with_registry(0x41, &fixture_registry),
            EconomicByteClass::Assigned
        );
        assert_eq!(
            classify_with_registry(0x44, &fixture_registry),
            EconomicByteClass::Unassigned
        );
        assert_eq!(
            classify_with_registry(ESCAPE, &fixture_registry),
            EconomicByteClass::Escape
        );
    }

    #[test]
    fn unknown_is_not_collapsed_into_an_assigned_operation() {
        assert_eq!(
            admit_with_registry(
                EconomicFrame {
                    opcode: UNKNOWN,
                    extension_id: None,
                },
                &fixture_registry,
            ),
            Ok(AdmittedEconomicFrame::Unknown)
        );
    }

    #[test]
    fn unassigned_byte_is_refused_not_guessed() {
        assert_eq!(
            admit_with_registry(
                EconomicFrame {
                    opcode: 0x44,
                    extension_id: None,
                },
                &fixture_registry,
            ),
            Err(EconomicWireRefusal::UnassignedByte(0x44))
        );
    }

    #[test]
    fn extension_boundary_is_explicit() {
        assert_eq!(
            admit_with_registry(
                EconomicFrame {
                    opcode: ESCAPE,
                    extension_id: Some(17),
                },
                &fixture_registry,
            ),
            Ok(AdmittedEconomicFrame::Extension(17))
        );
        assert_eq!(
            admit_with_registry(
                EconomicFrame {
                    opcode: ESCAPE,
                    extension_id: None,
                },
                &fixture_registry,
            ),
            Err(EconomicWireRefusal::EscapeRequiresExtension)
        );
        assert_eq!(
            admit_with_registry(
                EconomicFrame {
                    opcode: 0x41,
                    extension_id: Some(17),
                },
                &fixture_registry,
            ),
            Err(EconomicWireRefusal::UnexpectedExtension)
        );
    }

    #[test]
    fn exhaustive_classification_is_total_over_byte_space() {
        for byte in u8::MIN..=u8::MAX {
            let class = classify_with_registry(byte, &fixture_registry);
            match class {
                EconomicByteClass::Unknown => assert_eq!(byte, UNKNOWN),
                EconomicByteClass::Escape => assert_eq!(byte, ESCAPE),
                EconomicByteClass::Assigned => assert!(fixture_registry(byte)),
                EconomicByteClass::Unassigned => {
                    assert_ne!(byte, UNKNOWN);
                    assert_ne!(byte, ESCAPE);
                    assert!(!fixture_registry(byte));
                }
            }
        }
    }
}
