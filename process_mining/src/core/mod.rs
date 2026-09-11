//! Core modules for process mining

pub use chrono;
pub mod event_data;
/// DfCM-compatible economic ISA wire boundary for process evidence.
pub mod economic_isa;

/// IO Traits
pub mod io;
/// Bytes of a tabular data file, held for an extraction to read.
pub mod tabular_source;

pub mod process_models;

pub use event_data::case_centric::EventLog;
pub use event_data::object_centric::OCEL;
pub use process_models::case_centric::petri_net::PetriNet;
