//! `.sqlite` source -> `DuckDB`: materialize a full OCEL, then feed it to the streaming sink.
#![cfg(all(feature = "ocel-duckdb", feature = "ocel-sqlite"))]

use std::path::Path;

use crate::core::event_data::object_centric::appendable::AppendableOCEL;
use crate::core::event_data::object_centric::io::OCELIOError;
use crate::core::event_data::object_centric::ocel_sql::import_ocel_sqlite_from_path;

use super::stream::{run_import, DuckDbImportOptions};

/// Load a `.sqlite` OCEL into `DuckDB`, reached via
/// [`stream_ocel_file_to_duckdb`](super::stream::stream_ocel_file_to_duckdb).
///
/// v1 reuses `import_ocel_sqlite_from_path` (whole-file read), so `DuckDB` write memory is
/// bounded but `SQLite` read memory is not. `SQLite` reads are whole-file today anyway (no
/// regression); true row-streaming on the `SQLite` side is a future optimization.
pub(super) fn stream_ocel_sqlite_to_duckdb<P: AsRef<Path>, Q: AsRef<Path>>(
    sqlite_path: P,
    db_path: Q,
    options: &DuckDbImportOptions,
) -> Result<(), OCELIOError> {
    let ocel = import_ocel_sqlite_from_path(sqlite_path)?;
    run_import(db_path.as_ref(), options, |sink| {
        for et in &ocel.event_types {
            sink.declare_event_type(et.clone())
                .map_err(OCELIOError::from)?;
        }
        for ot in &ocel.object_types {
            sink.declare_object_type(ot.clone())
                .map_err(OCELIOError::from)?;
        }
        for e in &ocel.events {
            sink.append_event(
                e.id.clone(),
                &e.event_type,
                e.time,
                e.attributes.clone(),
                e.relationships.clone(),
            )
            .map_err(OCELIOError::from)?;
        }
        for o in &ocel.objects {
            sink.append_object(
                o.id.clone(),
                &o.object_type,
                o.attributes.clone(),
                o.relationships.clone(),
            )
            .map_err(OCELIOError::from)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use crate::core::event_data::object_centric::ocel_sql::{
        import_ocel_sqlite_from_path, stream_ocel_file_to_duckdb,
    };
    use crate::test_utils::get_test_data_path;
    use duckdb::Connection;

    #[test]
    fn sqlite_stream_roundtrip_counts() {
        let src = get_test_data_path()
            .join("ocel")
            .join("order-management.sqlite");
        let reference = import_ocel_sqlite_from_path(&src).unwrap();
        let out = get_test_data_path()
            .join("export")
            .join("stream-from-sqlite.duckdb");
        let _ = std::fs::remove_file(&out);
        stream_ocel_file_to_duckdb(&src, &out).unwrap();
        let con = Connection::open(&out).unwrap();
        let n_ev: i64 = con
            .query_row("SELECT count(*) FROM events", [], |r| r.get(0))
            .unwrap();
        let n_ob: i64 = con
            .query_row("SELECT count(*) FROM objects", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_ev as usize, reference.events.len());
        assert_eq!(n_ob as usize, reference.objects.len());
    }
}
