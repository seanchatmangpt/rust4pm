//! Table/index schema (`CREATE` statements) and shared table/column name constants.
use duckdb::Connection;

use crate::core::event_data::object_centric::ocel_struct::OCELAttributeType;

pub(crate) const T_EVENTS: &str = "events";
pub(crate) const T_OBJECTS: &str = "objects";
pub(crate) const T_EVENT_ATTR_META: &str = "event_attr_meta";
pub(crate) const T_OBJECT_ATTR_CHANGES: &str = "object_attribute_changes";
pub(crate) const T_E2O: &str = "e2o";
pub(crate) const T_O2O: &str = "o2o";

// Static `CREATE TABLE`s for everything except `events`
// (wide `events` table is built dynamically (`build_events_table`) once its typed attribute columns are known)
const CREATE_TABLES: &str = r#"
CREATE TABLE objects (id TEXT PRIMARY KEY, ocel_type TEXT);
CREATE TABLE object_attribute_changes (id TEXT, name TEXT, "time" TIMESTAMPTZ, value VARCHAR, value_type TEXT);
CREATE TABLE event_attr_meta (event_type TEXT, attr_name TEXT, attr_type TEXT);
CREATE TABLE e2o (event_id TEXT, object_id TEXT, qualifier TEXT);
CREATE TABLE o2o (source_id TEXT, target_id TEXT, qualifier TEXT);
"#;

const INDEXES: &str = r#"
CREATE INDEX object_attribute_changes_id ON object_attribute_changes(id);
CREATE INDEX e2o_event ON e2o(event_id);
CREATE INDEX e2o_object ON e2o(object_id);
CREATE INDEX o2o_source ON o2o(source_id);
CREATE INDEX o2o_target ON o2o(target_id);
"#;

pub(crate) fn create_schema(con: &Connection) -> Result<(), duckdb::Error> {
    con.execute_batch(CREATE_TABLES)
}

pub(crate) fn create_indexes(con: &Connection) -> Result<(), duckdb::Error> {
    con.execute_batch(INDEXES)
}

/// SQL-quote an identifier by doubling embedded double-quotes.
pub(crate) fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// `DuckDB` column type for an event-attribute [`OCELAttributeType`]. `String`/`Null`
/// (and the collision downgrade) store as text (`VARCHAR`).
pub(crate) fn event_attr_sql_type(t: OCELAttributeType) -> &'static str {
    match t {
        OCELAttributeType::Integer => "BIGINT",
        OCELAttributeType::Float => "DOUBLE",
        OCELAttributeType::Boolean => "BOOLEAN",
        OCELAttributeType::Time => "TIMESTAMPTZ",
        OCELAttributeType::String | OCELAttributeType::Null => "VARCHAR",
    }
}

/// Add a column for an attribute never declared on an event type. Always `VARCHAR`: the
/// type is unknown and may vary between events, so only text holds every value.
pub(crate) fn add_events_column(name: &str) -> String {
    format!(
        "ALTER TABLE {T_EVENTS} ADD COLUMN {} VARCHAR",
        quote_ident(name)
    )
}

/// Build the `CREATE TABLE events` statement: fixed `id`/`ocel_type`/`"time"` columns plus one
/// typed, quoted column per `(name, type)` in `cols` (order preserved).
pub(crate) fn build_events_table(cols: &[(String, OCELAttributeType)]) -> String {
    let mut sql = String::from(
        r#"CREATE TABLE events (id TEXT PRIMARY KEY, ocel_type TEXT, "time" TIMESTAMPTZ"#,
    );
    for (name, ty) in cols {
        sql.push_str(", ");
        sql.push_str(&quote_ident(name));
        sql.push(' ');
        sql.push_str(event_attr_sql_type(*ty));
    }
    sql.push(')');
    sql
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_schema_makes_all_static_tables() {
        // `events` is intentionally absent: it is created dynamically by the sink once
        // the event-attribute schema is known.
        let con = duckdb::Connection::open_in_memory().unwrap();
        create_schema(&con).unwrap();
        let mut stmt = con
            .prepare("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name")
            .unwrap();
        let names: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "e2o",
                "event_attr_meta",
                "o2o",
                "object_attribute_changes",
                "objects",
            ]
        );
    }

    #[test]
    fn build_events_table_quotes_and_types_columns() {
        let sql = build_events_table(&[
            ("amount".to_string(), OCELAttributeType::Float),
            ("weird \"name\"".to_string(), OCELAttributeType::String),
        ]);
        assert!(sql.contains(r#"id TEXT PRIMARY KEY, ocel_type TEXT, "time" TIMESTAMPTZ"#));
        assert!(sql.contains(r#""amount" DOUBLE"#));
        assert!(sql.contains(r#""weird ""name""" VARCHAR"#));
        let con = duckdb::Connection::open_in_memory().unwrap();
        con.execute_batch(&sql).unwrap();
    }

    #[test]
    fn timestamps_are_utc_anchored() {
        // TIMESTAMPTZ, not a bare TIMESTAMP, so stored instants are unambiguous.
        let con = duckdb::Connection::open_in_memory().unwrap();
        create_schema(&con).unwrap();
        con.execute_batch(&build_events_table(&[(
            "at".to_string(),
            OCELAttributeType::Time,
        )]))
        .unwrap();
        for (table, column) in [
            ("events", "time"),
            ("events", "at"),
            ("object_attribute_changes", "time"),
        ] {
            let ty: String = con
                .query_row(
                    "SELECT data_type FROM information_schema.columns \
                     WHERE table_name = ? AND column_name = ?",
                    [table, column],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(ty, "TIMESTAMP WITH TIME ZONE", "{table}.{column}");
        }
    }
}
