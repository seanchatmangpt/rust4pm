//! Compare OCEL 2.0 import speed across importers, per source format:
//! - `ocel_direct`   -> `OCEL::import_from_path` (materialize full log in memory as OCEL)
//! - `slim_streaming`-> `SlimLinkedOCEL::import_from_path` (stream into the SlimLinkedOCEL)
//! - `duckdb_*`      -> `stream_ocel_file_to_duckdb` (stream into a DuckDB database)
//!     - `duckdb_raw` / `duckdb_default`: on-disk file
//          - `_raw` = pure streaming-append,
//!         - `_default` adds compression + cluster-by-key optimize + index building
//!     - `duckdb_mem_*`: same, but an in-memory (`:memory:`) database
use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use process_mining::{
    core::event_data::object_centric::{
        linked_ocel::SlimLinkedOCEL,
        ocel_sql::{stream_ocel_file_to_duckdb_with, DuckDbImportOptions},
    },
    test_utils::get_test_data_path,
    Importable, OCEL,
};

fn fixture(ext: &str) -> PathBuf {
    get_test_data_path()
        .join("ocel")
        .join(format!("order-management.{ext}"))
}

fn bench_format(c: &mut Criterion, ext: &str) {
    let src = fixture(ext);
    let out = std::env::temp_dir().join(format!("bench_import_{ext}.duckdb"));
    let opt_default = DuckDbImportOptions::default();
    let opt_raw = DuckDbImportOptions {
        compression: false,
        optimize_filesize: false,
    };

    let mut g = c.benchmark_group(format!("ocel_import_{ext}"));
    g.sample_size(10); // multi-MB fixtures: keep the sample count low

    g.bench_function("ocel_direct", |b| {
        b.iter(|| black_box(OCEL::import_from_path(black_box(&src)).unwrap()))
    });
    g.bench_function("slim_streaming", |b| {
        b.iter(|| black_box(SlimLinkedOCEL::import_from_path(black_box(&src)).unwrap()))
    });
    g.bench_function("duckdb_default", |b| {
        b.iter_batched(
            || {
                let _ = std::fs::remove_file(&out);
            },
            |()| stream_ocel_file_to_duckdb_with(&src, &out, &opt_default).unwrap(),
            BatchSize::SmallInput,
        )
    });
    g.bench_function("duckdb_raw", |b| {
        b.iter_batched(
            || {
                let _ = std::fs::remove_file(&out);
            },
            |()| stream_ocel_file_to_duckdb_with(&src, &out, &opt_raw).unwrap(),
            BatchSize::SmallInput,
        )
    });
    // In-memory DuckDB (":memory:") isolates engine cost from disk I/O; a fresh in-memory
    // database per iteration, discarded after.
    let mem = std::path::Path::new(":memory:");
    g.bench_function("duckdb_mem_default", |b| {
        b.iter(|| stream_ocel_file_to_duckdb_with(&src, mem, &opt_default).unwrap())
    });
    g.bench_function("duckdb_mem_raw", |b| {
        b.iter(|| stream_ocel_file_to_duckdb_with(&src, mem, &opt_raw).unwrap())
    });
    g.finish();
    let _ = std::fs::remove_file(&out);
}

fn bench_import(c: &mut Criterion) {
    bench_format(c, "json");
    bench_format(c, "xml");
}

criterion_group!(benches, bench_import);
criterion_main!(benches);
