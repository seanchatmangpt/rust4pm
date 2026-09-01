//! Real, runnable end-to-end demo: constructs an `EventLog` in-memory, discovers a POWL model
//! from it via `discover_powl`, and prints the resulting partial order plus its translated
//! Petri net's transition/place counts. Run with:
//!
//!   cargo run --example powl_demo -p process_mining
use process_mining::core::event_data::case_centric::{Event, Trace};
use process_mining::discovery::case_centric::powl::discover_powl;
use process_mining::EventLog;

fn main() {
    let mut log = EventLog::new();

    // Two traces: a always precedes c; b appears in both relative orders around... actually
    // interleaved with c, so (b, c) stays genuinely unordered while a stays first.
    for trace_activities in [vec!["a", "b", "c"], vec!["a", "c", "b"]] {
        let mut trace = Trace::new();
        for activity in trace_activities {
            trace.events.push(Event::new(activity.to_string()));
        }
        log.traces.push(trace);
    }

    let powl = discover_powl(&log);
    let json = serde_json::to_string_pretty(&powl).expect("powl model must serialize");
    println!("Discovered POWL model:\n{json}");

    let net = powl.to_petri_net();
    println!(
        "Translated Petri net: {} places, {} transitions, {} arcs",
        net.places.len(),
        net.transitions.len(),
        net.arcs.len()
    );
}
