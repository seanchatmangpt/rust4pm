//! Real, runnable end-to-end demo: constructs an `EventLog` in-memory, discovers a POWL model
//! from it via `discover_powl`, and prints the resulting partial order plus its translated
//! Petri net's transition/place counts. Run with:
//!
//!   cargo run --example powl_demo -p process_mining
use process_mining::core::event_data::case_centric::{Event, Trace};
use process_mining::discovery::case_centric::powl::discover_powl;
use process_mining::EventLog;

fn discover_and_print(label: &str, traces: Vec<Vec<&str>>) {
    let mut log = EventLog::new();
    for trace_activities in traces {
        let mut trace = Trace::new();
        for activity in trace_activities {
            trace.events.push(Event::new(activity.to_string()));
        }
        log.traces.push(trace);
    }

    let powl = discover_powl(&log);
    let json = serde_json::to_string_pretty(&powl).expect("powl model must serialize");
    println!("=== {label} ===\nDiscovered POWL model:\n{json}");

    let net = powl.to_petri_net();
    println!(
        "Translated Petri net: {} places, {} transitions, {} arcs\n",
        net.places.len(),
        net.transitions.len(),
        net.arcs.len()
    );
}

fn main() {
    // Two traces: a always precedes c; b appears interleaved with c both ways, so (b, c) stays
    // genuinely unordered while a stays first -- exercises PartialOrder (POWL 1.0's construct).
    discover_and_print("PartialOrder (a before b, c; b/c unordered)", vec![
        vec!["a", "b", "c"],
        vec!["a", "c", "b"],
    ]);

    // "b" directly follows itself: discover_powl wraps it in a POWL 2.0 ChoiceGraph self-loop
    // (Def. 3.6 -- a genuine cyclic graph over one child), not a block-structured Loop operator.
    // This is the construct that makes this fork POWL __2.0__, not POWL 1.0.
    discover_and_print("ChoiceGraph (b self-loops)", vec![vec!["a", "b", "b", "c"]]);
}
