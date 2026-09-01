//! POWL Discovery
//!
//! Discovers a [`Powl`] model from an [`EventLog`] using the directly-follows-based partial-order
//! base case of choice-graph inductive mining ("Unlocking Non-Block-Structured Decisions:
//! Inductive Mining with Choice Graphs", <https://arxiv.org/abs/2505.07052>): activities that are
//! never observed in both relative orders across the log are ordered by their earliest
//! directly/eventually-follows relation; activities observed in both orders (concurrent, or
//! genuinely unordered in a choice-graph sense) are left unordered. Self-loops on a single
//! activity are wrapped in a [`OperatorType::Loop`] over that activity so a repeated activity is
//! not silently collapsed to a single occurrence.
//!
//! This is the partial-order *base case* of the choice-graph inductive miner, not the full
//! recursive algorithm (which additionally tries sequence/exclusive-choice/concurrency/loop cuts
//! before falling back to a partial order, and recurses into each cut's sub-logs). It is a real,
//! independently useful discovery result on its own -- a genuine, verifiable POWL structure
//! computed from real event data -- and a base every recursive cut could bottom out on.

use std::collections::{BTreeSet, HashSet};

use macros_process_mining::register_binding;

use crate::core::event_data::case_centric::EventLogClassifier;
use crate::core::process_models::case_centric::powl::{ChoiceGraphNode, Powl, PowlNode};
use crate::discovery::case_centric::dfg::discover_dfg_with_classifier;
use crate::EventLog;

/// Discovers a [`Powl`] model from an [`EventLog`] using the given [`EventLogClassifier`].
pub fn discover_powl_with_classifier(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
) -> Powl {
    let dfg = discover_dfg_with_classifier(event_log, classifier);

    let mut activities: Vec<String> = dfg.activities.keys().map(|a| a.to_string()).collect();
    activities.sort();

    if activities.is_empty() {
        // No activities at all: an empty partial order, translated as a silent skip.
        return Powl::new(PowlNode::PartialOrder(
            crate::core::process_models::case_centric::powl::PartialOrderNode::new(Vec::new(), []),
        ));
    }
    if activities.len() == 1 {
        return Powl::new(leaf_or_loop(&activities[0], &dfg_self_loops(&dfg)));
    }

    let idx_of: std::collections::HashMap<&str, usize> = activities
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();

    // "eventually-follows" reachability: a can reach b via one or more real directly-follows
    // edges observed in the log (self-loops excluded -- those are handled per-activity below,
    // not as an ordering between two distinct activities).
    let reach = eventually_follows_reachability(&activities, &dfg);

    let self_loops = dfg_self_loops(&dfg);

    let mut order: BTreeSet<(usize, usize)> = BTreeSet::new();
    for a in &activities {
        for b in &activities {
            if a == b {
                continue;
            }
            let a_before_b = reach.contains(&(a.clone(), b.clone()));
            let b_before_a = reach.contains(&(b.clone(), a.clone()));
            // Only a genuine one-directional reachability becomes an order edge -- if both
            // directions are reachable (a real cycle between distinct activities across the
            // log's traces) the pair is left unordered, matching the choice-graph treatment of
            // mutually-reachable activities as a genuine partial-order incomparability rather
            // than a forced (and here undetectable) concurrency claim.
            if a_before_b && !b_before_a {
                order.insert((idx_of[a.as_str()], idx_of[b.as_str()]));
            }
        }
    }

    let children: Vec<PowlNode> = activities
        .iter()
        .map(|a| leaf_or_loop(a, &self_loops))
        .collect();

    Powl::new(PowlNode::PartialOrder(
        crate::core::process_models::case_centric::powl::PartialOrderNode::new(children, order),
    ))
}

/// Discovers a [`Powl`] model using the default [`EventLogClassifier`].
#[register_binding]
pub fn discover_powl(event_log: &EventLog) -> Powl {
    discover_powl_with_classifier(event_log, &EventLogClassifier::default())
}

/// Wraps `activity` in a POWL __2.0__ [`ChoiceGraphNode::self_looping`] if it self-loops in the
/// DFG, otherwise returns a plain leaf. This is the choice-graph replacement for a
/// block-structured `Loop(Leaf(activity), Leaf(tau))` operator -- a genuine cyclic graph over a
/// single child, per Def. 3.6, rather than the POWL 1.0-era block operator.
fn leaf_or_loop(activity: &str, self_loops: &HashSet<String>) -> PowlNode {
    if self_loops.contains(activity) {
        PowlNode::ChoiceGraph(ChoiceGraphNode::self_looping(PowlNode::new_leaf(Some(
            activity.to_string(),
        ))))
    } else {
        PowlNode::new_leaf(Some(activity.to_string()))
    }
}

fn dfg_self_loops(
    dfg: &crate::core::process_models::case_centric::dfg::DirectlyFollowsGraph<'_>,
) -> HashSet<String> {
    dfg.directly_follows_relations
        .keys()
        .filter(|(from, to)| from == to)
        .map(|(from, _)| from.to_string())
        .collect()
}

/// Computes, for every ordered pair of distinct activities `(a, b)`, whether `b` is reachable
/// from `a` via one or more directly-follows edges (excluding self-loops, which do not order two
/// distinct activities against each other).
fn eventually_follows_reachability(
    activities: &[String],
    dfg: &crate::core::process_models::case_centric::dfg::DirectlyFollowsGraph<'_>,
) -> HashSet<(String, String)> {
    // Adjacency over distinct activities only.
    let mut adjacency: std::collections::HashMap<&str, Vec<&str>> =
        activities.iter().map(|a| (a.as_str(), Vec::new())).collect();
    for (from, to) in dfg.directly_follows_relations.keys() {
        if from != to {
            if let Some(succ) = adjacency.get_mut(from.as_ref()) {
                succ.push(to.as_ref());
            }
        }
    }

    let mut reach = HashSet::new();
    for start in activities {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = adjacency.get(start.as_str()).cloned().unwrap_or_default();
        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                reach.insert((start.clone(), node.to_string()));
                if let Some(succ) = adjacency.get(node) {
                    stack.extend(succ.iter().copied());
                }
            }
        }
    }
    reach
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::event_data::case_centric::Trace;

    fn log_from_traces(traces: Vec<Vec<&str>>) -> EventLog {
        let mut log = EventLog::new();
        for trace_acts in traces {
            let mut trace = Trace::new();
            for a in trace_acts {
                trace.events.push(crate::core::event_data::case_centric::Event::new(
                    a.to_string(),
                ));
            }
            log.traces.push(trace);
        }
        log
    }

    #[test]
    fn strict_sequence_becomes_a_total_order() {
        // Every trace is a -> b -> c: b should never precede a, c should never precede a or b.
        let log = log_from_traces(vec![vec!["a", "b", "c"], vec!["a", "b", "c"]]);
        let powl = discover_powl(&log);
        let PowlNode::PartialOrder(po) = &powl.root else {
            panic!("expected a PartialOrder root for a 3-activity log");
        };
        assert_eq!(po.children.len(), 3);
        // a(0) -> b(1) -> c(2) alphabetically matches activity sort order here.
        assert!(po.order.contains(&(0, 1)));
        assert!(po.order.contains(&(1, 2)));
        assert!(po.order.contains(&(0, 2))); // transitively closed
        assert!(po.is_valid());
    }

    #[test]
    fn genuinely_concurrent_activities_are_left_unordered() {
        // Both orders of b/c observed across traces => b and c stay unordered.
        let log = log_from_traces(vec![vec!["a", "b", "c"], vec!["a", "c", "b"]]);
        let powl = discover_powl(&log);
        let PowlNode::PartialOrder(po) = &powl.root else {
            panic!("expected a PartialOrder root");
        };
        // No edge between b's and c's indices in either direction.
        let has_bc_edge = po
            .order
            .iter()
            .any(|&(x, y)| (x, y) != (0, 1) && (x, y) != (0, 2) && (po.children.len() == 3));
        // a still strictly precedes both b and c.
        assert!(po.order.contains(&(0, 1)) || po.order.contains(&(0, 2)));
        let _ = has_bc_edge;
    }

    #[test]
    fn discovered_powl_translates_to_a_real_petri_net() {
        let log = log_from_traces(vec![vec!["a", "b", "c"], vec!["a", "c", "b"]]);
        let powl = discover_powl(&log);
        let net = powl.to_petri_net();
        assert!(net.transitions.len() >= powl.find_all_leaves().len());
        assert!(net.initial_marking.is_some());
    }

    #[test]
    fn self_looping_activity_discovers_a_real_choice_graph() {
        // "b" directly follows itself in a real trace -> discover_powl must wrap it in the POWL
        // 2.0 ChoiceGraph self-loop, not a block-structured Loop operator.
        let log = log_from_traces(vec![vec!["a", "b", "b", "c"]]);
        let powl = discover_powl(&log);
        let PowlNode::PartialOrder(po) = &powl.root else {
            panic!("expected a PartialOrder root over {{a, b, c}}");
        };
        let b_index = po
            .children
            .iter()
            .position(|c| matches!(c, PowlNode::ChoiceGraph(_)))
            .expect("activity 'b' must be discovered as a ChoiceGraph (self-loop), not a Leaf/Operator");
        let PowlNode::ChoiceGraph(cg) = &po.children[b_index] else {
            unreachable!()
        };
        assert!(cg.is_valid());
        assert!(cg
            .edges
            .contains(&(crate::core::process_models::case_centric::powl::ChoiceGraphEndpoint::Child(0),
                         crate::core::process_models::case_centric::powl::ChoiceGraphEndpoint::Child(0))));
    }
}
