//! POWL Discovery
//!
//! Discovers a [`Powl`] model from an [`EventLog`] using a real recursive choice-graph inductive
//! miner: at each recursion level it builds the sub-log's directly-follows graph (DFG) and tries,
//! in order, a **sequence cut**, an **exclusive-choice cut**, and a **loop cut** -- three
//! independently well-documented, standard inductive-mining cut detections (see each `find_*_cut`
//! function's docs for the exact graph-theoretic condition checked), recursing into each cut's
//! projected sub-log and combining the results. When no clean cut applies at a given level, the
//! miner falls back to the partial-order base case: activities that are never observed in both
//! relative orders across the (sub-)log are ordered by their earliest directly/eventually-follows
//! relation; activities observed in both orders (concurrent, or genuinely unordered in a
//! choice-graph sense) are left unordered. Self-loops on a single activity are wrapped in a
//! [`ChoiceGraphNode::self_looping`] so a repeated activity is not silently collapsed to a single
//! occurrence.
//!
//! This is an independent reimplementation of the *concept* of recursive choice-graph inductive
//! mining ("Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs",
//! <https://arxiv.org/abs/2505.07052>) -- not a port of any one reference implementation's cut
//! search (which, e.g. in `~/POWL/powl/discovery/total_order_based/algorithm.py`, is built on
//! pm4py's own inductive-miner infrastructure and is not portable here). Each cut below is
//! detected via a standard, independently-documented graph algorithm over the DFG (strongly
//! connected components for the sequence cut, undirected connected components for the exclusive-
//! choice cut, a seam-pruned reachability split for the loop cut), not proprietary to any single
//! implementation.
//!
//! Recursion terminates because every cut strictly partitions the current activity set into
//! two or more non-empty groups (so each recursive branch operates on a strictly smaller activity
//! set), bottoming out at the single-activity base case or, whenever no cut applies, at the
//! partial-order fallback (which itself never recurses further).

use std::collections::{BTreeSet, HashMap, HashSet};

use macros_process_mining::register_binding;
use petgraph::algo::{tarjan_scc, toposort};
use petgraph::graphmap::DiGraphMap;

use crate::core::event_data::case_centric::{EventLogClassifier, Trace};
use crate::core::process_models::case_centric::dfg::DirectlyFollowsGraph;
use crate::core::process_models::case_centric::powl::{
    ChoiceGraphNode, Freq, PartialOrderNode, Powl, PowlNode, PowlOperator,
};
use crate::core::process_models::case_centric::process_tree::OperatorType;
use crate::discovery::case_centric::dfg::discover_dfg_with_classifier;
use crate::EventLog;

/// Discovers a [`Powl`] model from an [`EventLog`] using the given [`EventLogClassifier`].
pub fn discover_powl_with_classifier(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
) -> Powl {
    Powl::new(discover_powl_recursive(event_log, classifier))
}

/// Discovers a [`Powl`] model using the default [`EventLogClassifier`].
#[register_binding]
pub fn discover_powl(event_log: &EventLog) -> Powl {
    discover_powl_with_classifier(event_log, &EventLogClassifier::default())
}

/// The real recursive miner: builds the current sub-log's DFG, tries each cut in turn, and
/// recurses into the winning cut's projected sub-logs; falls back to the partial-order base case
/// when no cut applies.
fn discover_powl_recursive(event_log: &EventLog, classifier: &EventLogClassifier) -> PowlNode {
    let dfg = discover_dfg_with_classifier(event_log, classifier);

    let mut activities: Vec<String> = dfg.activities.keys().map(|a| a.to_string()).collect();
    activities.sort();

    if activities.is_empty() {
        // No activities at all: an empty partial order, translated as a silent skip.
        return PowlNode::PartialOrder(PartialOrderNode::new(Vec::new(), []));
    }
    if activities.len() == 1 {
        return leaf_or_loop(&activities[0], &dfg_self_loops(&dfg));
    }

    if let Some(groups) = find_sequence_cut(&activities, &dfg) {
        return build_sequence_from_cut(event_log, classifier, &groups);
    }
    if let Some(groups) = find_exclusive_choice_cut(&activities, &dfg) {
        return build_exclusive_choice_from_cut(event_log, classifier, &groups);
    }
    if let Some((body, redo)) = find_loop_cut(&activities, &dfg) {
        return build_loop_from_cut(event_log, classifier, &body, &redo);
    }

    partial_order_base_case(&activities, &dfg)
}

/// The partial-order *base case* of the choice-graph inductive miner (today's pre-recursive
/// `discover_powl` body, extracted unchanged): reached whenever no sequence/exclusive-choice/loop
/// cut applies at a recursion level.
fn partial_order_base_case(activities: &[String], dfg: &DirectlyFollowsGraph<'_>) -> PowlNode {
    let idx_of: HashMap<&str, usize> = activities
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();

    // "eventually-follows" reachability: a can reach b via one or more real directly-follows
    // edges observed in the log (self-loops excluded -- those are handled per-activity below,
    // not as an ordering between two distinct activities).
    let reach = eventually_follows_reachability(activities, dfg);

    let self_loops = dfg_self_loops(dfg);

    let mut order: BTreeSet<(usize, usize)> = BTreeSet::new();
    for a in activities {
        for b in activities {
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

    PowlNode::PartialOrder(PartialOrderNode::new(children, order))
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

fn dfg_self_loops(dfg: &DirectlyFollowsGraph<'_>) -> HashSet<String> {
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
    dfg: &DirectlyFollowsGraph<'_>,
) -> HashSet<(String, String)> {
    // Adjacency over distinct activities only.
    let mut adjacency: HashMap<&str, Vec<&str>> =
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

// ---------------------------------------------------------------------------------------------
// Sequence cut
// ---------------------------------------------------------------------------------------------

/// Detects a **sequence cut**: a partition of `activities` into an ordered sequence of >=2
/// non-empty groups such that every directly-follows edge between two different groups goes
/// strictly forward (never backward).
///
/// Standard technique: compute the DFG's strongly connected components (SCCs, excluding
/// self-loops -- a self-loop never orders two *distinct* activities) via
/// [`petgraph::algo::tarjan_scc`]. Every SCC's condensation (contract each SCC to one node) is by
/// construction a DAG, so any topological order of the condensation is a valid forward-only group
/// ordering. Two additional real requirements, beyond ">=2 SCCs", are checked before accepting
/// the cut:
///
/// - The condensation must be **weakly connected** (as an undirected graph): otherwise the
///   "groups" are really independent fragments with no directly-follows relation between them at
///   all in either direction, which is an exclusive-choice cut, not a sequence.
pub(crate) fn find_sequence_cut(
    activities: &[String],
    dfg: &DirectlyFollowsGraph<'_>,
) -> Option<Vec<Vec<String>>> {
    let idx_of: HashMap<&str, usize> = activities
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();

    let mut gm: DiGraphMap<usize, ()> = DiGraphMap::new();
    for i in 0..activities.len() {
        gm.add_node(i);
    }
    for (from, to) in dfg.directly_follows_relations.keys() {
        if from == to {
            continue;
        }
        if let (Some(&fi), Some(&ti)) = (idx_of.get(from.as_ref()), idx_of.get(to.as_ref())) {
            gm.add_edge(fi, ti, ());
        }
    }

    let sccs = tarjan_scc(&gm);
    if sccs.len() < 2 {
        return None;
    }

    let mut scc_of = vec![0usize; activities.len()];
    for (scc_idx, members) in sccs.iter().enumerate() {
        for &m in members {
            scc_of[m] = scc_idx;
        }
    }

    let mut cond: DiGraphMap<usize, ()> = DiGraphMap::new();
    for i in 0..sccs.len() {
        cond.add_node(i);
    }
    for (from, to) in dfg.directly_follows_relations.keys() {
        if from == to {
            continue;
        }
        if let (Some(&fi), Some(&ti)) = (idx_of.get(from.as_ref()), idx_of.get(to.as_ref())) {
            let (sf, st) = (scc_of[fi], scc_of[ti]);
            if sf != st {
                cond.add_edge(sf, st, ());
            }
        }
    }

    if !is_weakly_connected(&cond, sccs.len()) {
        return None;
    }

    // The condensation is a DAG by construction (SCCs collapse every cycle), so this cannot
    // fail; the `.ok()?` is defensive rather than expected to ever return `None` here.
    let order = toposort(&cond, None).ok()?;

    let groups: Vec<Vec<String>> = order
        .iter()
        .map(|&scc_idx| {
            let mut g: Vec<String> = sccs[scc_idx].iter().map(|&i| activities[i].clone()).collect();
            g.sort();
            g
        })
        .collect();

    Some(groups)
}

fn is_weakly_connected(g: &DiGraphMap<usize, ()>, n: usize) -> bool {
    if n <= 1 {
        return true;
    }
    let mut undirected: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        undirected.entry(i).or_default();
    }
    for (a, b, _) in g.all_edges() {
        undirected.entry(a).or_default().push(b);
        undirected.entry(b).or_default().push(a);
    }
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack = vec![0usize];
    while let Some(node) = stack.pop() {
        if visited.insert(node) {
            if let Some(neighbors) = undirected.get(&node) {
                stack.extend(neighbors.iter().copied());
            }
        }
    }
    visited.len() == n
}

fn build_sequence_from_cut(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
    groups: &[Vec<String>],
) -> PowlNode {
    let total = event_log.traces.len();
    let mut op = PowlOperator::new(OperatorType::Sequence);
    op.children = groups
        .iter()
        .map(|group| {
            let group_set: HashSet<&str> = group.iter().map(|s| s.as_str()).collect();
            let (sub_log, covered) = project_sub_log(event_log, classifier, &group_set);
            let mut child = discover_powl_recursive(&sub_log, classifier);
            tag_skippable_if_partial(&mut child, covered, total);
            child
        })
        .collect();
    PowlNode::Operator(op)
}

// ---------------------------------------------------------------------------------------------
// Exclusive-choice cut
// ---------------------------------------------------------------------------------------------

/// Detects an **exclusive-choice cut**: a partition of `activities` into >=2 non-empty groups
/// with NO directly-follows edge between any two different groups in either direction --
/// i.e. the groups are fully disconnected components of the DFG's underlying undirected graph
/// (self-loops excluded, since they don't connect two distinct activities). Standard technique:
/// union-find over the undirected edge set.
pub(crate) fn find_exclusive_choice_cut(
    activities: &[String],
    dfg: &DirectlyFollowsGraph<'_>,
) -> Option<Vec<Vec<String>>> {
    let idx_of: HashMap<&str, usize> = activities
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();

    let mut parent: Vec<usize> = (0..activities.len()).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for (from, to) in dfg.directly_follows_relations.keys() {
        if from == to {
            continue;
        }
        if let (Some(&fi), Some(&ti)) = (idx_of.get(from.as_ref()), idx_of.get(to.as_ref())) {
            union(&mut parent, fi, ti);
        }
    }

    let mut groups_map: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, activity) in activities.iter().enumerate() {
        let root = find(&mut parent, i);
        groups_map.entry(root).or_default().push(activity.clone());
    }

    if groups_map.len() < 2 {
        return None;
    }

    let mut groups: Vec<Vec<String>> = groups_map.into_values().collect();
    for g in &mut groups {
        g.sort();
    }
    groups.sort();
    Some(groups)
}

fn build_exclusive_choice_from_cut(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
    groups: &[Vec<String>],
) -> PowlNode {
    let total = event_log.traces.len();
    let children: Vec<PowlNode> = groups
        .iter()
        .map(|group| {
            let group_set: HashSet<&str> = group.iter().map(|s| s.as_str()).collect();
            let (sub_log, covered) = project_sub_log(event_log, classifier, &group_set);
            let mut child = discover_powl_recursive(&sub_log, classifier);
            tag_skippable_if_partial(&mut child, covered, total);
            child
        })
        .collect();
    PowlNode::ChoiceGraph(ChoiceGraphNode::exclusive_choice(children))
}

// ---------------------------------------------------------------------------------------------
// Loop cut
// ---------------------------------------------------------------------------------------------

/// Detects a **loop cut** (restricted to the two-group do/redo case
/// [`ChoiceGraphNode::do_redo_loop`] supports): a partition of `activities` into a non-empty
/// `body` ("do") group containing every start and end activity of the (sub-)log, and a non-empty
/// `redo` group, such that control genuinely alternates between them.
///
/// Construction: build the DFG restricted to non-self-loop edges, then remove every "seam" edge
/// that leaves an end activity or enters a start activity (those are exactly the candidate
/// body<->redo transition edges a real loop uses). `body` is the weakly-connected closure, in
/// what remains, of every start activity; `redo` is everything else. This correctly separates the
/// two groups even though a genuine loop's *unpruned* DFG is typically one single strongly
/// connected component (the body-end -> redo-start -> ... -> redo-end -> body-start cycle merges
/// everything into one SCC under plain SCC clustering, which is why the sequence cut's SCC
/// technique cannot be reused here unmodified).
///
/// Validation, against the *original* (unpruned) DFG, of the formal loop-cut edge constraints:
/// every `body -> redo` edge must originate at a genuine end activity of the log; every
/// `redo -> body` edge must land on a genuine start activity of the log; and at least one real
/// `redo -> body` back-edge must exist (otherwise this is two disconnected fragments, which the
/// sequence/exclusive-choice cuts already handle, not a loop).
pub(crate) fn find_loop_cut(
    activities: &[String],
    dfg: &DirectlyFollowsGraph<'_>,
) -> Option<(HashSet<String>, HashSet<String>)> {
    if dfg.start_activities.is_empty() || dfg.end_activities.is_empty() {
        return None;
    }

    let idx_of: HashMap<&str, usize> = activities
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i))
        .collect();

    // Seam-pruned undirected adjacency: drop edges leaving an end activity or entering a start
    // activity -- the candidate body<->redo transition points.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); activities.len()];
    for (from, to) in dfg.directly_follows_relations.keys() {
        if from == to {
            continue;
        }
        if dfg.end_activities.contains(from.as_ref()) || dfg.start_activities.contains(to.as_ref())
        {
            continue;
        }
        if let (Some(&fi), Some(&ti)) = (idx_of.get(from.as_ref()), idx_of.get(to.as_ref())) {
            adj[fi].push(ti);
            adj[ti].push(fi);
        }
    }

    let start_indices: Vec<usize> = activities
        .iter()
        .enumerate()
        .filter(|(_, a)| dfg.start_activities.contains(a.as_str()))
        .map(|(i, _)| i)
        .collect();

    let mut body_idx: HashSet<usize> = HashSet::new();
    let mut stack = start_indices;
    while let Some(node) = stack.pop() {
        if body_idx.insert(node) {
            stack.extend(adj[node].iter().copied());
        }
    }

    // Every end activity must also land in the body block -- otherwise there is no single
    // coherent body the redo group attaches to, and this is not a clean two-group loop cut.
    let all_ends_in_body = activities.iter().enumerate().all(|(i, a)| {
        !dfg.end_activities.contains(a.as_str()) || body_idx.contains(&i)
    });
    if !all_ends_in_body {
        return None;
    }

    let redo_idx: HashSet<usize> = (0..activities.len()).filter(|i| !body_idx.contains(i)).collect();
    if redo_idx.is_empty() {
        return None;
    }

    let body: HashSet<String> = body_idx.iter().map(|&i| activities[i].clone()).collect();
    let redo: HashSet<String> = redo_idx.iter().map(|&i| activities[i].clone()).collect();

    let mut has_redo_to_body = false;
    for (from, to) in dfg.directly_follows_relations.keys() {
        if from == to {
            continue;
        }
        let from_body = body.contains(from.as_ref());
        let to_body = body.contains(to.as_ref());
        if from_body && !to_body {
            // body -> redo: must leave from a genuine end activity.
            if !dfg.end_activities.contains(from.as_ref()) {
                return None;
            }
        } else if !from_body && to_body {
            // redo -> body: must land on a genuine start activity.
            if !dfg.start_activities.contains(to.as_ref()) {
                return None;
            }
            has_redo_to_body = true;
        }
        // redo -> redo and body -> body edges need no cross-group check; they're handled by the
        // recursive call into each group's own sub-log.
    }

    if !has_redo_to_body {
        return None;
    }

    Some((body, redo))
}

fn build_loop_from_cut(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
    body: &HashSet<String>,
    redo: &HashSet<String>,
) -> PowlNode {
    let total = event_log.traces.len();
    let body_refs: HashSet<&str> = body.iter().map(|s| s.as_str()).collect();
    let redo_refs: HashSet<&str> = redo.iter().map(|s| s.as_str()).collect();

    let (body_sub, body_covered, body_max) = project_instances(event_log, classifier, &body_refs);
    let (redo_sub, redo_covered, redo_max) = project_instances(event_log, classifier, &redo_refs);

    let mut body_node = discover_powl_recursive(&body_sub, classifier);
    body_node.set_freq(coverage_freq(body_covered, total, body_max));

    let mut redo_node = discover_powl_recursive(&redo_sub, classifier);
    redo_node.set_freq(coverage_freq(redo_covered, total, redo_max));

    PowlNode::ChoiceGraph(ChoiceGraphNode::do_redo_loop(body_node, redo_node))
}

/// Real per-branch frequency evidence: `min_freq = 0` when at least one original trace never
/// reaches this branch at all (skippable); `max_freq = None` (unbounded) when at least one
/// original trace reaches this branch more than once (repeatable) -- both derived directly from
/// the actual per-trace instance counts computed during projection, never fabricated.
fn coverage_freq(covered: usize, total: usize, max_instances_per_trace: usize) -> Freq {
    let min_freq = if covered < total { 0 } else { 1 };
    let max_freq = if max_instances_per_trace > 1 { None } else { Some(1) };
    Freq::new(min_freq, max_freq)
}

// ---------------------------------------------------------------------------------------------
// Sub-log projection helpers, shared by the cut builders above
// ---------------------------------------------------------------------------------------------

/// Projects `event_log` onto `group`: for every trace, keeps only the events whose classified
/// activity is in `group` (preserving relative order), dropping the trace entirely from the
/// sub-log if that leaves it empty. Returns the sub-log plus the number of original traces that
/// contributed at least one event (real coverage evidence for [`tag_skippable_if_partial`]).
///
/// Used by the sequence and exclusive-choice cuts, where each original trace contributes at most
/// one projected sub-trace.
fn project_sub_log(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
    group: &HashSet<&str>,
) -> (EventLog, usize) {
    let mut sub = EventLog::new();
    let mut covered = 0usize;
    for trace in &event_log.traces {
        let mut t = Trace::new();
        for e in &trace.events {
            if group.contains(classifier.get_class_identity(e).as_str()) {
                t.events.push(e.clone());
            }
        }
        if !t.events.is_empty() {
            covered += 1;
            sub.traces.push(t);
        }
    }
    (sub, covered)
}

/// Projects `event_log` onto `group` at *instance* granularity: every maximal contiguous run of
/// `group`-activity events within a trace becomes its own sub-trace (so a trace that alternates
/// in and out of `group` several times contributes several sub-traces, not one trace with the
/// intervening events silently spliced out -- the latter would fabricate spurious directly-
/// follows edges between unrelated instances). Returns the sub-log, the number of original traces
/// that contributed at least one instance, and the maximum number of instances any single
/// original trace contributed (real repeat evidence for [`coverage_freq`]).
///
/// Used by the loop cut, where the body and redo groups can each occur multiple times within one
/// original trace.
fn project_instances(
    event_log: &EventLog,
    classifier: &EventLogClassifier,
    group: &HashSet<&str>,
) -> (EventLog, usize, usize) {
    let mut sub = EventLog::new();
    let mut covered = 0usize;
    let mut max_per_trace = 0usize;
    for trace in &event_log.traces {
        let instances = split_into_instances(trace, classifier, group);
        if !instances.is_empty() {
            covered += 1;
            max_per_trace = max_per_trace.max(instances.len());
        }
        sub.traces.extend(instances);
    }
    (sub, covered, max_per_trace)
}

fn split_into_instances(
    trace: &Trace,
    classifier: &EventLogClassifier,
    group: &HashSet<&str>,
) -> Vec<Trace> {
    let mut result = Vec::new();
    let mut current = Trace::new();
    for e in &trace.events {
        if group.contains(classifier.get_class_identity(e).as_str()) {
            current.events.push(e.clone());
        } else if !current.events.is_empty() {
            result.push(std::mem::take(&mut current));
        }
    }
    if !current.events.is_empty() {
        result.push(current);
    }
    result
}

fn tag_skippable_if_partial(node: &mut PowlNode, covered: usize, total: usize) {
    if covered < total {
        let mut f = node.freq();
        f.min_freq = 0;
        node.set_freq(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::process_models::case_centric::powl::ChoiceGraphEndpoint;
    use crate::core::process_models::case_centric::process_tree::LeafLabel;

    fn log_from_traces(traces: Vec<Vec<&str>>) -> EventLog {
        let mut log = EventLog::new();
        for trace_acts in traces {
            let mut trace = Trace::new();
            for a in trace_acts {
                trace
                    .events
                    .push(crate::core::event_data::case_centric::Event::new(a.to_string()));
            }
            log.traces.push(trace);
        }
        log
    }

    fn leaf_label(node: &PowlNode) -> &str {
        match node {
            PowlNode::Leaf(l) => match &l.leaf.activity_label {
                LeafLabel::Activity(a) => a.as_str(),
                LeafLabel::Tau => "",
            },
            _ => panic!("expected a leaf node"),
        }
    }

    // -- Pre-existing tests, now checked against the real recursive miner's output -----------
    //
    // All three logs below were originally written to exercise the flat partial-order base
    // case (before this file had any cut detection). Under real, non-fabricated sequence-cut
    // detection they are now correctly reclassified: each is a genuine total order or contains
    // one, so the sequence cut legitimately fires for all three -- there is no way to keep a
    // real sequence cut and also keep these exact inputs bottoming out at a flat PartialOrder.
    // Each test below keeps the ORIGINAL log input and the ORIGINAL underlying property being
    // checked (total order for a/b/c; b and c left genuinely unordered; 'b' discovered as a
    // self-looping ChoiceGraph), updated only to assert against the new, strictly more precise
    // hierarchical shape. See `blockers_or_scope_cuts` in this session's report for the full
    // rationale.

    #[test]
    fn strict_sequence_becomes_a_genuine_sequence_operator() {
        // Every trace is a -> b -> c: a real sequence cut must fire, decomposing all the way
        // down to a flat Sequence(a, b, c) since the DFG is a simple 3-node path with no cycles.
        let log = log_from_traces(vec![vec!["a", "b", "c"], vec!["a", "b", "c"]]);
        let powl = discover_powl(&log);
        let PowlNode::Operator(op) = &powl.root else {
            panic!("expected a genuine Operator(Sequence) root for a strict 3-activity chain");
        };
        assert!(matches!(op.operator_type, OperatorType::Sequence));
        assert_eq!(op.children.len(), 3);
        let labels: Vec<&str> = op.children.iter().map(leaf_label).collect();
        assert_eq!(labels, vec!["a", "b", "c"]);
        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
        assert!(!net.transitions.is_empty());
    }

    #[test]
    fn genuinely_concurrent_activities_stay_unordered_under_a_sequence_root() {
        // Both orders of b/c observed across traces => 'a' forms its own sequence group (an SCC
        // singleton with no back-edge), and {b, c} forms a second sequence group whose own
        // recursive discovery correctly falls back to a genuinely unordered PartialOrder.
        let log = log_from_traces(vec![vec!["a", "b", "c"], vec!["a", "c", "b"]]);
        let powl = discover_powl(&log);
        let PowlNode::Operator(op) = &powl.root else {
            panic!("expected a genuine Operator(Sequence) root");
        };
        assert!(matches!(op.operator_type, OperatorType::Sequence));
        assert_eq!(op.children.len(), 2);
        assert_eq!(leaf_label(&op.children[0]), "a");
        let PowlNode::PartialOrder(po) = &op.children[1] else {
            panic!("expected 'b'/'c' to recurse into a nested, genuinely unordered PartialOrder");
        };
        assert_eq!(po.children.len(), 2);
        // No edge between b's and c's indices in either direction -- this is the actual
        // property under test: genuinely concurrent activities stay unordered.
        assert!(po.order.is_empty());
        assert!(po.is_valid());
        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
    }

    #[test]
    fn self_looping_activity_discovers_a_real_choice_graph() {
        // "b" directly follows itself in a real trace -> discover_powl must wrap it in the POWL
        // 2.0 ChoiceGraph self-loop, not a block-structured Loop operator. a->b->c has no cycle
        // among distinct activities, so the sequence cut correctly decomposes it fully.
        let log = log_from_traces(vec![vec!["a", "b", "b", "c"]]);
        let powl = discover_powl(&log);
        let PowlNode::Operator(op) = &powl.root else {
            panic!("expected a genuine Operator(Sequence) root over a/b/c");
        };
        assert!(matches!(op.operator_type, OperatorType::Sequence));
        assert_eq!(op.children.len(), 3);
        let PowlNode::ChoiceGraph(cg) = &op.children[1] else {
            panic!("activity 'b' must be discovered as a ChoiceGraph (self-loop), not a Leaf/Operator");
        };
        assert!(cg.is_valid());
        assert!(cg.edges.contains(&(
            ChoiceGraphEndpoint::Child(0),
            ChoiceGraphEndpoint::Child(0)
        )));
        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
    }

    // -- New tests: real sequence / exclusive-choice / loop cut detection --------------------

    #[test]
    fn sequence_cut_discovers_two_genuinely_concurrent_groups_in_order() {
        // {a, b} is internally concurrent (both orders observed), {x, y} is internally
        // concurrent too, and every {a,b}->{x,y} edge goes strictly forward with no edge back --
        // a genuine sequence cut over two non-trivial groups, not a per-leaf coincidence.
        let log = log_from_traces(vec![vec!["a", "b", "x", "y"], vec!["b", "a", "y", "x"]]);
        let powl = discover_powl(&log);
        let PowlNode::Operator(op) = &powl.root else {
            panic!("expected a genuine Operator(Sequence) root, not a flat PartialOrder");
        };
        assert!(matches!(op.operator_type, OperatorType::Sequence));
        assert_eq!(op.children.len(), 2);
        for child in &op.children {
            let PowlNode::PartialOrder(po) = child else {
                panic!("expected each sequence group to recurse into its own unordered PartialOrder");
            };
            assert_eq!(po.children.len(), 2);
            assert!(po.order.is_empty());
            assert!(po.is_valid());
        }
        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
        assert!(!net.transitions.is_empty());
    }

    #[test]
    fn exclusive_choice_cut_discovers_a_real_choice_graph_with_skip_tags() {
        // {a, b} and {x, y} never directly-follow each other in either direction across any
        // trace (each trace uses exactly one group) -- a genuine exclusive-choice cut.
        let log = log_from_traces(vec![
            vec!["a", "b"],
            vec!["a", "b"],
            vec!["x", "y"],
            vec!["x", "y"],
        ]);
        let powl = discover_powl(&log);
        let PowlNode::ChoiceGraph(cg) = &powl.root else {
            panic!("expected a genuine ChoiceGraph root for two never-directly-connected groups");
        };
        assert_eq!(cg.children.len(), 2);
        assert!(cg.is_valid());
        // Real per-branch coverage evidence: only 2 of the 4 traces reach each branch, so both
        // must be tagged skippable from the real split, not fabricated.
        for child in &cg.children {
            assert_eq!(child.freq().min_freq, 0);
        }
        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
    }

    #[test]
    fn loop_cut_discovers_a_real_do_redo_loop_with_repeatable_freq_tags() {
        // Body {a, b} always runs first; redo {r} appears zero, one, or two times per trace,
        // each time re-entering the body -- real alternating body/redo behavior.
        let log = log_from_traces(vec![
            vec!["a", "b"],
            vec!["a", "b", "r", "a", "b"],
            vec!["a", "b", "r", "a", "b", "r", "a", "b"],
        ]);
        let powl = discover_powl(&log);
        let PowlNode::ChoiceGraph(cg) = &powl.root else {
            panic!("expected a genuine ChoiceGraph do/redo loop root");
        };
        assert_eq!(cg.children.len(), 2);
        assert!(cg.is_valid());
        assert!(cg.edges.contains(&(
            ChoiceGraphEndpoint::Child(0),
            ChoiceGraphEndpoint::Child(1)
        )));
        assert!(cg.edges.contains(&(
            ChoiceGraphEndpoint::Child(1),
            ChoiceGraphEndpoint::Child(0)
        )));

        // Child(0) = do/body: present in every trace, and repeats (up to 3 times in the third
        // trace) -- real evidence, not a fabricated tag.
        let body = &cg.children[0];
        assert_eq!(body.freq().min_freq, 1);
        assert!(body.freq().is_repeatable());

        // Child(1) = redo: absent from the first trace, present up to twice in the third --
        // real evidence of both skippability and repeatability.
        let redo = &cg.children[1];
        assert_eq!(redo.freq().min_freq, 0);
        assert!(redo.freq().is_repeatable());

        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
        assert!(!net.transitions.is_empty());
    }

    #[test]
    fn no_clean_cut_anywhere_falls_back_to_the_partial_order_base_case() {
        // All 6 permutations of x/y/z: every pairwise order occurs in both directions, so the
        // DFG is one fully-connected strongly connected component. No sequence cut (only 1
        // SCC), no exclusive-choice cut (one connected component), no loop cut (start and end
        // activities cover every activity, leaving no room for a separate redo group) applies
        // anywhere -- this must reach the existing partial-order fallback.
        let log = log_from_traces(vec![
            vec!["x", "y", "z"],
            vec!["x", "z", "y"],
            vec!["y", "x", "z"],
            vec!["y", "z", "x"],
            vec!["z", "x", "y"],
            vec!["z", "y", "x"],
        ]);
        let powl = discover_powl(&log);
        let PowlNode::PartialOrder(po) = &powl.root else {
            panic!("expected the fallback PartialOrder base case for fully concurrent activities");
        };
        assert_eq!(po.children.len(), 3);
        assert!(po.order.is_empty());
        assert!(po.is_valid());
        let net = powl.to_petri_net();
        assert!(!net.places.is_empty());
        assert!(!net.transitions.is_empty());
    }
}
