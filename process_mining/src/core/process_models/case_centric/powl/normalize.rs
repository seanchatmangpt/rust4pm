//! [`Powl::normalize`]: structural reduction of a POWL model to a smaller/simpler equivalent.
//!
//! An independent reimplementation of the reference's `TaggedPOWL.normalize()` family
//! (`~/POWL/powl/objects/tagged_powl/choice_graph.py`'s `_reduce_silent_activities` /
//! `_reduce_simple_silent_transitions` / `_mark_skippable_nodes` / `_abstract_self_loop` /
//! `_abstract_sccs` / `_abstract_sequences` / `_flatten_single_node`, and
//! `~/POWL/powl/objects/tagged_powl/partial_order.py`'s `_reduce_silent_activities` /
//! `flatten`), read in full and reimplemented over this fork's own [`PowlNode`] /
//! [`ChoiceGraphNode`] / [`PartialOrderNode`] types rather than transliterated -- every helper
//! below is a fresh reimplementation over this crate's own index/`BTreeSet`-based graph
//! representation (the reference uses `networkx.DiGraph` over live Python object identity),
//! with two deliberate, documented deviations from the reference's exact algorithm shape:
//!
//! 1. **Children are normalized bottom-up before any graph-shape reduction runs on their
//!    parent** (matching this function's own contract), whereas the reference normalizes a
//!    choice graph's raw children only *after* doing silent-bypass/skippable-marking on the
//!    untouched children, then recurses. Normalizing first is strictly more thorough: a
//!    [`PartialOrderNode`] child that flattens down to a single silent leaf during its own
//!    normalization becomes visible to the parent's silent-bypass step, which it otherwise
//!    would not be.
//! 2. **Sequential cut-point detection** (the reference's `_abstract_sequences`) uses immediate
//!    dominators traced back from the graph's end node. This file instead uses the simpler,
//!    real algorithm the task explicitly sanctions: for each candidate node `c`, remove `c` and
//!    check whether the graph's end becomes unreachable from its start -- if so, `c` lies on
//!    every start-to-end path and is a genuine sequential cut point. This is less asymptotically
//!    efficient (`O(V * E)` instead of a single dominator-tree pass) but is real, correct, and
//!    does not require porting `networkx.immediate_dominators`.
//!
//! Strongly-connected-component detection (the reference's `_abstract_sccs`, which calls
//! `networkx.strongly_connected_components`) uses [`petgraph::algo::tarjan_scc`] here -- a real
//! Tarjan implementation from an already-vendored dependency, not hand-rolled.

use std::collections::{BTreeMap, BTreeSet};

use petgraph::algo::{is_cyclic_directed, tarjan_scc, toposort};
use petgraph::graphmap::DiGraphMap;

use crate::core::process_models::case_centric::process_tree::{Leaf, LeafLabel};

use super::{
    ChoiceGraphEndpoint, ChoiceGraphNode, Freq, PartialOrderNode, Powl, PowlLeaf, PowlNode,
    PowlOperator,
};

impl Powl {
    /// Structurally reduces this model to a smaller/simpler equivalent, applied bottom-up
    /// (every node's children are normalized before the node itself is reduced). Covers, at
    /// minimum:
    ///
    /// - **Silent-transition bypass**: a silent leaf child of a [`ChoiceGraphNode`] with at most
    ///   one predecessor or at most one successor edge is removed, wiring its predecessors
    ///   directly to its successors (fixpoint).
    /// - **Skippable-node marking**: a [`ChoiceGraphNode`] child whose full predecessor-set x
    ///   successor-set product is already covered by direct edges becomes skippable
    ///   (`min_freq = 0`), and the now-redundant direct edges are dropped.
    /// - **Self-loop abstraction**: the do/redo-loop and self-loop shapes
    ///   [`Powl::expand_frequency_tags`] (and [`ChoiceGraphNode::self_looping`] /
    ///   [`ChoiceGraphNode::do_redo_loop`]) produce are collapsed back into a single tagged
    ///   node -- the inverse of expansion.
    /// - **SCC-based nested abstraction**: a strongly connected component (size > 1) of a
    ///   choice graph with a clean, complete entry-set/exit-set boundary is abstracted into a
    ///   nested [`ChoiceGraphNode`].
    /// - **Sequence chunking**: the remaining acyclic graph is cut at genuine sequential cut
    ///   points into a [`PartialOrderNode`] of sequential stages, each recursively normalized.
    /// - **[`PartialOrderNode`] reduction**: silent-child bypass (order-relation predecessor/
    ///   successor count, not the transitive closure), then flattening to a bare child when
    ///   exactly one child remains, merging [`Freq`] per the reference's `flatten()` rule.
    /// - **[`PowlOperator`]**: children normalized recursively, node itself unchanged otherwise.
    /// - **[`PowlLeaf`]**: returned unchanged.
    pub fn normalize(&self) -> Powl {
        Powl::new(normalize_node(&self.root))
    }
}

// ---------------------------------------------------------------------------------------------
// Top-level dispatch (task items 6 and 7: PartialOrder / Operator / Leaf) plus the shared leaf
// helpers every branch needs.
// ---------------------------------------------------------------------------------------------

fn normalize_node(node: &PowlNode) -> PowlNode {
    match node {
        PowlNode::Leaf(leaf) => PowlNode::Leaf(clone_leaf(leaf)),
        PowlNode::Operator(op) => {
            let mut new_op = PowlOperator::new(op.operator_type_clone());
            new_op.children = op.children.iter().map(normalize_node).collect();
            new_op.freq = op.freq;
            PowlNode::Operator(new_op)
        }
        PowlNode::PartialOrder(po) => normalize_partial_order(po),
        PowlNode::ChoiceGraph(cg) => normalize_choice_graph(cg),
    }
}

fn clone_leaf(leaf: &PowlLeaf) -> PowlLeaf {
    PowlLeaf {
        leaf: Leaf::new(match &leaf.leaf.activity_label {
            LeafLabel::Activity(a) => Some(a.clone()),
            LeafLabel::Tau => None,
        }),
        freq: leaf.freq,
    }
}

/// `true` iff `node` is a silent (tau) leaf with the default exactly-once frequency. Bypass/
/// abstraction steps below only ever remove a tau under this narrower condition (the reference
/// checks silence alone) -- a deliberate safety margin so a tau deliberately re-tagged with a
/// non-default [`Freq`] (e.g. by a hand-built model) is never silently discarded.
fn is_silent_leaf(node: &PowlNode) -> bool {
    matches!(node, PowlNode::Leaf(l) if matches!(l.leaf.activity_label, LeafLabel::Tau) && l.freq == Freq::EXACTLY_ONE)
}

fn merge_freq(outer: Freq, inner: Freq) -> Freq {
    let min = outer.min_freq.min(inner.min_freq);
    let max = match (outer.max_freq, inner.max_freq) {
        (None, _) | (_, None) => None,
        (Some(a), Some(b)) => Some(a.max(b)),
    };
    Freq::new(min, max)
}

// ---------------------------------------------------------------------------------------------
// PartialOrderNode reduction (task item 6).
// ---------------------------------------------------------------------------------------------

/// Returns the direct (non-transitively-implied) edges of a transitively-closed order relation:
/// `(a, c)` is dropped whenever some `b` makes it implied by `(a, b)` and `(b, c)`. The same
/// computation [`PartialOrderNode::add_to_petri_net`] already performs inline, factored out here
/// as a standalone helper -- this fork's own established pattern, not a port of the reference
/// (whose `PartialOrder`'s backing `networkx.DiGraph` stores literal, non-closed edges instead).
fn direct_edges_of(order: &BTreeSet<(usize, usize)>) -> BTreeSet<(usize, usize)> {
    order
        .iter()
        .copied()
        .filter(|&(a, c)| !order.iter().any(|&(x, y)| x == a && y != c && order.contains(&(y, c))))
        .collect()
}

fn normalize_partial_order(po: &PartialOrderNode) -> PowlNode {
    let mut nodes: Vec<Option<PowlNode>> = po.children.iter().map(|c| Some(normalize_node(c))).collect();
    let mut edges: BTreeSet<(usize, usize)> = direct_edges_of(&po.order);

    loop {
        let silent_ids: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.as_ref().filter(|nd| is_silent_leaf(nd)).map(|_| i))
            .collect();

        let mut changed = false;
        for i in silent_ids {
            let preds: BTreeSet<usize> = edges.iter().filter(|&&(_, t)| t == i).map(|&(f, _)| f).collect();
            let succs: BTreeSet<usize> = edges.iter().filter(|&&(f, _)| f == i).map(|&(_, t)| t).collect();
            if preds.len() <= 1 || succs.len() <= 1 {
                edges.retain(|&(f, t)| f != i && t != i);
                for &p in &preds {
                    for &s in &succs {
                        // p == s cannot arise from a genuine strict partial order (it would
                        // require both (p, i) and (i, p), violating antisymmetry), but the
                        // check is kept as a defensive no-op rather than assumed.
                        if p != s {
                            edges.insert((p, s));
                        }
                    }
                }
                nodes[i] = None;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let remaining: Vec<(usize, PowlNode)> = nodes
        .into_iter()
        .enumerate()
        .filter_map(|(i, n)| n.map(|nd| (i, nd)))
        .collect();

    if remaining.is_empty() {
        // Every child was bypassed away: the whole partial order collapses to a no-op skip.
        let mut leaf = PowlLeaf::new(None);
        leaf.freq = po.freq;
        return PowlNode::Leaf(leaf);
    }
    if remaining.len() == 1 {
        let (_, mut only) = remaining.into_iter().next().expect("checked len == 1 above");
        let merged = merge_freq(po.freq, only.freq());
        only.set_freq(merged);
        return only;
    }

    let old_to_new: BTreeMap<usize, usize> = remaining
        .iter()
        .enumerate()
        .map(|(new_i, &(old_i, _))| (old_i, new_i))
        .collect();
    let new_children: Vec<PowlNode> = remaining.into_iter().map(|(_, n)| n).collect();
    let new_edges: Vec<(usize, usize)> = edges
        .iter()
        .filter_map(|&(f, t)| match (old_to_new.get(&f), old_to_new.get(&t)) {
            (Some(&nf), Some(&nt)) => Some((nf, nt)),
            _ => None,
        })
        .collect();

    let mut new_po = PartialOrderNode::new(new_children, new_edges);
    new_po.freq = po.freq;
    PowlNode::PartialOrder(new_po)
}

// ---------------------------------------------------------------------------------------------
// ChoiceGraphNode reduction (task items 1-4). Internal working representation: nodes get their
// own stable small-integer id (independent of `Vec` position, since SCC/sequence abstraction
// replace many ids with one new composite id), edges range over `GNode` = Start | Id(id) | End.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum GNode {
    Start,
    Id(usize),
    End,
}

struct CgGraph {
    nodes: BTreeMap<usize, PowlNode>,
    edges: BTreeSet<(GNode, GNode)>,
    next_id: usize,
}

impl CgGraph {
    fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            edges: BTreeSet::new(),
            next_id: 0,
        }
    }

    fn insert_node(&mut self, node: PowlNode) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, node);
        id
    }

    fn predecessors(&self, n: GNode) -> BTreeSet<GNode> {
        self.edges.iter().filter(|&&(_, t)| t == n).map(|&(f, _)| f).collect()
    }

    fn successors(&self, n: GNode) -> BTreeSet<GNode> {
        self.edges.iter().filter(|&&(f, _)| f == n).map(|&(_, t)| t).collect()
    }

    fn remove_node(&mut self, id: usize) {
        self.nodes.remove(&id);
        let gid = GNode::Id(id);
        self.edges.retain(|&(f, t)| f != gid && t != gid);
    }

    /// Consumes a [`ChoiceGraphNode`], assigning every child a fresh graph-local id.
    fn from_choice_graph(cg: ChoiceGraphNode) -> Self {
        let mut g = CgGraph::new();
        let mut id_of_child = Vec::with_capacity(cg.children.len());
        for child in cg.children {
            id_of_child.push(g.insert_node(child));
        }
        let map_endpoint = |ep: ChoiceGraphEndpoint| match ep {
            ChoiceGraphEndpoint::Start => GNode::Start,
            ChoiceGraphEndpoint::End => GNode::End,
            ChoiceGraphEndpoint::Child(i) => GNode::Id(id_of_child[i]),
        };
        for (from, to) in cg.edges {
            g.edges.insert((map_endpoint(from), map_endpoint(to)));
        }
        g
    }
}

fn normalize_choice_graph(cg: &ChoiceGraphNode) -> PowlNode {
    let normalized_children: Vec<PowlNode> = cg.children.iter().map(normalize_node).collect();
    let mut freq = cg.freq;

    let working = ChoiceGraphNode {
        children: normalized_children,
        edges: cg.edges.clone(),
        freq: Freq::EXACTLY_ONE,
    };
    let mut g = CgGraph::from_choice_graph(working);

    strip_trivial_self_loops(&mut g);
    reduce_simple_silent_transitions(&mut g, &mut freq);
    mark_skippable_nodes(&mut g);
    apply_self_loop_reduction(&mut g, &mut freq);

    graph_to_powl_node(g, freq)
}

/// Task item: a genuine cyclic self-edge `Child(i) -> Child(i)` (e.g. from
/// [`ChoiceGraphNode::self_looping`]) collapses to a bare repeatable node: drop the edge, mark
/// the node unbounded-repeatable (`max_freq = None`).
fn strip_trivial_self_loops(g: &mut CgGraph) {
    let looping_ids: Vec<usize> = g
        .nodes
        .keys()
        .copied()
        .filter(|&id| g.edges.contains(&(GNode::Id(id), GNode::Id(id))))
        .collect();
    for id in looping_ids {
        g.edges.remove(&(GNode::Id(id), GNode::Id(id)));
        let mut f = g.nodes[&id].freq();
        f.max_freq = None;
        g.nodes.get_mut(&id).unwrap().set_freq(f);
    }
}

/// Task item 1: fixpoint silent-transition bypass.
fn reduce_simple_silent_transitions(g: &mut CgGraph, freq: &mut Freq) {
    loop {
        let silent_ids: Vec<usize> = g
            .nodes
            .iter()
            .filter(|(_, n)| is_silent_leaf(n))
            .map(|(&id, _)| id)
            .collect();

        let mut changed = false;
        for id in silent_ids {
            if !g.nodes.contains_key(&id) {
                continue;
            }
            let tau = GNode::Id(id);
            let preds = g.predecessors(tau);
            let succs = g.successors(tau);
            if preds.len() <= 1 || succs.len() <= 1 {
                bypass_silent_node(g, freq, id, &preds, &succs);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn bypass_silent_node(g: &mut CgGraph, freq: &mut Freq, tau_id: usize, preds: &BTreeSet<GNode>, succs: &BTreeSet<GNode>) {
    for &p in preds {
        for &s in succs {
            if p == GNode::Start && s == GNode::End {
                // The tau was the whole graph's only content on this path: the graph itself
                // becomes skippable.
                freq.min_freq = 0;
            } else if p == GNode::End || s == GNode::Start {
                // Structurally unreachable for a valid choice graph (End has no outgoing edges,
                // Start has no incoming); skip defensively rather than panic on malformed input.
                continue;
            } else if p == s {
                // p was both predecessor and successor of tau: a genuine loop through tau back
                // onto itself. Mark p repeatable instead of adding a self-edge.
                if let GNode::Id(pid) = p {
                    let mut f = g.nodes[&pid].freq();
                    f.max_freq = None;
                    g.nodes.get_mut(&pid).unwrap().set_freq(f);
                }
            } else {
                g.edges.insert((p, s));
            }
        }
    }
    g.remove_node(tau_id);
}

/// Task item 2: a node whose predecessor-set x successor-set product is already fully covered by
/// direct edges is marked skippable and the now-redundant edges are dropped. Fixpoint.
fn mark_skippable_nodes(g: &mut CgGraph) {
    loop {
        let mut edges_to_remove: BTreeSet<(GNode, GNode)> = BTreeSet::new();
        let ids: Vec<usize> = g.nodes.keys().copied().collect();

        for id in ids {
            let node = GNode::Id(id);
            let preds = g.predecessors(node);
            let succs = g.successors(node);
            let fully_covered = preds.iter().all(|&p| succs.iter().all(|&s| g.edges.contains(&(p, s))));
            if fully_covered {
                for &p in &preds {
                    for &s in &succs {
                        edges_to_remove.insert((p, s));
                    }
                }
                let mut f = g.nodes[&id].freq();
                f.min_freq = 0;
                g.nodes.get_mut(&id).unwrap().set_freq(f);
            }
        }

        if edges_to_remove.is_empty() {
            break;
        }
        for e in &edges_to_remove {
            g.edges.remove(e);
        }
    }
}

/// Task item 3: recognizes the skippable self-loop (`do_redo_loop(silent, body)`) and
/// non-skippable self-loop (`do_redo_loop(body, silent)`) shapes via their silent do/redo part,
/// plus the raw-back-edge shape (a cycle not routed through any silent node), and collapses each
/// into the appropriate [`Freq`] tag on the surviving structure. Returns whether anything
/// changed.
fn apply_self_loop_reduction(g: &mut CgGraph, freq: &mut Freq) -> bool {
    let start_nodes: BTreeSet<GNode> = g.successors(GNode::Start);
    let end_nodes: BTreeSet<GNode> = g.predecessors(GNode::End);

    let silent_ids: Vec<usize> = g.nodes.iter().filter(|(_, n)| is_silent_leaf(n)).map(|(&id, _)| id).collect();

    for id in silent_ids {
        let tau = GNode::Id(id);
        let preds = g.predecessors(tau);
        let succs = g.successors(tau);

        if start_nodes.len() == 1 && start_nodes.contains(&tau) && end_nodes.len() == 1 && end_nodes.contains(&tau) {
            // Skippable self-loop: tau is the sole start AND sole end node.
            for &p in preds.iter() {
                if p != GNode::Start {
                    g.edges.insert((p, GNode::End));
                }
            }
            for &s in succs.iter() {
                if s != GNode::End {
                    g.edges.insert((GNode::Start, s));
                }
            }
            g.remove_node(id);
            freq.min_freq = 0;
            freq.max_freq = None;
            return true;
        } else if preds == end_nodes && succs == start_nodes {
            // Non-skippable self-loop: tau's predecessors are exactly the graph's end nodes and
            // its successors are exactly the graph's start nodes (the redo-part of a
            // do_redo_loop(body, silent)).
            g.remove_node(id);
            freq.max_freq = None;
            return true;
        }
    }

    // No silent-node-routed loop: check for a raw back-edge cycle (every end node connects
    // directly to every start node).
    if end_nodes.is_empty() || start_nodes.is_empty() {
        return false;
    }
    let mut back_edges = Vec::new();
    for &u in &end_nodes {
        for &v in &start_nodes {
            if g.edges.contains(&(u, v)) {
                back_edges.push((u, v));
            } else {
                return false;
            }
        }
    }
    if back_edges.is_empty() {
        return false;
    }
    freq.max_freq = None;
    for e in back_edges {
        g.edges.remove(&e);
    }
    true
}

/// `_apply_advanced_reductions` equivalent: flatten a single remaining node, otherwise try SCC
/// abstraction then sequence chunking.
fn graph_to_powl_node(mut g: CgGraph, freq: Freq) -> PowlNode {
    if g.nodes.len() == 1 {
        return flatten_single(g, freq);
    }
    abstract_sccs(&mut g);
    if g.nodes.len() == 1 {
        return flatten_single(g, freq);
    }
    abstract_sequences(g, freq)
}

fn flatten_single(mut g: CgGraph, freq: Freq) -> PowlNode {
    let (_, mut node) = g.nodes.pop_first().expect("caller checked nodes.len() == 1");
    let merged = merge_freq(freq, node.freq());
    node.set_freq(merged);
    node
}

fn rebuild_choice_graph_node(g: CgGraph, freq: Freq) -> PowlNode {
    if g.nodes.is_empty() {
        let mut leaf = PowlLeaf::new(None);
        leaf.freq = freq;
        return PowlNode::Leaf(leaf);
    }

    let ids: Vec<usize> = g.nodes.keys().copied().collect();
    let local_index: BTreeMap<usize, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut nodes = g.nodes;
    let children: Vec<PowlNode> = ids.iter().map(|id| nodes.remove(id).expect("id from nodes.keys()")).collect();

    let map_endpoint = |n: GNode| -> ChoiceGraphEndpoint {
        match n {
            GNode::Start => ChoiceGraphEndpoint::Start,
            GNode::End => ChoiceGraphEndpoint::End,
            GNode::Id(x) => ChoiceGraphEndpoint::Child(local_index[&x]),
        }
    };
    let edges: BTreeSet<(ChoiceGraphEndpoint, ChoiceGraphEndpoint)> =
        g.edges.iter().map(|&(f, t)| (map_endpoint(f), map_endpoint(t))).collect();

    let mut cg = ChoiceGraphNode::new(children, edges);
    cg.freq = freq;
    PowlNode::ChoiceGraph(cg)
}

/// Task item 4: SCC-based nested abstraction via [`petgraph::algo::tarjan_scc`] (a real Tarjan
/// implementation, not hand-rolled). An SCC of size > 1 with a complete entry-set/exit-set
/// boundary (every outside-source -> inside-entry pair is a real edge, and likewise for
/// inside-exit -> outside-target) is contracted into one nested [`ChoiceGraphNode`].
fn abstract_sccs(g: &mut CgGraph) {
    loop {
        let mut gm: DiGraphMap<usize, ()> = DiGraphMap::new();
        for &id in g.nodes.keys() {
            gm.add_node(id);
        }
        for &(f, t) in &g.edges {
            if let (GNode::Id(fi), GNode::Id(ti)) = (f, t) {
                gm.add_edge(fi, ti, ());
            }
        }
        let sccs = tarjan_scc(&gm);

        let mut did_abstract = false;
        for scc in sccs {
            if scc.len() <= 1 {
                continue;
            }
            let scc_set: BTreeSet<usize> = scc.into_iter().collect();

            let mut a_set: BTreeSet<GNode> = BTreeSet::new();
            let mut b_set: BTreeSet<GNode> = BTreeSet::new();
            let mut edges_in = 0usize;
            let mut c_set: BTreeSet<GNode> = BTreeSet::new();
            let mut d_set: BTreeSet<GNode> = BTreeSet::new();
            let mut edges_out = 0usize;

            for &(f, t) in &g.edges {
                let f_in = matches!(f, GNode::Id(x) if scc_set.contains(&x));
                let t_in = matches!(t, GNode::Id(x) if scc_set.contains(&x));
                if !f_in && t_in {
                    a_set.insert(f);
                    b_set.insert(t);
                    edges_in += 1;
                } else if f_in && !t_in {
                    c_set.insert(f);
                    d_set.insert(t);
                    edges_out += 1;
                }
            }

            if a_set.is_empty() || b_set.is_empty() || c_set.is_empty() || d_set.is_empty() {
                // Should not happen for a valid choice graph (Def. 3.6 guarantees every SCC has
                // a real boundary both ways); skip this SCC defensively rather than panic.
                continue;
            }
            if edges_in != a_set.len() * b_set.len() {
                continue;
            }
            if edges_out != c_set.len() * d_set.len() {
                continue;
            }

            let sub_powl = build_scc_subgraph(g, &scc_set, &b_set, &c_set);
            let new_id = g.insert_node(sub_powl);

            let mut new_edges = BTreeSet::new();
            for &(f, t) in &g.edges {
                let f2 = if matches!(f, GNode::Id(x) if scc_set.contains(&x)) {
                    GNode::Id(new_id)
                } else {
                    f
                };
                let t2 = if matches!(t, GNode::Id(x) if scc_set.contains(&x)) {
                    GNode::Id(new_id)
                } else {
                    t
                };
                if f2 == t2 {
                    // Edge internal to the SCC being contracted away: discard.
                    continue;
                }
                new_edges.insert((f2, t2));
            }
            g.edges = new_edges;
            did_abstract = true;
            break; // Ids changed; recompute SCCs fresh on the next outer iteration.
        }

        if !did_abstract {
            break;
        }
    }
}

fn build_scc_subgraph(g: &mut CgGraph, scc_set: &BTreeSet<usize>, b_set: &BTreeSet<GNode>, c_set: &BTreeSet<GNode>) -> PowlNode {
    let ids: Vec<usize> = scc_set.iter().copied().collect();
    let local_index: BTreeMap<usize, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let sub_children: Vec<PowlNode> = ids.iter().map(|id| g.nodes.remove(id).expect("scc node must exist")).collect();

    let mut sub_edges: BTreeSet<(ChoiceGraphEndpoint, ChoiceGraphEndpoint)> = BTreeSet::new();
    for &(f, t) in &g.edges {
        if let (GNode::Id(fi), GNode::Id(ti)) = (f, t) {
            if scc_set.contains(&fi) && scc_set.contains(&ti) {
                sub_edges.insert((
                    ChoiceGraphEndpoint::Child(local_index[&fi]),
                    ChoiceGraphEndpoint::Child(local_index[&ti]),
                ));
            }
        }
    }
    for &b in b_set {
        if let GNode::Id(bi) = b {
            sub_edges.insert((ChoiceGraphEndpoint::Start, ChoiceGraphEndpoint::Child(local_index[&bi])));
        }
    }
    for &c in c_set {
        if let GNode::Id(ci) = c {
            sub_edges.insert((ChoiceGraphEndpoint::Child(local_index[&ci]), ChoiceGraphEndpoint::End));
        }
    }

    let sub_cg = ChoiceGraphNode::new(sub_children, sub_edges);
    let mut sub_freq = sub_cg.freq; // EXACTLY_ONE
    let mut sub_g = CgGraph::from_choice_graph(sub_cg);

    let changed = apply_self_loop_reduction(&mut sub_g, &mut sub_freq);
    if changed {
        graph_to_powl_node(sub_g, sub_freq)
    } else {
        // Still genuinely cyclic (no clean self-loop shape found): leave it as a real, valid,
        // if not further-simplified, nested ChoiceGraphNode.
        rebuild_choice_graph_node(sub_g, sub_freq)
    }
}

// ---------------------------------------------------------------------------------------------
// Task item 5: sequence chunking over the now-acyclic residual graph.
// ---------------------------------------------------------------------------------------------

fn is_acyclic(g: &CgGraph) -> bool {
    let mut gm: DiGraphMap<usize, ()> = DiGraphMap::new();
    for &id in g.nodes.keys() {
        gm.add_node(id);
    }
    for &(f, t) in &g.edges {
        if let (GNode::Id(fi), GNode::Id(ti)) = (f, t) {
            gm.add_edge(fi, ti, ());
        }
    }
    !is_cyclic_directed(&gm)
}

fn topo_positions(g: &CgGraph) -> BTreeMap<usize, usize> {
    let mut gm: DiGraphMap<usize, ()> = DiGraphMap::new();
    for &id in g.nodes.keys() {
        gm.add_node(id);
    }
    for &(f, t) in &g.edges {
        if let (GNode::Id(fi), GNode::Id(ti)) = (f, t) {
            gm.add_edge(fi, ti, ());
        }
    }
    let order = toposort(&gm, None).expect("is_acyclic checked by caller");
    order.into_iter().enumerate().map(|(i, id)| (id, i)).collect()
}

/// `true` iff removing `id` (and its incident edges) makes [`GNode::End`] unreachable from
/// [`GNode::Start`] -- i.e. every real Start-to-End path passes through `id`.
fn is_cut_point(g: &CgGraph, id: usize) -> bool {
    let blocked = GNode::Id(id);
    let mut visited: BTreeSet<GNode> = BTreeSet::new();
    let mut stack = vec![GNode::Start];
    visited.insert(GNode::Start);
    while let Some(n) = stack.pop() {
        for &(f, t) in &g.edges {
            if f == n && t != blocked && visited.insert(t) {
                stack.push(t);
            }
        }
    }
    !visited.contains(&GNode::End)
}

fn reachable_forward(g: &CgGraph, start: GNode) -> BTreeSet<GNode> {
    let mut visited = BTreeSet::new();
    visited.insert(start);
    let mut stack = vec![start];
    while let Some(n) = stack.pop() {
        for &(f, t) in &g.edges {
            if f == n && visited.insert(t) {
                stack.push(t);
            }
        }
    }
    visited.remove(&start);
    visited
}

fn reachable_backward(g: &CgGraph, target: GNode) -> BTreeSet<GNode> {
    let mut visited = BTreeSet::new();
    visited.insert(target);
    let mut stack = vec![target];
    while let Some(n) = stack.pop() {
        for &(f, t) in &g.edges {
            if t == n && visited.insert(f) {
                stack.push(f);
            }
        }
    }
    visited.remove(&target);
    visited
}

/// Real internal nodes reachable from `left` that can also reach `right`, excluding both
/// endpoints and any other cut point -- the "gap" of ordinary nodes strictly between two
/// adjacent sequential cut points (or a boundary and the nearest cut point).
fn gap_between(g: &CgGraph, left: GNode, right: GNode, cut_set: &BTreeSet<usize>) -> BTreeSet<usize> {
    let forward = reachable_forward(g, left);
    let backward = reachable_backward(g, right);
    forward
        .intersection(&backward)
        .filter_map(|&n| match n {
            GNode::Id(id) if !cut_set.contains(&id) => Some(id),
            _ => None,
        })
        .collect()
}

fn build_gap_chunk(g: &mut CgGraph, gap_ids: &BTreeSet<usize>, skippable: bool) -> PowlNode {
    if gap_ids.len() == 1 {
        let id = *gap_ids.iter().next().expect("checked len == 1");
        let mut node = g.nodes.remove(&id).expect("gap node must exist");
        if skippable {
            let mut f = node.freq();
            f.min_freq = 0;
            node.set_freq(f);
        }
        return node;
    }

    let ids: Vec<usize> = gap_ids.iter().copied().collect();
    let local_index: BTreeMap<usize, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let sub_children: Vec<PowlNode> = ids.iter().map(|id| g.nodes.remove(id).expect("gap node must exist")).collect();

    let mut sub_edges: BTreeSet<(ChoiceGraphEndpoint, ChoiceGraphEndpoint)> = BTreeSet::new();
    for &(f, t) in &g.edges {
        if let (GNode::Id(fi), GNode::Id(ti)) = (f, t) {
            if gap_ids.contains(&fi) && gap_ids.contains(&ti) {
                sub_edges.insert((
                    ChoiceGraphEndpoint::Child(local_index[&fi]),
                    ChoiceGraphEndpoint::Child(local_index[&ti]),
                ));
            }
        }
    }
    for &id in &ids {
        let has_outside_pred = g
            .edges
            .iter()
            .any(|&(f, t)| t == GNode::Id(id) && !matches!(f, GNode::Id(x) if gap_ids.contains(&x)));
        if has_outside_pred {
            sub_edges.insert((ChoiceGraphEndpoint::Start, ChoiceGraphEndpoint::Child(local_index[&id])));
        }
        let has_outside_succ = g
            .edges
            .iter()
            .any(|&(f, t)| f == GNode::Id(id) && !matches!(t, GNode::Id(x) if gap_ids.contains(&x)));
        if has_outside_succ {
            sub_edges.insert((ChoiceGraphEndpoint::Child(local_index[&id]), ChoiceGraphEndpoint::End));
        }
    }

    let mut sub_cg = ChoiceGraphNode::new(sub_children, sub_edges);
    sub_cg.freq = Freq::new(if skippable { 0 } else { 1 }, Some(1));
    let sub_freq = sub_cg.freq;
    let sub_g = CgGraph::from_choice_graph(sub_cg);
    graph_to_powl_node(sub_g, sub_freq)
}

fn abstract_sequences(mut g: CgGraph, mut freq: Freq) -> PowlNode {
    apply_self_loop_reduction(&mut g, &mut freq);
    if g.nodes.len() == 1 {
        return flatten_single(g, freq);
    }
    if g.nodes.is_empty() {
        let mut leaf = PowlLeaf::new(None);
        leaf.freq = freq;
        return PowlNode::Leaf(leaf);
    }
    if !is_acyclic(&g) {
        // An SCC failed the clean-boundary check and stayed genuinely cyclic: sequential
        // chunking's cut-point argument requires acyclicity, so stop here with a real, valid,
        // if not maximally reduced, ChoiceGraphNode.
        return rebuild_choice_graph_node(g, freq);
    }

    let pos = topo_positions(&g);
    let mut cut_points: Vec<usize> = g.nodes.keys().copied().filter(|&id| is_cut_point(&g, id)).collect();
    cut_points.sort_by_key(|id| pos[id]);

    if cut_points.is_empty() {
        return rebuild_choice_graph_node(g, freq);
    }

    let cut_set: BTreeSet<usize> = cut_points.iter().copied().collect();
    let mut anchors: Vec<GNode> = vec![GNode::Start];
    anchors.extend(cut_points.iter().map(|&id| GNode::Id(id)));
    anchors.push(GNode::End);

    let mut chunk_list: Vec<PowlNode> = Vec::new();
    for w in 0..anchors.len() - 1 {
        let left = anchors[w];
        let right = anchors[w + 1];

        let gap_ids = gap_between(&g, left, right, &cut_set);
        let skippable = g.edges.contains(&(left, right));

        if !gap_ids.is_empty() {
            chunk_list.push(build_gap_chunk(&mut g, &gap_ids, skippable));
        }
        if let GNode::Id(id) = right {
            if cut_set.contains(&id) {
                chunk_list.push(g.nodes.remove(&id).expect("cut point node must exist"));
            }
        }
    }

    let mut chunks: Vec<PowlNode> = chunk_list.into_iter().filter(|c| !is_silent_leaf(c)).collect();

    if chunks.is_empty() {
        let mut leaf = PowlLeaf::new(None);
        leaf.freq = freq;
        return PowlNode::Leaf(leaf);
    }
    if chunks.len() == 1 {
        let mut only = chunks.pop().expect("checked len == 1");
        let merged = merge_freq(freq, only.freq());
        only.set_freq(merged);
        return only;
    }

    let n = chunks.len();
    let seq_edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let mut po = PartialOrderNode::new(chunks, seq_edges);
    po.freq = freq;
    PowlNode::PartialOrder(po)
}
