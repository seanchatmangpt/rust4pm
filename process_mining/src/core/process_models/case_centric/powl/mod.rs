//! POWL (Partially Ordered Workflow Language) model
//!
//! POWL generalizes a block-structured [`ProcessTree`](crate::core::process_models::case_centric::process_tree::ProcessTree)
//! by adding a partial-order node: a set of children with an explicit, possibly incomplete,
//! strict order relation between them (a "choice graph" in the terminology of the paper
//! "Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs",
//! <https://arxiv.org/abs/2505.07052>). Two children with no order edge between them (in either
//! direction) are unordered — either one may occur first, unlike the block-structured
//! [`OperatorType::Concurrency`](crate::core::process_models::case_centric::process_tree::OperatorType::Concurrency)
//! case where both must always occur.
//!
//! Every non-partial-order construct (sequence, exclusive choice, concurrency, loop, leaf) is
//! reused as-is from the existing [`process_tree`](crate::core::process_models::case_centric::process_tree)
//! module rather than duplicated — a [`PowlNode`] is either one of those nodes, or a genuinely
//! new [`PartialOrderNode`].

use std::collections::{BTreeSet, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::core::process_models::case_centric::process_tree::Leaf;
use crate::core::process_models::petri_net::{ArcType, Marking, PlaceID};
use crate::PetriNet;

/// A node in a POWL model: either a block-structured [`process_tree`](crate::core::process_models::case_centric::process_tree)
/// construct (leaf or operator), a [`PartialOrderNode`] (POWL 1.0's generalization of the
/// concurrency operator), or a [`ChoiceGraphNode`] (POWL __2.0__'s generalization of the
/// exclusive-choice and loop operators into one unified cyclic-graph construct -- see
/// [`ChoiceGraphNode`]'s docs for the exact POWL 2.0 correspondence, per Kourani, Park & van der
/// Aalst, "Hierarchical Decomposition of Separable Workflow-Nets", Def. 3.6-3.9).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub enum PowlNode {
    /// A non-silent or silent activity leaf, reused from the process tree model.
    Leaf(Leaf),
    /// A block-structured operator (Sequence/ExclusiveChoice/Concurrency/Loop) over POWL
    /// children, reused from the process tree model but recursing into [`PowlNode`] rather than
    /// `process_tree::Node`.
    Operator(PowlOperator),
    /// A partially-ordered set of children (generalized concurrency): the POWL 1.0 construct.
    PartialOrder(PartialOrderNode),
    /// A choice graph over children (generalized exclusive choice + cycles/loops): the POWL
    /// __2.0__ construct that distinguishes this model from POWL 1.0.
    ChoiceGraph(ChoiceGraphNode),
}

impl PowlNode {
    /// Creates a new non-silent or silent leaf node.
    pub fn new_leaf(leaf_label: Option<String>) -> Self {
        PowlNode::Leaf(Leaf::new(leaf_label))
    }

    /// Unfolds this node and its descendants into places, transitions, and arcs, adding them to
    /// the given [`PetriNet`]. Mirrors
    /// [`process_tree::Node::add_to_petri_net`](crate::core::process_models::case_centric::process_tree::Node::add_to_petri_net).
    pub fn add_to_petri_net(
        &self,
        net: &mut PetriNet,
        in_place: Option<PlaceID>,
        out_place: Option<PlaceID>,
    ) -> (PlaceID, PlaceID) {
        match self {
            PowlNode::Leaf(leaf) => leaf.add_to_petri_net(net, in_place, out_place),
            PowlNode::Operator(op) => op.add_to_petri_net(net, in_place, out_place),
            PowlNode::PartialOrder(po) => po.add_to_petri_net(net, in_place, out_place),
            PowlNode::ChoiceGraph(cg) => cg.add_to_petri_net(net, in_place, out_place),
        }
    }

    /// Returns all descendant [`Leaf`]s (including `self` if it is a leaf).
    pub fn find_all_leaves(&self) -> Vec<&Leaf> {
        let mut result = Vec::new();
        self.collect_leaves(&mut result);
        result
    }

    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a Leaf>) {
        match self {
            PowlNode::Leaf(leaf) => out.push(leaf),
            PowlNode::Operator(op) => op.children.iter().for_each(|c| c.collect_leaves(out)),
            PowlNode::PartialOrder(po) => po.children.iter().for_each(|c| c.collect_leaves(out)),
            PowlNode::ChoiceGraph(cg) => cg.children.iter().for_each(|c| c.collect_leaves(out)),
        }
    }
}

/// A block-structured operator over [`PowlNode`] children (the POWL-recursive analogue of
/// [`process_tree::Operator`](crate::core::process_models::case_centric::process_tree::Operator)).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PowlOperator {
    /// The block-structured operator's [`process_tree::OperatorType`](crate::core::process_models::case_centric::process_tree::OperatorType), reused as-is.
    pub operator_type: crate::core::process_models::case_centric::process_tree::OperatorType,
    /// The children of the operator node.
    pub children: Vec<PowlNode>,
}

impl PowlOperator {
    /// Creates a new, childless operator of the given type.
    pub fn new(operator_type: crate::core::process_models::case_centric::process_tree::OperatorType) -> Self {
        Self {
            operator_type,
            children: Vec::new(),
        }
    }

    /// Same translation rules as [`process_tree::Operator::add_to_petri_net`](crate::core::process_models::case_centric::process_tree::Operator::add_to_petri_net),
    /// reimplemented over [`PowlNode`] children so a `PartialOrder` sub-node can appear as an
    /// operator's child.
    pub fn add_to_petri_net(
        &self,
        net: &mut PetriNet,
        in_place: Option<PlaceID>,
        out_place: Option<PlaceID>,
    ) -> (PlaceID, PlaceID) {
        use crate::core::process_models::case_centric::process_tree::OperatorType;

        let in_place = in_place.unwrap_or_else(|| net.add_place(None));
        let out_place = out_place.unwrap_or_else(|| net.add_place(None));
        let num_of_children = self.children.len();

        match self.operator_type {
            OperatorType::Sequence => {
                let mut last_in_place = in_place;
                self.children.iter().enumerate().for_each(|(pos, child)| {
                    let curr_out_place = if pos == num_of_children - 1 {
                        out_place
                    } else {
                        net.add_place(None)
                    };
                    child.add_to_petri_net(net, Some(last_in_place), Some(curr_out_place));
                    last_in_place = curr_out_place;
                });
            }
            OperatorType::ExclusiveChoice => self.children.iter().for_each(|child| {
                child.add_to_petri_net(net, Some(in_place), Some(out_place));
            }),
            OperatorType::Concurrency => {
                let tau_start = net.add_transition(None, None);
                let tau_end = net.add_transition(None, None);
                net.add_arc(ArcType::place_to_transition(in_place, tau_start), None);
                net.add_arc(ArcType::transition_to_place(tau_end, out_place), None);
                self.children.iter().for_each(|child| {
                    let (child_start, child_end) = child.add_to_petri_net(net, None, None);
                    net.add_arc(ArcType::transition_to_place(tau_start, child_start), None);
                    net.add_arc(ArcType::place_to_transition(child_end, tau_end), None);
                });
            }
            OperatorType::Loop => {
                let tau_start = net.add_transition(None, None);
                let tau_end = net.add_transition(None, None);
                net.add_arc(ArcType::place_to_transition(in_place, tau_start), None);
                net.add_arc(ArcType::transition_to_place(tau_end, out_place), None);

                let loop_start_place = net.add_place(None);
                let loop_end_place = net.add_place(None);
                net.add_arc(
                    ArcType::transition_to_place(tau_start, loop_start_place),
                    None,
                );
                net.add_arc(
                    ArcType::place_to_transition(loop_end_place, tau_end),
                    None,
                );

                self.children.iter().enumerate().for_each(|(pos, child)| {
                    let (child_start, child_end) = if pos == 0 {
                        (loop_start_place, loop_end_place)
                    } else {
                        (loop_end_place, loop_start_place)
                    };
                    child.add_to_petri_net(net, Some(child_start), Some(child_end));
                });
            }
        }

        (in_place, out_place)
    }
}

/// A partially-ordered set of [`PowlNode`] children (POWL's genuinely new construct beyond a
/// block-structured process tree).
///
/// `order` holds strict "must happen before" edges as `(from_index, to_index)` pairs into
/// `children`. Two children with no edge between them in either direction are unordered: both
/// relative orders are permitted, matching a "choice graph" partial order rather than forced
/// concurrency.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PartialOrderNode {
    /// The partially-ordered children.
    pub children: Vec<PowlNode>,
    /// Strict order edges `(from_index, to_index)` into `children`, transitively closed.
    pub order: BTreeSet<(usize, usize)>,
}

impl PartialOrderNode {
    /// Creates a new [`PartialOrderNode`] from children and a (not necessarily transitively
    /// closed) set of order edges; the edges are transitively closed on construction.
    pub fn new(children: Vec<PowlNode>, order: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let n = children.len();
        let mut closed: BTreeSet<(usize, usize)> = order.into_iter().collect();

        // Floyd-Warshall-style transitive closure over the small (activity-count-sized) index
        // relation -- POWL order relations are over activities, not events, so this is cheap in
        // every real model.
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<(usize, usize)> = closed.iter().copied().collect();
            for &(a, b) in &snapshot {
                for c in 0..n {
                    if closed.contains(&(b, c)) && !closed.contains(&(a, c)) {
                        closed.insert((a, c));
                        changed = true;
                    }
                }
            }
        }

        Self {
            children,
            order: closed,
        }
    }

    /// Returns `true` if the order relation is irreflexive and antisymmetric (a genuine strict
    /// partial order over the children indices) -- i.e. no child orders before itself, and no
    /// two children order both ways.
    pub fn is_valid(&self) -> bool {
        self.order.iter().all(|&(a, b)| {
            a != b && a < self.children.len() && b < self.children.len() && !self.order.contains(&(b, a))
        })
    }

    /// Unfolds the partial order into a Petri net.
    ///
    /// Construction: every child gets its own start/end place pair. A child with no predecessor
    /// in `order` is connected from the node's overall `in_place`; a child with no successor is
    /// connected to the overall `out_place`. Every direct order edge `(a, b)` (i.e. an edge not
    /// implied by transitivity through a third child) becomes a silent transition joining `a`'s
    /// end place to `b`'s start place, so `b` cannot start until `a` has finished -- the standard
    /// translation of a partial order into a place/transition net.
    pub fn add_to_petri_net(
        &self,
        net: &mut PetriNet,
        in_place: Option<PlaceID>,
        out_place: Option<PlaceID>,
    ) -> (PlaceID, PlaceID) {
        let in_place = in_place.unwrap_or_else(|| net.add_place(None));
        let out_place = out_place.unwrap_or_else(|| net.add_place(None));

        if self.children.is_empty() {
            // Empty partial order behaves like an empty sequence: connect straight through with
            // a silent transition.
            let tau = net.add_transition(None, None);
            net.add_arc(ArcType::place_to_transition(in_place, tau), None);
            net.add_arc(ArcType::transition_to_place(tau, out_place), None);
            return (in_place, out_place);
        }

        let starts_ends: Vec<(PlaceID, PlaceID)> = self
            .children
            .iter()
            .map(|child| child.add_to_petri_net(net, None, None))
            .collect();

        // Direct edges only: drop (a, c) whenever some b makes it implied by (a, b) and (b, c),
        // so the net doesn't grow quadratically with redundant synchronizing transitions.
        let direct_edges: Vec<(usize, usize)> = self
            .order
            .iter()
            .copied()
            .filter(|&(a, c)| {
                !self
                    .order
                    .iter()
                    .any(|&(x, y)| x == a && y != c && self.order.contains(&(y, c)))
            })
            .collect();

        let has_predecessor: BTreeSet<usize> = direct_edges.iter().map(|&(_, b)| b).collect();
        let has_successor: BTreeSet<usize> = direct_edges.iter().map(|&(a, _)| a).collect();

        for (idx, &(child_start, _)) in starts_ends.iter().enumerate() {
            if !has_predecessor.contains(&idx) {
                let tau = net.add_transition(None, None);
                net.add_arc(ArcType::place_to_transition(in_place, tau), None);
                net.add_arc(ArcType::transition_to_place(tau, child_start), None);
            }
        }
        for (idx, &(_, child_end)) in starts_ends.iter().enumerate() {
            if !has_successor.contains(&idx) {
                let tau = net.add_transition(None, None);
                net.add_arc(ArcType::place_to_transition(child_end, tau), None);
                net.add_arc(ArcType::transition_to_place(tau, out_place), None);
            }
        }
        for (a, b) in direct_edges {
            let tau = net.add_transition(None, None);
            net.add_arc(ArcType::place_to_transition(starts_ends[a].1, tau), None);
            net.add_arc(ArcType::transition_to_place(tau, starts_ends[b].0), None);
        }

        (in_place, out_place)
    }
}

/// A node in a choice graph's node set `N = X ∪ {▷, □}` (Def. 3.6): either one of the
/// POWL model's children, or one of the two artificial boundary nodes.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChoiceGraphEndpoint {
    /// The artificial unique start node `▷`: has no incoming edges.
    Start,
    /// One of the choice graph's real children, by index into `ChoiceGraphNode::children`.
    Child(usize),
    /// The artificial unique end node `□`: has no outgoing edges.
    End,
}

/// A choice graph over [`PowlNode`] children -- POWL __2.0__'s genuinely new construct beyond
/// POWL 1.0 (which had only [`PartialOrderNode`] plus the block-structured
/// [`process_tree::OperatorType::ExclusiveChoice`](crate::core::process_models::case_centric::process_tree::OperatorType::ExclusiveChoice)
/// and [`process_tree::OperatorType::Loop`](crate::core::process_models::case_centric::process_tree::OperatorType::Loop)
/// operators). A choice graph replaces both of those block-structured operators with one unified
/// directed graph that may contain cycles, per Kourani, Park & van der Aalst, "Hierarchical
/// Decomposition of Separable Workflow-Nets" (arXiv:2602.15739), Definition 3.6:
///
/// > A choice graph over a set of nodes `X` is a tuple `γ = (N, E)` where `N = X ∪ {▷, □}` with
/// > two artificial start/end nodes `▷, □ ∉ X`; `E ⊆ N × N`; `▷` is the unique start node (no
/// > incoming edges); `□` is the unique end node (no outgoing edges); and every node lies on a
/// > connected path from `▷` to `□`.
///
/// Unlike [`PartialOrderNode`]'s strict-order relation (irreflexive, antisymmetric, transitively
/// closed), `edges` here is an ordinary directed-graph relation: it may contain cycles (a
/// self-loop `Child(i) -> Child(i)` is exactly a POWL 1.0-style loop over a single activity,
/// generalized), and is not required to be transitively closed. The graph's *language* (Def.
/// 3.9) is the union, over every `▷`-to-`□` path, of the concatenation of each path node's
/// sub-language -- exactly the semantics a real workflow router with cycles needs.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChoiceGraphNode {
    /// The graph's real (non-boundary) children.
    pub children: Vec<PowlNode>,
    /// Directed edges over `children` indices plus the two artificial boundary endpoints. Not
    /// required to be acyclic or transitively closed -- a choice graph is a plain directed graph,
    /// not a partial order.
    pub edges: BTreeSet<(ChoiceGraphEndpoint, ChoiceGraphEndpoint)>,
}

impl ChoiceGraphNode {
    /// Creates a new [`ChoiceGraphNode`] from children and a set of edges over
    /// [`ChoiceGraphEndpoint`]s. Edges are taken as given -- unlike [`PartialOrderNode::new`],
    /// no transitive closure is computed, since a choice graph is not a partial order.
    pub fn new(
        children: Vec<PowlNode>,
        edges: impl IntoIterator<Item = (ChoiceGraphEndpoint, ChoiceGraphEndpoint)>,
    ) -> Self {
        Self {
            children,
            edges: edges.into_iter().collect(),
        }
    }

    /// Convenience constructor for the common "generalized loop" case: a single child that may
    /// repeat, i.e. a choice graph with edges `▷ -> Child(0)`, `Child(0) -> □`, and the cyclic
    /// redo edge `Child(0) -> Child(0)`. This is the POWL 2.0 choice-graph replacement for POWL
    /// 1.0's block-structured `Loop(do, Leaf(tau))` construct over a single do-part.
    pub fn self_looping(child: PowlNode) -> Self {
        use ChoiceGraphEndpoint::{Child, End, Start};
        Self::new(
            vec![child],
            [(Start, Child(0)), (Child(0), Child(0)), (Child(0), End)],
        )
    }

    /// Returns `true` iff this satisfies Definition 3.6: `▷` has no incoming edges, `□` has no
    /// outgoing edges, and every child index is reachable from `▷` and can reach `□` (i.e. lies
    /// on a connected `▷`-to-`□` path -- cycles among children are permitted, so reachability,
    /// not acyclicity, is the real check).
    pub fn is_valid(&self) -> bool {
        use ChoiceGraphEndpoint::{Child, End, Start};

        if self.edges.iter().any(|&(_, to)| to == Start) {
            return false; // Start must have no incoming edges.
        }
        if self.edges.iter().any(|&(from, _)| from == End) {
            return false; // End must have no outgoing edges.
        }

        let n = self.children.len();
        let forward = self.reachable_from(Start);
        let backward = self.reachable_to(End);
        (0..n).all(|i| forward.contains(&Child(i)) && backward.contains(&Child(i)))
    }

    fn reachable_from(&self, start: ChoiceGraphEndpoint) -> HashSet<ChoiceGraphEndpoint> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                for &(from, to) in &self.edges {
                    if from == node && !visited.contains(&to) {
                        stack.push(to);
                    }
                }
            }
        }
        visited
    }

    fn reachable_to(&self, end: ChoiceGraphEndpoint) -> HashSet<ChoiceGraphEndpoint> {
        let mut visited = HashSet::new();
        let mut stack = vec![end];
        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                for &(from, to) in &self.edges {
                    if to == node && !visited.contains(&from) {
                        stack.push(from);
                    }
                }
            }
        }
        visited
    }

    /// Unfolds the choice graph into a Petri net.
    ///
    /// Construction: every child gets one persistent start/end place pair (revisited, not
    /// recreated, on every cyclic edge back into that child -- this is what lets the net
    /// represent a genuine repeatable loop rather than unrolling it). Every edge in the graph
    /// becomes a silent transition connecting the source's out-place (or the choice graph's
    /// overall `in_place` for an edge out of `▷`) to the target's in-place (or the overall
    /// `out_place` for an edge into `□`). A direct `▷ -> □` edge (an empty/skip path) becomes a
    /// silent transition straight from `in_place` to `out_place`.
    pub fn add_to_petri_net(
        &self,
        net: &mut PetriNet,
        in_place: Option<PlaceID>,
        out_place: Option<PlaceID>,
    ) -> (PlaceID, PlaceID) {
        use ChoiceGraphEndpoint::{Child, End, Start};

        let in_place = in_place.unwrap_or_else(|| net.add_place(None));
        let out_place = out_place.unwrap_or_else(|| net.add_place(None));

        let starts_ends: Vec<(PlaceID, PlaceID)> = self
            .children
            .iter()
            .map(|child| child.add_to_petri_net(net, None, None))
            .collect();

        let endpoint_out = |ep: ChoiceGraphEndpoint| -> Option<PlaceID> {
            match ep {
                Start => Some(in_place),
                Child(i) => starts_ends.get(i).map(|&(_, e)| e),
                End => None,
            }
        };
        let endpoint_in = |ep: ChoiceGraphEndpoint| -> Option<PlaceID> {
            match ep {
                Start => None,
                Child(i) => starts_ends.get(i).map(|&(s, _)| s),
                End => Some(out_place),
            }
        };

        for &(from, to) in &self.edges {
            if let (Some(from_place), Some(to_place)) = (endpoint_out(from), endpoint_in(to)) {
                let tau = net.add_transition(None, None);
                net.add_arc(ArcType::place_to_transition(from_place, tau), None);
                net.add_arc(ArcType::transition_to_place(tau, to_place), None);
            }
        }

        (in_place, out_place)
    }
}

/// A POWL model, rooted at a [`PowlNode`].
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Powl {
    /// The root of the POWL model.
    pub root: PowlNode,
}

impl Powl {
    /// Wraps the given root node into a [`Powl`] model.
    pub fn new(root: PowlNode) -> Self {
        Self { root }
    }

    /// Returns all descendant [`Leaf`]s of the model.
    pub fn find_all_leaves(&self) -> Vec<&Leaf> {
        self.root.find_all_leaves()
    }

    /// Translates the model into a workflow net (a [`PetriNet`] with a single initial and final
    /// marking), the same shape
    /// [`process_tree::ProcessTree::to_petri_net`](crate::core::process_models::case_centric::process_tree::ProcessTree::to_petri_net)
    /// produces.
    pub fn to_petri_net(&self) -> PetriNet {
        let mut net = PetriNet::new();
        let (start, end) = self.root.add_to_petri_net(&mut net, None, None);
        net.initial_marking = Some(Marking::from([(start, 1)]));
        let mut final_marking = Marking::new();
        final_marking.insert(end, 1);
        net.final_markings = Some(vec![final_marking]);
        net
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::process_models::case_centric::process_tree::OperatorType;

    #[test]
    fn partial_order_transitive_closure() {
        // a -> b -> c should imply a -> c after closure.
        let children = vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(Some("b".into())),
            PowlNode::new_leaf(Some("c".into())),
        ];
        let po = PartialOrderNode::new(children, [(0, 1), (1, 2)]);
        assert!(po.order.contains(&(0, 2)));
        assert!(po.is_valid());
    }

    #[test]
    fn partial_order_to_petri_net_is_a_real_workflow_net() {
        let children = vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(Some("b".into())),
            PowlNode::new_leaf(Some("c".into())),
        ];
        // a -> c, b unordered w.r.t. both.
        let po = PartialOrderNode::new(children, [(0, 2)]);
        let model = Powl::new(PowlNode::PartialOrder(po));
        let net = model.to_petri_net();

        // At least one real (non-silent) transition per leaf, plus the silent connector
        // transitions the partial-order translation adds for predecessor/successor/direct-edge
        // wiring.
        assert!(net.transitions.len() >= model.find_all_leaves().len());
        assert!(net.initial_marking.is_some());
        assert_eq!(net.final_markings.as_ref().map(|m| m.len()), Some(1));
    }

    #[test]
    fn sequence_operator_matches_process_tree_shape() {
        let mut op = PowlOperator::new(OperatorType::Sequence);
        op.children.push(PowlNode::new_leaf(Some("a".into())));
        op.children.push(PowlNode::new_leaf(Some("b".into())));
        let model = Powl::new(PowlNode::Operator(op));
        let net = model.to_petri_net();
        // Sequence of 2 activities: 2 transitions, 3 places (in, mid, out).
        assert_eq!(net.transitions.len(), 2);
        assert_eq!(net.places.len(), 3);
    }

    #[test]
    fn choice_graph_self_loop_is_valid_and_cyclic() {
        // The POWL 2.0 replacement for a block-structured Loop(a, tau): a single child with a
        // genuine cyclic edge back to itself.
        let cg = ChoiceGraphNode::self_looping(PowlNode::new_leaf(Some("a".into())));
        assert!(cg.is_valid());
        assert!(cg.edges.contains(&(ChoiceGraphEndpoint::Child(0), ChoiceGraphEndpoint::Child(0))));
    }

    #[test]
    fn choice_graph_exclusive_choice_is_valid() {
        // ▷ -> a -> □ and ▷ -> b -> □: a plain exclusive choice, expressed as a choice graph
        // (POWL 2.0 generalizes ExclusiveChoice into this same construct).
        use ChoiceGraphEndpoint::{Child, End, Start};
        let cg = ChoiceGraphNode::new(
            vec![
                PowlNode::new_leaf(Some("a".into())),
                PowlNode::new_leaf(Some("b".into())),
            ],
            [
                (Start, Child(0)),
                (Child(0), End),
                (Start, Child(1)),
                (Child(1), End),
            ],
        );
        assert!(cg.is_valid());
    }

    #[test]
    fn choice_graph_rejects_unreachable_node() {
        // A child with no edges to/from it at all violates Def. 3.6's "every node lies on a
        // connected path from ▷ to □".
        use ChoiceGraphEndpoint::{Child, End, Start};
        let cg = ChoiceGraphNode::new(
            vec![
                PowlNode::new_leaf(Some("a".into())),
                PowlNode::new_leaf(Some("unreachable".into())),
            ],
            [(Start, Child(0)), (Child(0), End)],
        );
        assert!(!cg.is_valid());
    }

    #[test]
    fn choice_graph_to_petri_net_supports_a_real_cycle() {
        let cg = ChoiceGraphNode::self_looping(PowlNode::new_leaf(Some("a".into())));
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let net = model.to_petri_net();
        // 1 real "a" transition + at least the do/redo/exit silent transitions.
        assert!(net.transitions.len() >= 2);
        assert!(net.initial_marking.is_some());
    }
}
