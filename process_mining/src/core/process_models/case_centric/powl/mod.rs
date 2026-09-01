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

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::core::process_models::case_centric::process_tree::Leaf;
use crate::core::process_models::petri_net::{ArcType, Marking, PlaceID};
use crate::PetriNet;

/// A node in a POWL model: either a block-structured [`process_tree`](crate::core::process_models::case_centric::process_tree)
/// construct (leaf or operator), or a [`PartialOrderNode`].
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub enum PowlNode {
    /// A non-silent or silent activity leaf, reused from the process tree model.
    Leaf(Leaf),
    /// A block-structured operator (Sequence/ExclusiveChoice/Concurrency/Loop) over POWL
    /// children, reused from the process tree model but recursing into [`PowlNode`] rather than
    /// `process_tree::Node`.
    Operator(PowlOperator),
    /// A partially-ordered set of children: the genuinely new POWL construct.
    PartialOrder(PartialOrderNode),
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
}
