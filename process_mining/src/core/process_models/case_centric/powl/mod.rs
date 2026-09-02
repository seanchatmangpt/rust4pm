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
//! new [`PartialOrderNode`]/[`ChoiceGraphNode`].
//!
//! Every node also carries a [`Freq`] multiplicity tag (min/max occurrence count), matching the
//! real reference implementation's `TaggedPOWL.min_freq`/`max_freq` semantics
//! (`~/POWL/powl/objects/tagged_powl/base.py`), and [`Powl::expand_frequency_tags`] materializes
//! non-default frequency tags into real choice-graph structure before Petri net compilation --
//! an independent reimplementation of `~/POWL/powl/objects/tagged_powl/builders.py`'s
//! `expand_frequency_tags`/`_wrap_frequency_tags`, verified against that source this session.

use std::collections::{BTreeSet, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::core::process_models::case_centric::process_tree::Leaf;
use crate::core::process_models::petri_net::{ArcType, Marking, PlaceID};
use crate::PetriNet;

mod normalize;

/// A multiplicity/frequency tag on a POWL node: how many times it may occur.
///
/// Mirrors the reference implementation's `TaggedPOWL.min_freq`/`max_freq` semantics
/// (`~/POWL/powl/objects/tagged_powl/base.py:13-30,94-104`): `min_freq` is the minimum
/// occurrence count (`0` = skippable), `max_freq` is the maximum (`None` = unbounded). The
/// default, [`Freq::EXACTLY_ONE`], means "occurs exactly once" -- every node in this fork's
/// prior (pre-frequency-tag) API implicitly had this tag, so adding it is purely additive.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
pub struct Freq {
    /// Minimum occurrence count. `0` means the node is skippable.
    pub min_freq: u32,
    /// Maximum occurrence count. `None` means unbounded.
    pub max_freq: Option<u32>,
}

impl Freq {
    /// The default tag: occurs exactly once.
    pub const EXACTLY_ONE: Freq = Freq {
        min_freq: 1,
        max_freq: Some(1),
    };

    /// Creates a new frequency tag. `max_freq`, if present, must be `>= min_freq` (checked with
    /// a debug assertion, mirroring the reference's `_validate_freqs`, which raises `ValueError`
    /// under the same condition -- kept as a debug assertion here rather than a `Result` because
    /// every call site in this crate constructs `Freq` from a literal or a value already known
    /// valid, matching the "only literals and proven-prior-line values" exception for `.unwrap`-
    /// style invariants).
    pub fn new(min_freq: u32, max_freq: Option<u32>) -> Self {
        if let Some(max) = max_freq {
            debug_assert!(
                max >= min_freq,
                "Freq::new: max_freq ({max}) must be >= min_freq ({min_freq})"
            );
        }
        Self { min_freq, max_freq }
    }

    /// `true` iff this node may be skipped entirely (`min_freq == 0`).
    pub fn is_skippable(&self) -> bool {
        self.min_freq == 0
    }

    /// `true` iff this node may occur more than once (`max_freq` is `None` or `> 1`). Matches
    /// the reference's `is_repeatable()` exactly, including its notable simplification: a
    /// finite `max_freq` greater than 1 is treated identically to unbounded for the purpose of
    /// [`Powl::expand_frequency_tags`] (see that function's docs) -- a real, deliberate
    /// simplification in the reference, not an approximation introduced here.
    pub fn is_repeatable(&self) -> bool {
        self.max_freq.is_none_or(|m| m > 1)
    }

    /// `true` iff this node has no upper occurrence bound.
    pub fn is_unbounded(&self) -> bool {
        self.max_freq.is_none()
    }
}

impl Default for Freq {
    fn default() -> Self {
        Self::EXACTLY_ONE
    }
}

/// A leaf activity carrying a [`Freq`] tag -- the POWL-recursive analogue of
/// [`process_tree::Leaf`](crate::core::process_models::case_centric::process_tree::Leaf), which
/// has no frequency concept of its own.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PowlLeaf {
    /// The wrapped activity leaf (silent or non-silent), reused from the process tree model.
    pub leaf: Leaf,
    /// This leaf's multiplicity tag.
    pub freq: Freq,
}

impl PowlLeaf {
    /// Creates a new leaf with the default (exactly-once) frequency tag.
    pub fn new(leaf_label: Option<String>) -> Self {
        Self {
            leaf: Leaf::new(leaf_label),
            freq: Freq::EXACTLY_ONE,
        }
    }
}

/// A node in a POWL model: either a block-structured [`process_tree`](crate::core::process_models::case_centric::process_tree)
/// construct (leaf or operator), a [`PartialOrderNode`] (POWL 1.0's generalization of the
/// concurrency operator), or a [`ChoiceGraphNode`] (POWL __2.0__'s generalization of the
/// exclusive-choice and loop operators into one unified cyclic-graph construct -- see
/// [`ChoiceGraphNode`]'s docs for the exact POWL 2.0 correspondence, per Kourani, Park & van der
/// Aalst, "Hierarchical Decomposition of Separable Workflow-Nets", Def. 3.6-3.9).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub enum PowlNode {
    /// A non-silent or silent activity leaf, carrying its own [`Freq`] tag.
    Leaf(PowlLeaf),
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
    /// Creates a new non-silent or silent leaf node with the default (exactly-once) frequency.
    pub fn new_leaf(leaf_label: Option<String>) -> Self {
        PowlNode::Leaf(PowlLeaf::new(leaf_label))
    }

    /// Returns this node's [`Freq`] multiplicity tag.
    pub fn freq(&self) -> Freq {
        match self {
            PowlNode::Leaf(leaf) => leaf.freq,
            PowlNode::Operator(op) => op.freq,
            PowlNode::PartialOrder(po) => po.freq,
            PowlNode::ChoiceGraph(cg) => cg.freq,
        }
    }

    /// Sets this node's [`Freq`] multiplicity tag in place.
    pub fn set_freq(&mut self, freq: Freq) {
        match self {
            PowlNode::Leaf(leaf) => leaf.freq = freq,
            PowlNode::Operator(op) => op.freq = freq,
            PowlNode::PartialOrder(po) => po.freq = freq,
            PowlNode::ChoiceGraph(cg) => cg.freq = freq,
        }
    }

    /// Unfolds this node and its descendants into places, transitions, and arcs, adding them to
    /// the given [`PetriNet`]. Mirrors
    /// [`process_tree::Node::add_to_petri_net`](crate::core::process_models::case_centric::process_tree::Node::add_to_petri_net).
    ///
    /// Does NOT itself apply skip/repeat semantics for a non-default [`Freq`] -- callers that
    /// need that must call [`Powl::expand_frequency_tags`] first, exactly as the reference's
    /// `apply()` calls `expand_frequency_tags` before compiling to a Petri net
    /// (`~/POWL/powl/conversion/variants/to_petri_net.py:201-205`). [`Powl::to_petri_net`] does
    /// this automatically.
    pub fn add_to_petri_net(
        &self,
        net: &mut PetriNet,
        in_place: Option<PlaceID>,
        out_place: Option<PlaceID>,
    ) -> (PlaceID, PlaceID) {
        match self {
            PowlNode::Leaf(leaf) => leaf.leaf.add_to_petri_net(net, in_place, out_place),
            PowlNode::Operator(op) => op.add_to_petri_net(net, in_place, out_place),
            PowlNode::PartialOrder(po) => po.add_to_petri_net(net, in_place, out_place),
            PowlNode::ChoiceGraph(cg) => cg.add_to_petri_net(net, in_place, out_place),
        }
    }

    /// Returns all descendant [`PowlLeaf`]s (including `self` if it is a leaf).
    pub fn find_all_leaves(&self) -> Vec<&PowlLeaf> {
        let mut result = Vec::new();
        self.collect_leaves(&mut result);
        result
    }

    fn collect_leaves<'a>(&'a self, out: &mut Vec<&'a PowlLeaf>) {
        match self {
            PowlNode::Leaf(leaf) => out.push(leaf),
            PowlNode::Operator(op) => op.children.iter().for_each(|c| c.collect_leaves(out)),
            PowlNode::PartialOrder(po) => po.children.iter().for_each(|c| c.collect_leaves(out)),
            PowlNode::ChoiceGraph(cg) => cg.children.iter().for_each(|c| c.collect_leaves(out)),
        }
    }
}

/// A block-structured operator over [`PowlNode`] children (the POWL-recursive analogue of
/// [`process_tree::Operator`](crate::core::process_models::case_centric::process_tree::Operator)),
/// carrying its own [`Freq`] tag.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct PowlOperator {
    /// The block-structured operator's [`process_tree::OperatorType`](crate::core::process_models::case_centric::process_tree::OperatorType), reused as-is.
    pub operator_type: crate::core::process_models::case_centric::process_tree::OperatorType,
    /// The children of the operator node.
    pub children: Vec<PowlNode>,
    /// This operator's multiplicity tag.
    pub freq: Freq,
}

impl PowlOperator {
    /// Creates a new, childless operator of the given type with the default frequency tag.
    pub fn new(operator_type: crate::core::process_models::case_centric::process_tree::OperatorType) -> Self {
        Self {
            operator_type,
            children: Vec::new(),
            freq: Freq::EXACTLY_ONE,
        }
    }

    /// Same translation rules as [`process_tree::Operator::add_to_petri_net`](crate::core::process_models::case_centric::process_tree::Operator::add_to_petri_net),
    /// reimplemented over [`PowlNode`] children so a `PartialOrder`/`ChoiceGraph` sub-node can
    /// appear as an operator's child.
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
/// block-structured process tree), carrying its own [`Freq`] tag.
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
    /// This partial order's multiplicity tag.
    pub freq: Freq,
}

impl PartialOrderNode {
    /// Creates a new [`PartialOrderNode`] from children and a (not necessarily transitively
    /// closed) set of order edges; the edges are transitively closed on construction. The
    /// frequency tag defaults to exactly-once; set `.freq` directly to change it.
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
            freq: Freq::EXACTLY_ONE,
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
    /// This choice graph's multiplicity tag.
    pub freq: Freq,
}

impl ChoiceGraphNode {
    /// Creates a new [`ChoiceGraphNode`] from children and a set of edges over
    /// [`ChoiceGraphEndpoint`]s, with the default (exactly-once) frequency tag. Edges are taken
    /// as given -- unlike [`PartialOrderNode::new`], no transitive closure is computed, since a
    /// choice graph is not a partial order.
    pub fn new(
        children: Vec<PowlNode>,
        edges: impl IntoIterator<Item = (ChoiceGraphEndpoint, ChoiceGraphEndpoint)>,
    ) -> Self {
        Self {
            children,
            edges: edges.into_iter().collect(),
            freq: Freq::EXACTLY_ONE,
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

    /// Convenience constructor for a plain exclusive choice among `children` expressed as a
    /// choice graph: every child is both a start node and an end node, with no edges between
    /// children. Mirrors the reference's `xor()` builder
    /// (`~/POWL/powl/objects/tagged_powl/builders.py:30-43`).
    pub fn exclusive_choice(children: Vec<PowlNode>) -> Self {
        use ChoiceGraphEndpoint::{Child, End, Start};
        let n = children.len();
        let edges = (0..n).flat_map(|i| [(Start, Child(i)), (Child(i), End)]);
        Self::new(children, edges)
    }

    /// Convenience constructor for a general two-node do/redo loop: `do` executes first, then
    /// zero or more `do, redo` repetitions. Mirrors the reference's `loop()` builder
    /// (`~/POWL/powl/objects/tagged_powl/builders.py:46-60`); `do` is `Child(0)`, `redo` is
    /// `Child(1)`. Used by [`Powl::expand_frequency_tags`] to materialize a repeatable node's
    /// multiplicity.
    pub fn do_redo_loop(do_part: PowlNode, redo_part: PowlNode) -> Self {
        use ChoiceGraphEndpoint::{Child, End, Start};
        Self::new(
            vec![do_part, redo_part],
            [
                (Start, Child(0)),
                (Child(0), End),
                (Child(0), Child(1)),
                (Child(1), Child(0)),
            ],
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

    /// Returns all descendant [`PowlLeaf`]s of the model.
    pub fn find_all_leaves(&self) -> Vec<&PowlLeaf> {
        self.root.find_all_leaves()
    }

    /// Materializes every node's [`Freq`] tag into real choice-graph structure, returning a new
    /// [`Powl`] model where every node has the default (exactly-once) frequency.
    ///
    /// An independent reimplementation of the reference's `expand_frequency_tags`/
    /// `_wrap_frequency_tags` (`~/POWL/powl/objects/tagged_powl/builders.py:63-113`), verified
    /// against that source this session. For a node `N` with a non-default `Freq` tag, the body
    /// `N` (recursively expanded first, with its own tag reset to exactly-one) is wrapped as:
    ///
    /// - skippable AND repeatable (`min_freq == 0`, `is_repeatable() == true`): a do/redo loop
    ///   whose do-part is silent and whose redo-part is the body -- the graph can skip straight
    ///   through, or loop the body zero or more times.
    /// - repeatable only (`is_repeatable() == true`, not skippable): a do/redo loop whose
    ///   do-part is the body and whose redo-part is silent -- the body occurs one or more times.
    /// - skippable only (`min_freq == 0`, not repeatable): a plain exclusive choice between the
    ///   body and a silent activity -- occurs zero or one times.
    /// - neither: the body is returned unchanged.
    ///
    /// Note the same simplification the reference makes (see [`Freq::is_repeatable`]'s docs): a
    /// finite `max_freq > 1` is expanded identically to an unbounded one, via a graph cycle
    /// rather than a bounded unrolling -- this is the real reference algorithm's behavior, not
    /// an approximation introduced here.
    pub fn expand_frequency_tags(&self) -> Powl {
        Powl::new(expand_node(&self.root))
    }

    /// Translates the model into a workflow net (a [`PetriNet`] with a single initial and final
    /// marking), the same shape
    /// [`process_tree::ProcessTree::to_petri_net`](crate::core::process_models::case_centric::process_tree::ProcessTree::to_petri_net)
    /// produces. Frequency tags are expanded first (see [`Powl::expand_frequency_tags`]),
    /// matching the reference's `apply()`
    /// (`~/POWL/powl/conversion/variants/to_petri_net.py:201-205`), so a skippable or repeatable
    /// node's multiplicity is faithfully represented in the exported net.
    pub fn to_petri_net(&self) -> PetriNet {
        let expanded = self.expand_frequency_tags();
        let mut net = PetriNet::new();
        let (start, end) = expanded.root.add_to_petri_net(&mut net, None, None);
        net.initial_marking = Some(Marking::from([(start, 1)]));
        let mut final_marking = Marking::new();
        final_marking.insert(end, 1);
        net.final_markings = Some(vec![final_marking]);
        net
    }
}

/// Recursively expands one [`PowlNode`]'s frequency tag; the free function backing
/// [`Powl::expand_frequency_tags`] (kept as a standalone recursive helper since it must recurse
/// into every node kind uniformly, independent of which `Powl` it started from).
fn expand_node(node: &PowlNode) -> PowlNode {
    let freq = node.freq();

    // Recursively expand this node's own children first, with its own tag reset to
    // exactly-one -- matches the reference's structure exactly: `body.min_freq = 1;
    // body.max_freq = 1` before `_wrap_frequency_tags(body, model)`.
    let mut body = match node {
        PowlNode::Leaf(leaf) => PowlNode::Leaf(PowlLeaf {
            leaf: Leaf::new(match &leaf.leaf.activity_label {
                crate::core::process_models::case_centric::process_tree::LeafLabel::Activity(a) => {
                    Some(a.clone())
                }
                crate::core::process_models::case_centric::process_tree::LeafLabel::Tau => None,
            }),
            freq: Freq::EXACTLY_ONE,
        }),
        PowlNode::Operator(op) => {
            let mut new_op = PowlOperator::new(op.operator_type_clone());
            new_op.children = op.children.iter().map(expand_node).collect();
            PowlNode::Operator(new_op)
        }
        PowlNode::PartialOrder(po) => {
            let mut new_po =
                PartialOrderNode::new(po.children.iter().map(expand_node).collect(), po.order.iter().copied());
            new_po.freq = Freq::EXACTLY_ONE;
            PowlNode::PartialOrder(new_po)
        }
        PowlNode::ChoiceGraph(cg) => {
            let mut new_cg = ChoiceGraphNode::new(
                cg.children.iter().map(expand_node).collect(),
                cg.edges.iter().copied(),
            );
            new_cg.freq = Freq::EXACTLY_ONE;
            PowlNode::ChoiceGraph(new_cg)
        }
    };
    body.set_freq(Freq::EXACTLY_ONE);

    if freq.is_repeatable() {
        if freq.is_skippable() {
            // Zero or more: loop(silent, body).
            PowlNode::ChoiceGraph(ChoiceGraphNode::do_redo_loop(
                PowlNode::new_leaf(None),
                body,
            ))
        } else {
            // One or more: loop(body, silent).
            PowlNode::ChoiceGraph(ChoiceGraphNode::do_redo_loop(
                body,
                PowlNode::new_leaf(None),
            ))
        }
    } else if freq.is_skippable() {
        // Zero or one: xor(body, silent).
        PowlNode::ChoiceGraph(ChoiceGraphNode::exclusive_choice(vec![
            body,
            PowlNode::new_leaf(None),
        ]))
    } else {
        body
    }
}

impl PowlOperator {
    /// Clones just this operator's [`process_tree::OperatorType`](crate::core::process_models::case_centric::process_tree::OperatorType)
    /// (which does not itself implement `Clone`); a small helper for
    /// [`Powl::expand_frequency_tags`]'s recursion.
    fn operator_type_clone(&self) -> crate::core::process_models::case_centric::process_tree::OperatorType {
        use crate::core::process_models::case_centric::process_tree::OperatorType;
        match self.operator_type {
            OperatorType::Sequence => OperatorType::Sequence,
            OperatorType::ExclusiveChoice => OperatorType::ExclusiveChoice,
            OperatorType::Concurrency => OperatorType::Concurrency,
            OperatorType::Loop => OperatorType::Loop,
        }
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
    fn choice_graph_is_valid_across_cyclic_and_branching_topologies() {
        // Same underlying property (Def. 3.6 validity: every child reachable from ▷ and
        // can reach □) checked over two structurally distinct topologies -- merged from two
        // formerly separate tests that both only asserted `is_valid()` on a well-formed
        // graph, differing solely in surface topology.

        // Cyclic: the POWL 2.0 replacement for a block-structured Loop(a, tau) -- a single
        // child with a genuine cyclic edge back to itself.
        let self_loop = ChoiceGraphNode::self_looping(PowlNode::new_leaf(Some("a".into())));
        assert!(self_loop.is_valid());
        assert!(self_loop
            .edges
            .contains(&(ChoiceGraphEndpoint::Child(0), ChoiceGraphEndpoint::Child(0))));

        // Branching, acyclic: ▷ -> a -> □ and ▷ -> b -> □, a plain exclusive choice
        // expressed as a choice graph (POWL 2.0 generalizes ExclusiveChoice into this same
        // construct). This is the direct positive counterpart of
        // `choice_graph_rejects_unreachable_node` below -- same two-child branching shape,
        // but with both children actually reachable.
        let exclusive_choice = ChoiceGraphNode::exclusive_choice(vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(Some("b".into())),
        ]);
        assert!(exclusive_choice.is_valid());
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

    #[test]
    fn default_freq_is_exactly_one_and_expansion_is_a_no_op() {
        // Every node built through this crate's constructors defaults to Freq::EXACTLY_ONE, so
        // expand_frequency_tags must be a structural no-op for the existing (pre-frequency-tag)
        // test suite above -- backward compatibility, checked for real rather than assumed.
        let leaf = PowlNode::new_leaf(Some("a".into()));
        assert_eq!(leaf.freq(), Freq::EXACTLY_ONE);
        assert!(!leaf.freq().is_skippable());
        assert!(!leaf.freq().is_repeatable());

        let model = Powl::new(leaf);
        let expanded = model.expand_frequency_tags();
        // Still a plain Leaf, not wrapped in a ChoiceGraph -- exactly-one needs no wrapping.
        assert!(matches!(expanded.root, PowlNode::Leaf(_)));
    }

    #[test]
    fn skippable_and_repeatable_leaf_expands_to_a_zero_or_more_choice_graph() {
        let mut leaf = PowlNode::new_leaf(Some("a".into()));
        leaf.set_freq(Freq::new(0, None)); // zero or more
        assert!(leaf.freq().is_skippable());
        assert!(leaf.freq().is_repeatable());

        let expanded = Powl::new(leaf).expand_frequency_tags();
        let PowlNode::ChoiceGraph(cg) = &expanded.root else {
            panic!("expected a ChoiceGraph wrapper for a skippable+repeatable node");
        };
        assert!(cg.is_valid());
        // do/redo loop: 2 children (silent do-part, the real body as redo-part).
        assert_eq!(cg.children.len(), 2);
        assert!(matches!(cg.children[1], PowlNode::Leaf(_)));

        // The expanded model must compile to a real, valid workflow net.
        let net = expanded.to_petri_net();
        assert!(net.initial_marking.is_some());
    }

    #[test]
    fn repeatable_only_leaf_expands_to_a_one_or_more_choice_graph() {
        let mut leaf = PowlNode::new_leaf(Some("a".into()));
        leaf.set_freq(Freq::new(1, None)); // one or more, not skippable
        let expanded = Powl::new(leaf).expand_frequency_tags();
        let PowlNode::ChoiceGraph(cg) = &expanded.root else {
            panic!("expected a ChoiceGraph wrapper for a repeatable-only node");
        };
        assert!(cg.is_valid());
        // do/redo loop: do-part is the real body (not silent), redo-part is silent.
        assert!(matches!(cg.children[0], PowlNode::Leaf(_)));
    }

    #[test]
    fn skippable_only_leaf_expands_to_a_zero_or_one_exclusive_choice() {
        let mut leaf = PowlNode::new_leaf(Some("a".into()));
        leaf.set_freq(Freq::new(0, Some(1))); // zero or one
        let expanded = Powl::new(leaf).expand_frequency_tags();
        let PowlNode::ChoiceGraph(cg) = &expanded.root else {
            panic!("expected a ChoiceGraph wrapper for a skippable-only node");
        };
        assert!(cg.is_valid());
        assert_eq!(cg.children.len(), 2);
        // No cyclic edge -- this is a plain either/or, not a loop.
        assert!(!cg
            .edges
            .contains(&(ChoiceGraphEndpoint::Child(0), ChoiceGraphEndpoint::Child(0))));
    }

    // -- Powl::normalize -----------------------------------------------------------------

    /// Extracts the activity label of a plain (non-silent) leaf, panicking otherwise -- a small
    /// assertion helper for the tests below.
    fn leaf_label(node: &PowlNode) -> &str {
        let PowlNode::Leaf(leaf) = node else {
            panic!("expected a Leaf node, got {node:?}");
        };
        let crate::core::process_models::case_centric::process_tree::LeafLabel::Activity(label) =
            &leaf.leaf.activity_label
        else {
            panic!("expected a non-silent leaf, got {node:?}");
        };
        label.as_str()
    }

    #[test]
    fn normalize_is_a_no_op_on_an_already_minimal_choice_graph() {
        // A plain exclusive choice between two real activities has no redundant silent nodes,
        // no skippable-marking opportunity (no direct Start->End edge), and no sequential cut
        // point (removing either child still leaves the other's path intact) -- normalize must
        // leave it structurally unchanged.
        let cg = ChoiceGraphNode::exclusive_choice(vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(Some("b".into())),
        ]);
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let normalized = model.normalize();

        let PowlNode::ChoiceGraph(result) = &normalized.root else {
            panic!("expected the result to still be a ChoiceGraph");
        };
        assert!(result.is_valid());
        assert_eq!(result.children.len(), 2);
        let mut labels: Vec<&str> = result.children.iter().map(leaf_label).collect();
        labels.sort();
        assert_eq!(labels, vec!["a", "b"]);
        // Neither child was marked skippable -- there was no redundant direct edge to justify it.
        assert!(result.children.iter().all(|c| c.freq() == Freq::EXACTLY_ONE));
    }

    #[test]
    fn normalize_bypasses_a_redundant_silent_node_in_a_choice_graph() {
        // Start -[tau]-> a -> End, plus a genuinely separate Start -> b -> End branch. The tau
        // has exactly one predecessor (Start) and one successor (a), so it must be bypassed,
        // wiring Start directly to a and dropping the tau node entirely.
        use ChoiceGraphEndpoint::{Child, End, Start};
        let children = vec![
            PowlNode::new_leaf(None),          // 0: tau
            PowlNode::new_leaf(Some("a".into())), // 1
            PowlNode::new_leaf(Some("b".into())), // 2
        ];
        let cg = ChoiceGraphNode::new(
            children,
            [
                (Start, Child(0)),
                (Child(0), Child(1)),
                (Child(1), End),
                (Start, Child(2)),
                (Child(2), End),
            ],
        );
        assert_eq!(cg.children.len(), 3, "sanity: 3 nodes (tau, a, b) before normalize");

        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let normalized = model.normalize();

        let PowlNode::ChoiceGraph(result) = &normalized.root else {
            panic!("expected the result to still be a ChoiceGraph");
        };
        assert!(result.is_valid());
        // The tau is gone: only "a" and "b" remain -- a real, verifiable node-count reduction.
        assert_eq!(result.children.len(), 2);
        let mut labels: Vec<&str> = result.children.iter().map(leaf_label).collect();
        labels.sort();
        assert_eq!(labels, vec!["a", "b"]);
    }

    #[test]
    fn normalize_marks_a_redundant_direct_choice_as_skippable() {
        // Start -> a -> End, plus a direct Start -> End edge: "a" can always be bypassed
        // entirely, so the reduction must mark it skippable and drop the now-redundant direct
        // edge -- collapsing the whole graph down to a bare skippable "a" leaf.
        use ChoiceGraphEndpoint::{Child, End, Start};
        let cg = ChoiceGraphNode::new(
            vec![PowlNode::new_leaf(Some("a".into()))],
            [(Start, Child(0)), (Child(0), End), (Start, End)],
        );
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let normalized = model.normalize();

        assert_eq!(leaf_label(&normalized.root), "a");
        assert_eq!(normalized.root.freq(), Freq::new(0, Some(1)));
    }

    #[test]
    fn normalize_collapses_a_skippable_and_repeatable_expansion_round_trip() {
        // expand_frequency_tags(zero-or-more "a") wraps "a" in a do/redo ChoiceGraph; normalize
        // must be its real structural inverse, recovering a bare "a" leaf with the original tag.
        let mut leaf = PowlNode::new_leaf(Some("a".into()));
        leaf.set_freq(Freq::new(0, None));
        let expanded = Powl::new(leaf).expand_frequency_tags();
        assert!(matches!(expanded.root, PowlNode::ChoiceGraph(_)));

        let normalized = expanded.normalize();
        assert_eq!(leaf_label(&normalized.root), "a");
        assert_eq!(normalized.root.freq(), Freq::new(0, None));
    }

    #[test]
    fn normalize_collapses_a_repeatable_only_expansion_round_trip() {
        let mut leaf = PowlNode::new_leaf(Some("a".into()));
        leaf.set_freq(Freq::new(1, None));
        let expanded = Powl::new(leaf).expand_frequency_tags();

        let normalized = expanded.normalize();
        assert_eq!(leaf_label(&normalized.root), "a");
        assert_eq!(normalized.root.freq(), Freq::new(1, None));
    }

    #[test]
    fn normalize_collapses_a_raw_self_looping_choice_graph() {
        // ChoiceGraphNode::self_looping's own genuine Child(0)->Child(0) cyclic edge (not
        // routed through a silent node) must collapse to a bare unbounded-repeatable leaf.
        let cg = ChoiceGraphNode::self_looping(PowlNode::new_leaf(Some("a".into())));
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let normalized = model.normalize();

        assert_eq!(leaf_label(&normalized.root), "a");
        let freq = normalized.root.freq();
        assert_eq!(freq.min_freq, 1);
        assert_eq!(freq.max_freq, None);
    }

    #[test]
    fn normalize_chunks_a_two_stage_choice_graph_into_a_partial_order() {
        // Start -> a -> {c, d} -> End: "a" always happens first (a genuine sequential cut
        // point), then an exclusive choice between "c" and "d" -- two real sequential stages
        // with no cross-cutting cycle.
        use ChoiceGraphEndpoint::{Child, End, Start};
        let children = vec![
            PowlNode::new_leaf(Some("a".into())), // 0
            PowlNode::new_leaf(Some("c".into())), // 1
            PowlNode::new_leaf(Some("d".into())), // 2
        ];
        let cg = ChoiceGraphNode::new(
            children,
            [
                (Start, Child(0)),
                (Child(0), Child(1)),
                (Child(0), Child(2)),
                (Child(1), End),
                (Child(2), End),
            ],
        );
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let normalized = model.normalize();

        let PowlNode::PartialOrder(po) = &normalized.root else {
            panic!("expected a two-stage sequential ChoiceGraph to normalize into a PartialOrder, got {:?}", normalized.root);
        };
        assert_eq!(po.children.len(), 2, "exactly two sequential stages/chunks");
        assert_eq!(leaf_label(&po.children[0]), "a");
        let PowlNode::ChoiceGraph(stage2) = &po.children[1] else {
            panic!("expected the second stage to still be a ChoiceGraph (c XOR d), got {:?}", po.children[1]);
        };
        assert!(stage2.is_valid());
        let mut stage2_labels: Vec<&str> = stage2.children.iter().map(leaf_label).collect();
        stage2_labels.sort();
        assert_eq!(stage2_labels, vec!["c", "d"]);
        assert!(po.order.contains(&(0, 1)));
    }

    #[test]
    fn normalize_abstracts_a_nested_scc_loop_within_a_larger_sequence() {
        // Start -> a -> x <-> y -> b -> End: {x, y} is a genuine 2-node strongly connected
        // component with a clean single entry (x, from a) / single exit (y, to b) boundary,
        // nested inside an otherwise-acyclic 3-stage sequence. Exercises SCC abstraction (via
        // petgraph::algo::tarjan_scc), the nested self-loop collapse inside that SCC, and outer
        // sequence chunking, all together.
        use ChoiceGraphEndpoint::{Child, End, Start};
        let children = vec![
            PowlNode::new_leaf(Some("a".into())), // 0
            PowlNode::new_leaf(Some("x".into())), // 1
            PowlNode::new_leaf(Some("y".into())), // 2
            PowlNode::new_leaf(Some("b".into())), // 3
        ];
        let cg = ChoiceGraphNode::new(
            children,
            [
                (Start, Child(0)),
                (Child(0), Child(1)),
                (Child(1), Child(2)),
                (Child(2), Child(1)), // the back-edge closing the {x, y} cycle
                (Child(2), Child(3)),
                (Child(3), End),
            ],
        );
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let normalized = model.normalize();

        let PowlNode::PartialOrder(po) = &normalized.root else {
            panic!("expected a PartialOrder of 3 sequential stages, got {:?}", normalized.root);
        };
        assert_eq!(po.children.len(), 3);
        assert_eq!(leaf_label(&po.children[0]), "a");
        assert_eq!(leaf_label(&po.children[2]), "b");

        let PowlNode::PartialOrder(inner) = &po.children[1] else {
            panic!("expected the middle stage to be the abstracted {{x, y}} loop, got {:?}", po.children[1]);
        };
        assert_eq!(inner.children.len(), 2);
        let mut inner_labels: Vec<&str> = inner.children.iter().map(leaf_label).collect();
        inner_labels.sort();
        assert_eq!(inner_labels, vec!["x", "y"]);
        // The {x, y} cycle was correctly recognized as "repeats one or more times", not skippable.
        assert_eq!(inner.freq.min_freq, 1);
        assert_eq!(inner.freq.max_freq, None);

        assert!(po.order.contains(&(0, 1)));
        assert!(po.order.contains(&(1, 2)));
    }

    #[test]
    fn normalize_partial_order_bypasses_a_silent_child_and_reindexes() {
        // a -> tau -> b: tau has exactly one direct predecessor and one direct successor (using
        // the DIRECT, non-transitively-implied edges -- not po.order's transitive closure, which
        // would otherwise make "a" look like a second predecessor of "b").
        let children = vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(None),
            PowlNode::new_leaf(Some("b".into())),
        ];
        let po = PartialOrderNode::new(children, [(0, 1), (1, 2)]);
        let model = Powl::new(PowlNode::PartialOrder(po));
        let normalized = model.normalize();

        let PowlNode::PartialOrder(result) = &normalized.root else {
            panic!("expected the result to still be a PartialOrder, got {:?}", normalized.root);
        };
        assert_eq!(result.children.len(), 2, "the silent bypass node must be gone");
        assert_eq!(leaf_label(&result.children[0]), "a");
        assert_eq!(leaf_label(&result.children[1]), "b");
        assert!(result.order.contains(&(0, 1)));
    }

    #[test]
    fn normalize_partial_order_flattens_to_a_single_child() {
        // a -> tau (tau has no successor at all -- trivially <=1 -- so it is bypassed away),
        // leaving a single child: the whole PartialOrder wrapper must flatten to a bare "a" leaf.
        let children = vec![PowlNode::new_leaf(Some("a".into())), PowlNode::new_leaf(None)];
        let po = PartialOrderNode::new(children, [(0, 1)]);
        let model = Powl::new(PowlNode::PartialOrder(po));
        let normalized = model.normalize();

        assert_eq!(leaf_label(&normalized.root), "a");
        assert_eq!(normalized.root.freq(), Freq::EXACTLY_ONE);
    }

    #[test]
    fn normalize_operator_recurses_into_its_children() {
        // Sequence(a, ChoiceGraph-with-a-redundant-tau): the Operator itself must survive
        // unchanged in kind, but its second child must come back reduced.
        use ChoiceGraphEndpoint::{Child, End, Start};
        let inner_children = vec![PowlNode::new_leaf(None), PowlNode::new_leaf(Some("b".into()))];
        let inner_cg = ChoiceGraphNode::new(
            inner_children,
            [(Start, Child(0)), (Child(0), Child(1)), (Child(1), End)],
        );

        let mut op = PowlOperator::new(OperatorType::Sequence);
        op.children.push(PowlNode::new_leaf(Some("a".into())));
        op.children.push(PowlNode::ChoiceGraph(inner_cg));
        let model = Powl::new(PowlNode::Operator(op));
        let normalized = model.normalize();

        let PowlNode::Operator(result) = &normalized.root else {
            panic!("expected an Operator node, got {:?}", normalized.root);
        };
        assert!(matches!(result.operator_type, OperatorType::Sequence));
        assert_eq!(result.children.len(), 2);
        assert_eq!(leaf_label(&result.children[0]), "a");
        // The nested ChoiceGraph's redundant tau bypass must have collapsed it to a bare "b" leaf.
        assert_eq!(leaf_label(&result.children[1]), "b");
    }
}
