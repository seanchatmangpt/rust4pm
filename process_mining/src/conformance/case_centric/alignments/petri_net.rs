//! Optimal alignment search: the synchronous product net as a [`SearchProblem`], solved with the
//! generic [`crate::utils::dijkstra_search`] Dijkstra.
use std::hash::Hasher;

use hashbrown::HashTable;
use rustc_hash::FxHasher;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    conformance::alignments::{
        sync_prod_net::{SyncProdNetConstructionError, SyncProdNetTransition, SyncProductNet},
        AlignmentMove, AlignmentResult,
    },
    utils::dijkstra_search::{
        search_bidirectional, NodeID, ReversibleSearchProblem, SearchError, SearchLimits,
        SearchProblem, SearchResult, SearchState,
    },
};

/// Alignment Error
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum AlignmentError {
    /// Search Error
    SearchError(SearchError),
    /// Constructing the synchronous product net failed
    SyncProdNetConstructionFailed(SyncProdNetConstructionError),
}

impl std::fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for AlignmentError {}

impl From<SyncProdNetConstructionError> for AlignmentError {
    fn from(value: SyncProdNetConstructionError) -> Self {
        Self::SyncProdNetConstructionFailed(value)
    }
}

impl From<SearchError> for AlignmentError {
    fn from(value: SearchError) -> Self {
        Self::SearchError(value)
    }
}

/// Type representing the count of tokens (e.g., in a marking)
///
/// May be changed to larger types if Petri nets with more than 255 tokens in a place should be supported.
pub type TokenCount = u8;

/// Type representing trace position
pub type TracePos = u16;

/// An edge/step of the Petri-net search: the transition fired, and whether it is a log move.
///
/// The log-move-flag allows for log-before-model-moves pruning in [`PetriNetAlignment::expand`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct PetriNetStep {
    /// Index of the fired transition in the sync. prod. net
    transition: u32,
    /// Whether the fired transition is a log move
    log_move: bool,
}

/// Reusable state storage for a Petri-net alignment search
#[derive(Debug, Default)]
pub(crate) struct PetriNetAlignmentSpace {
    /// Number of model places
    num_places: usize,
    /// Flat storage for model markings, indexed using [`NodeID`]
    markings: Vec<TokenCount>,
    /// Trace position per node, indexed using [`NodeID`]
    trace_pos: Vec<TracePos>,
    /// Index of visited states, mapping a `(marking, trace_pos)` tuple to a [`NodeID`]
    seen: HashTable<NodeID>,
    /// Current marking (re-used to reduce allocations)
    current: Vec<TokenCount>,
    /// Next marking, reached by firing a transition (re-used to reduce allocations)
    next: Vec<TokenCount>,
}

impl PetriNetAlignmentSpace {
    fn reset(&mut self, net: &SyncProductNet<'_>, reverse: bool) {
        self.num_places = net.model.num_places;
        self.markings.clear();
        self.trace_pos.clear();
        self.seen.clear();
        self.current.resize(net.model.num_places, 0);
        self.next.resize(net.model.num_places, 0);
        // A backward search starts where a forward one ends
        let (marking, trace_pos) = if reverse {
            (&net.model.final_marking, net.trace_length)
        } else {
            (&net.model.initial_marking, 0)
        };
        self.markings
            .extend_from_slice(&marking[..net.model.num_places]);
        self.trace_pos.push(trace_pos);
        self.add_seen(0, hash_state(marking, trace_pos));
    }

    /// Record a node under the given hash of its state
    #[inline]
    fn add_seen(&mut self, node: NodeID, hash: u64) {
        let markings = &self.markings;
        let trace_pos = &self.trace_pos;
        let num_places = self.num_places;
        self.seen.insert_unique(hash, node, |other| {
            let off = *other as usize * num_places;
            hash_state(&markings[off..off + num_places], trace_pos[*other as usize])
        });
    }

    #[inline]
    fn find_seen(&self, marking: &[TokenCount], trace_position: TracePos) -> Option<NodeID> {
        self.find_seen_hashed(hash_state(marking, trace_position), marking, trace_position)
    }

    #[inline]
    fn find_seen_hashed(
        &self,
        hash: u64,
        marking: &[TokenCount],
        trace_position: TracePos,
    ) -> Option<NodeID> {
        let num_places = self.num_places;
        self.seen
            .find(hash, |node| {
                let off = *node as usize * num_places;
                &self.markings[off..off + num_places] == marking
                    && self.trace_pos[*node as usize] == trace_position
            })
            .copied()
    }
}

/// Bytes one queued state costs: its search node, its `(marking, trace position)`, and its slot
/// in the `seen` index. Both directions store the same per state.
pub(crate) const fn bytes_per_state(num_places: usize) -> usize {
    SearchState::<PetriNetStep>::bytes_per_node()
        + size_of::<TracePos>()
        // A `seen` slot is a NodeID, plus hashbrown's control byte and spare capacity
        + 2 * size_of::<NodeID>()
        + num_places
}

/// Alignment as a [`SearchProblem`]: a state is a `(model marking, trace position)`, an edge fires
/// a sync. prod. net transition
#[derive(Debug)]
struct PetriNetAlignment<'a> {
    net: &'a SyncProductNet<'a>,
    space: &'a mut PetriNetAlignmentSpace,
    /// Search backwards, from the final marking, firing every transition in reverse
    reverse: bool,
}

impl SearchProblem for PetriNetAlignment<'_> {
    type Step = PetriNetStep;
    type Cost = u32;

    fn initial(&mut self) -> NodeID {
        self.space.reset(self.net, self.reverse);
        0
    }

    fn max_edge_cost(&self) -> u32 {
        self.net.max_edge_cost
    }

    /// Whether the whole trace is consumed and the target marking reached
    #[inline]
    fn is_goal(&self, node: NodeID) -> bool {
        let np = self.net.model.num_places;
        let off = node as usize * np;
        let (marking, trace_pos) = if self.reverse {
            (&self.net.model.initial_marking, 0)
        } else {
            (&self.net.model.final_marking, self.net.trace_length)
        };
        self.space.trace_pos[node as usize] == trace_pos
            && self.space.markings[off..off + np] == marking[..np]
    }

    #[inline]
    fn expand<F: FnMut(NodeID, bool, u32, PetriNetStep)>(
        &mut self,
        node: NodeID,
        via: Option<PetriNetStep>,
        mut emit: F,
    ) {
        let reverse = self.reverse;
        let net = self.net;
        let space = &mut *self.space;
        let np = space.num_places;
        let off = node as usize * np;
        let trace_pos = space.trace_pos[node as usize];
        let last_move_was_log = via.is_some_and(|s| s.log_move);
        space
            .current
            .copy_from_slice(&space.markings[off..off + np]);

        // Log/sync moves for the current event, then model moves (fixed ordering prunes states).
        // The two commute at equal cost, so keeping one order per pair loses no shortest path.
        // Going backwards, the event to consume is the one before the current position.
        let event = if reverse {
            trace_pos.checked_sub(1)
        } else {
            Some(trace_pos)
        };
        let log_or_sync = event
            .and_then(|event| net.transitions_by_trace_pos.get(event as usize))
            .map(|v| v.as_slice())
            .unwrap_or_default();
        // After a log move, model moves are pruned, so the range collapses to empty.
        let model_end = if last_move_was_log {
            0
        } else {
            net.num_model_trans()
        };

        for trans_idx in log_or_sync.iter().copied().chain(0..model_end) {
            let trans = net.transition(trans_idx);
            if !is_enabled(&space.current, trans, reverse) {
                continue;
            }
            if fire_transition(&space.current, &mut space.next, trans, reverse).is_none() {
                continue;
            }
            let is_model_move = matches!(trans.move_type, AlignmentMove::ModelMove { .. });
            let new_trace_pos = if is_model_move {
                trace_pos
            } else if reverse {
                trace_pos - 1
            } else {
                trace_pos + 1
            };
            let step = PetriNetStep {
                transition: trans_idx as u32,
                log_move: matches!(trans.move_type, AlignmentMove::LogMove { .. }),
            };
            let cost = trans.cost;
            // One hash for both the lookup and a possible insert
            let hash = hash_state(&space.next, new_trace_pos);
            match space.find_seen_hashed(hash, &space.next, new_trace_pos) {
                Some(existing) => emit(existing, false, cost, step),
                None => {
                    let new_id = space.trace_pos.len() as NodeID;
                    space.markings.extend_from_slice(&space.next);
                    space.trace_pos.push(new_trace_pos);
                    space.add_seen(new_id, hash);
                    emit(new_id, true, cost, step);
                }
            }
        }
    }
}

impl ReversibleSearchProblem for PetriNetAlignment<'_> {
    #[inline]
    fn find_in(&self, node: NodeID, other: &Self) -> Option<NodeID> {
        let np = self.net.model.num_places;
        let off = node as usize * np;
        other.space.find_seen(
            &self.space.markings[off..off + np],
            self.space.trace_pos[node as usize],
        )
    }
}

/// Search spaces reused across traces, one per search direction
#[derive(Debug, Default)]
pub(crate) struct AlignmentContext {
    forward: (PetriNetAlignmentSpace, SearchState<PetriNetStep>),
    backward: (PetriNetAlignmentSpace, SearchState<PetriNetStep>),
}

/// Compute an optimal alignment, searching from the initial and the final marking at once
pub(crate) fn align(
    net: &SyncProductNet<'_>,
    ctx: &mut AlignmentContext,
    limits: SearchLimits,
) -> Result<AlignmentResult, AlignmentError> {
    let mut forward = PetriNetAlignment {
        net,
        space: &mut ctx.forward.0,
        reverse: false,
    };
    let mut backward = PetriNetAlignment {
        net,
        space: &mut ctx.backward.0,
        reverse: true,
    };
    let res = search_bidirectional(
        &mut forward,
        &mut backward,
        &mut ctx.forward.1,
        &mut ctx.backward.1,
        limits,
    )?;
    Ok(result_from(net, res))
}

fn result_from(net: &SyncProductNet<'_>, res: SearchResult<PetriNetStep>) -> AlignmentResult {
    AlignmentResult {
        moves: res
            .path
            .iter()
            .map(|s| net.transition(s.transition as usize).move_type.clone())
            .collect(),
        cost: res.cost,
        states_visited: res.states_visited,
    }
}

/// Tests whether the given transition is enabled in the given marking
#[inline]
fn is_enabled(marking: &[TokenCount], trans: &SyncProdNetTransition, reverse: bool) -> bool {
    let consumed = if reverse {
        &trans.outputs
    } else {
        &trans.inputs
    };
    consumed
        .iter()
        .all(|(place, weight)| &marking[*place] >= weight)
}

/// Fire the given transition, transforming the current marking into the `reached` marking.
/// When `reverse`, inputs and outputs swap roles.
///
/// Returns `None` if the reached marking would exceed `TokenCount::MAX` tokens.
/// In this case, the reached marking is considered out-of-bounds and should be pruned.
#[inline]
#[must_use]
fn fire_transition(
    current: &[TokenCount],
    reached: &mut [TokenCount],
    trans: &SyncProdNetTransition,
    reverse: bool,
) -> Option<()> {
    let (consumed, produced) = if reverse {
        (&trans.outputs, &trans.inputs)
    } else {
        (&trans.inputs, &trans.outputs)
    };
    reached.copy_from_slice(current);
    for (place, weight) in consumed {
        reached[*place] -= weight;
    }
    for (place, weight) in produced {
        reached[*place] = reached[*place].checked_add(*weight)?;
    }
    Some(())
}

/// Hash a given state (combination of marking and trace position)
#[inline]
fn hash_state(marking: &[TokenCount], trace_pos: TracePos) -> u64 {
    let mut h = FxHasher::default();
    h.write(marking);
    h.write_u16(trace_pos);
    h.finish()
}
