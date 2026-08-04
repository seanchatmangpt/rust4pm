//! Synchronous product net used for efficiently computing alignments

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    conformance::alignments::{
        cost::CostFunction,
        petri_net::{TokenCount, TracePos},
        AlignmentMove,
    },
    core::process_models::petri_net::{ArcType, PlaceID, TransitionID},
    PetriNet,
};

/// A transition in the synchronous product net
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SyncProdNetTransition {
    /// The move this transition represents (model-transition / trace-event indices)
    pub(crate) move_type: AlignmentMove,
    /// The pre-computed cost of firing this transition
    pub(crate) cost: u32,
    /// Incoming places (`place_index`, weight), i.e., which token to consume
    pub(crate) inputs: Vec<(usize, TokenCount)>,
    /// Outgoing places (`place_index`, weight), i.e., which token to produce
    pub(crate) outputs: Vec<(usize, TokenCount)>,
}

/// The trace-independent half of the synchronous product, built once per Petri net
#[derive(Debug, PartialEq)]
pub(crate) struct ModelNet {
    /// Number of model places
    pub(crate) num_places: usize,
    /// The model transitions (they keep these indices in the synchronous product)
    pub(crate) transitions: Vec<SyncProdNetTransition>,
    /// Model transitions per activity label, i.e. those a trace event can synchronize with
    by_label: HashMap<String, Vec<usize>>,
    /// Initial marking (tokens per place)
    pub(crate) initial_marking: Vec<TokenCount>,
    /// Final marking (tokens per place)
    pub(crate) final_marking: Vec<TokenCount>,
    /// Largest cost over the model transitions
    max_cost: u32,
}

/// The synchronous product of a Petri net and a trace.
///
/// Only model places exist; the trace position is tracked separately in the search.
/// Transitions are indexed as `[model moves .., log and sync moves ..]`.
#[derive(Debug, PartialEq)]
pub(crate) struct SyncProductNet<'a> {
    /// The shared, trace-independent half
    pub(crate) model: &'a ModelNet,
    /// Length of the trace
    pub(crate) trace_length: TracePos,
    /// The log and sync moves
    trace_transitions: Vec<SyncProdNetTransition>,
    /// Log/sync transition indices per trace position, i.e., `transitions_by_trace_pos[i]`
    /// holds those for event `i`
    pub(crate) transitions_by_trace_pos: Vec<Vec<usize>>,
    /// Largest cost over all transitions (precomputed for the search's bucket sizing)
    pub(crate) max_edge_cost: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
/// Error when constructing the sync product net
pub enum SyncProdNetConstructionError {
    /// A unknown place id was referenced in a marking
    InvalidPlaceInMarking(PlaceID),
    /// No final marking found
    NoFinalMarking,
    /// No initial marking found
    NoInitialMarking,
    /// An arc weight, or the sum over parallel arcs, exceeds [`TokenCount::MAX`]
    ///
    /// [`TokenCount::MAX`]: super::petri_net::TokenCount
    ArcWeightTooLarge(u32),
    /// A marking puts more than `TokenCount::MAX` tokens in a place
    MarkingTooLarge(PlaceID, u64),
}
impl std::fmt::Display for SyncProdNetConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SyncProdNetConstructionError {}

impl ModelNet {
    /// Build the trace-independent half of the synchronous product from a Petri net
    pub(crate) fn build(
        net: &PetriNet,
        cost_fn: &CostFunction,
    ) -> Result<Self, SyncProdNetConstructionError> {
        let place_index: HashMap<&Uuid, usize> = net
            .places
            .iter()
            .enumerate()
            .map(|(i, p)| (p.0, i))
            .collect();
        let trans_index: HashMap<&Uuid, usize> = net
            .transitions
            .iter()
            .enumerate()
            .map(|(i, t)| (t.0, i))
            .collect();

        // Parallel arcs collapse into one entry per place, summed as `u32`
        let mut inputs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); net.transitions.len()];
        let mut outputs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); net.transitions.len()];
        fn add_weight(arcs: &mut Vec<(usize, u32)>, place: usize, weight: u32) {
            match arcs.iter_mut().find(|(p, _)| *p == place) {
                Some((_, total)) => *total = total.saturating_add(weight),
                None => arcs.push((place, weight)),
            }
        }
        for arc in &net.arcs {
            match arc.from_to {
                ArcType::PlaceTransition(place, trans) => {
                    if let (Some(place), Some(trans)) =
                        (place_index.get(&place), trans_index.get(&trans))
                    {
                        add_weight(&mut inputs[*trans], *place, arc.weight);
                    }
                }
                ArcType::TransitionPlace(trans, place) => {
                    if let (Some(place), Some(trans)) =
                        (place_index.get(&place), trans_index.get(&trans))
                    {
                        add_weight(&mut outputs[*trans], *place, arc.weight);
                    }
                }
            }
        }
        // A place holds at most `TokenCount::MAX` tokens, so reject (rather than truncate) larger ones
        fn to_token_count(
            arcs: Vec<(usize, u32)>,
        ) -> Result<Vec<(usize, TokenCount)>, SyncProdNetConstructionError> {
            arcs.into_iter()
                .map(|(place, weight)| {
                    let weight = TokenCount::try_from(weight)
                        .map_err(|_| SyncProdNetConstructionError::ArcWeightTooLarge(weight))?;
                    Ok((place, weight))
                })
                .collect()
        }

        let mut by_label: HashMap<String, Vec<usize>> = HashMap::new();
        let mut transitions = Vec::with_capacity(net.transitions.len());
        for ((trans_id, trans), (inputs, outputs)) in
            net.transitions.iter().zip(inputs.into_iter().zip(outputs))
        {
            let index = transitions.len();
            if let Some(label) = &trans.label {
                by_label.entry(label.clone()).or_default().push(index);
            }
            transitions.push(SyncProdNetTransition {
                move_type: AlignmentMove::ModelMove {
                    transition: TransitionID(*trans_id),
                },
                cost: if trans.label.is_none() {
                    cost_fn.silent_move_cost
                } else {
                    cost_fn.model_move_cost
                },
                inputs: to_token_count(inputs)?,
                outputs: to_token_count(outputs)?,
            });
        }
        // Get marking as vec of token counts (based on place indices)
        let marking_of = |marking: &HashMap<PlaceID, u64>| {
            let mut tokens: Vec<TokenCount> = vec![0; net.places.len()];
            for (place_id, count) in marking {
                let index = place_index.get(&place_id.0).ok_or(
                    SyncProdNetConstructionError::InvalidPlaceInMarking(*place_id),
                )?;
                tokens[*index] = TokenCount::try_from(*count).map_err(|_| {
                    SyncProdNetConstructionError::MarkingTooLarge(*place_id, *count)
                })?;
            }
            Ok(tokens)
        };
        let initial_marking = marking_of(
            net.initial_marking
                .as_ref()
                .ok_or(SyncProdNetConstructionError::NoInitialMarking)?,
        )?;
        let final_marking = marking_of(
            net.final_markings
                .as_ref()
                .and_then(|f| f.first())
                .ok_or(SyncProdNetConstructionError::NoFinalMarking)?,
        )?;

        let max_cost = transitions.iter().map(|t| t.cost).max().unwrap_or(1);
        Ok(Self {
            num_places: net.places.len(),
            transitions,
            by_label,
            initial_marking,
            final_marking,
            max_cost,
        })
    }
}

impl<'a> SyncProductNet<'a> {
    /// Add the log and sync moves for `trace` to the shared model half
    pub(crate) fn construct(model: &'a ModelNet, trace: &[&str], cost_fn: &CostFunction) -> Self {
        let num_model_trans = model.transitions.len();
        let mut trace_transitions = Vec::with_capacity(trace.len() * 2);
        let mut transitions_by_trace_pos = vec![Vec::new(); trace.len()];
        // There are no log places: the search tracks the trace position, so a log move leaves
        // the marking alone and a sync move changes it like its model transition does.
        for (index, activity) in trace.iter().enumerate() {
            transitions_by_trace_pos[index].push(num_model_trans + trace_transitions.len());
            trace_transitions.push(SyncProdNetTransition {
                move_type: AlignmentMove::LogMove {
                    trace_event_index: index,
                },
                cost: cost_fn.log_move_cost,
                inputs: vec![],
                outputs: vec![],
            });
            for model_index in model.by_label.get(*activity).into_iter().flatten() {
                let model_trans = &model.transitions[*model_index];
                let AlignmentMove::ModelMove { transition } = model_trans.move_type else {
                    unreachable!("the model half holds model moves only")
                };
                transitions_by_trace_pos[index].push(num_model_trans + trace_transitions.len());
                trace_transitions.push(SyncProdNetTransition {
                    move_type: AlignmentMove::SyncMove {
                        transition,
                        trace_event_index: index,
                    },
                    cost: cost_fn.sync_move_cost,
                    inputs: model_trans.inputs.clone(),
                    outputs: model_trans.outputs.clone(),
                });
            }
        }
        let max_edge_cost = trace_transitions
            .iter()
            .map(|t| t.cost)
            .max()
            .unwrap_or(0)
            .max(model.max_cost);
        Self {
            model,
            trace_length: trace.len() as TracePos,
            trace_transitions,
            transitions_by_trace_pos,
            max_edge_cost,
        }
    }

    /// The transition an index addresses, whether it belongs to the model or to the trace
    #[inline]
    pub(crate) fn transition(&self, index: usize) -> &SyncProdNetTransition {
        match index.checked_sub(self.num_model_trans()) {
            Some(trace_index) => &self.trace_transitions[trace_index],
            None => &self.model.transitions[index],
        }
    }

    /// Number of model transitions; they occupy the indices `0..n`
    #[inline]
    pub(crate) fn num_model_trans(&self) -> usize {
        self.model.transitions.len()
    }
}
