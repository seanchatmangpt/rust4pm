//! A deliberately naive reference for [`super`], to check its answers against.
//!
//! It shares no code with the search: it works straight off the [`PetriNet`], keeps a marking as
//! one count per place, and finds optimal alignments by plain Dijkstra plus an exhaustive
//! depth-first walk. It is far too slow for anything but the tiny nets the tests generate, which
//! is the point: it is simple enough to be obviously right.
//!
//! It assumes [`CostFunction::standard`], i.e. sync and silent moves are free and log and model
//! moves cost `1`.
//!
//! [`CostFunction::standard`]: crate::conformance::alignments::cost::CostFunction::standard
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
};

use uuid::Uuid;

use crate::{
    conformance::alignments::AlignmentMove,
    core::process_models::petri_net::{ArcType, PlaceID, TransitionID},
    PetriNet,
};

/// Tokens a single place may hold before a case is treated as out of the reference's reach
const MAX_TOKENS: u32 = 6;

/// Moves a path may hold before the case is treated as out of the reference's reach
const MAX_DEPTH: usize = 40;

/// A marking as one token count per place, in the reference's place order
type Marking = Vec<u32>;

/// A state of the search: what the model holds, and how much of the trace is consumed
type State = (Marking, usize);

/// A transition, with its arcs resolved to place positions
struct RefTransition {
    id: Uuid,
    label: Option<String>,
    inputs: Vec<(usize, u32)>,
    outputs: Vec<(usize, u32)>,
}

impl RefTransition {
    /// A visible transition firing without a matching event costs `1`, a silent one nothing
    fn model_move_cost(&self) -> u32 {
        u32::from(self.label.is_some())
    }
}

/// The reference's own view of a Petri net
pub(super) struct Reference {
    transitions: Vec<RefTransition>,
    initial: Marking,
    goal: Marking,
}

/// What the reference could not answer, so the case has to be skipped rather than compared
#[derive(Debug, PartialEq)]
pub(super) enum Unanswerable {
    /// A marking grew past [`MAX_TOKENS`], which the reference refuses to follow
    Unbounded,
    /// The walk grew past the budget it was given
    TooLarge,
}

impl Reference {
    /// Read a net into the reference's own form, in a fixed place and transition order
    pub(super) fn build(net: &PetriNet) -> Self {
        let mut places: Vec<Uuid> = net.places.keys().copied().collect();
        places.sort_unstable();
        let position: HashMap<Uuid, usize> = places
            .iter()
            .enumerate()
            .map(|(index, place)| (*place, index))
            .collect();

        let mut ids: Vec<Uuid> = net.transitions.keys().copied().collect();
        ids.sort_unstable();
        let mut transitions: Vec<RefTransition> = ids
            .iter()
            .map(|id| RefTransition {
                id: *id,
                label: net.transitions[id].label.clone(),
                inputs: Vec::new(),
                outputs: Vec::new(),
            })
            .collect();
        let index_of: HashMap<Uuid, usize> = ids
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();
        // Parallel arcs between the same pair add up, and zero-weight ones move nothing
        for arc in &net.arcs {
            if arc.weight == 0 {
                continue;
            }
            let (place, transition, into_transition) = match arc.from_to {
                ArcType::PlaceTransition(place, transition) => (place, transition, true),
                ArcType::TransitionPlace(transition, place) => (place, transition, false),
            };
            let trans = &mut transitions[index_of[&transition]];
            let arcs = if into_transition {
                &mut trans.inputs
            } else {
                &mut trans.outputs
            };
            let place = position[&place];
            match arcs.iter_mut().find(|(at, _)| *at == place) {
                Some((_, weight)) => *weight += arc.weight,
                None => arcs.push((place, arc.weight)),
            }
        }

        let marking_of = |marking: &HashMap<PlaceID, u64>| {
            let mut counts = vec![0; places.len()];
            for (place, tokens) in marking {
                counts[position[&place.0]] = *tokens as u32;
            }
            counts
        };
        Self {
            transitions,
            initial: marking_of(net.initial_marking.as_ref().expect("test nets have one")),
            goal: marking_of(
                net.final_markings
                    .as_ref()
                    .expect("test nets have one")
                    .first()
                    .expect("test nets have one"),
            ),
        }
    }

    /// Every move leaving `state`, as `(cost, next state, the move it stands for)`
    fn successors(
        &self,
        state: &State,
        trace: &[&str],
    ) -> Result<Vec<(u32, State, AlignmentMove)>, Unanswerable> {
        let (marking, position) = state;
        let mut out = Vec::new();
        for transition in &self.transitions {
            if !transition
                .inputs
                .iter()
                .all(|(place, weight)| marking[*place] >= *weight)
            {
                continue;
            }
            let mut fired = marking.clone();
            for (place, weight) in &transition.inputs {
                fired[*place] -= weight;
            }
            for (place, weight) in &transition.outputs {
                fired[*place] += weight;
                if fired[*place] > MAX_TOKENS {
                    return Err(Unanswerable::Unbounded);
                }
            }
            out.push((
                transition.model_move_cost(),
                (fired.clone(), *position),
                AlignmentMove::ModelMove {
                    transition: TransitionID(transition.id),
                },
            ));
            // A synchronous move needs the transition's label to be the event about to be read
            if transition.label.as_deref() == trace.get(*position).copied() {
                out.push((
                    0,
                    (fired, position + 1),
                    AlignmentMove::SyncMove {
                        transition: TransitionID(transition.id),
                        trace_event_index: *position,
                    },
                ));
            }
        }
        if *position < trace.len() {
            out.push((
                1,
                (marking.clone(), position + 1),
                AlignmentMove::LogMove {
                    trace_event_index: *position,
                },
            ));
        }
        Ok(out)
    }

    fn is_goal(&self, (marking, position): &State, trace: &[&str]) -> bool {
        *position == trace.len() && *marking == self.goal
    }

    /// Cheapest alignment cost, or `None` when the final marking cannot be reached
    fn optimal_cost(&self, trace: &[&str], budget: usize) -> Result<Option<u32>, Unanswerable> {
        let start = (self.initial.clone(), 0);
        let mut distance: HashMap<State, u32> = HashMap::from([(start.clone(), 0)]);
        let mut queue = BinaryHeap::from([Reverse((0u32, start))]);
        while let Some(Reverse((cost, state))) = queue.pop() {
            if cost > distance[&state] {
                continue;
            }
            if self.is_goal(&state, trace) {
                return Ok(Some(cost));
            }
            if distance.len() > budget {
                return Err(Unanswerable::TooLarge);
            }
            for (step, next, _) in self.successors(&state, trace)? {
                let reached = cost + step;
                if distance.get(&next).is_none_or(|known| reached < *known) {
                    distance.insert(next.clone(), reached);
                    queue.push(Reverse((reached, next)));
                }
            }
        }
        Ok(None)
    }

    /// Every optimal alignment, as a simple path: no state is visited twice.
    ///
    /// `None` when the final marking cannot be reached at all.
    pub(super) fn all_optimal(
        &self,
        trace: &[&str],
        budget: usize,
    ) -> Result<Option<Vec<Vec<AlignmentMove>>>, Unanswerable> {
        let Some(optimal) = self.optimal_cost(trace, budget)? else {
            return Ok(None);
        };
        let start = (self.initial.clone(), 0);
        let mut found = Vec::new();
        let mut on_path = HashSet::from([start.clone()]);
        let mut moves = Vec::new();
        // Counts every state entered, not just the alignments found: a net full of free silent
        // loops has vastly more partial paths than optimal ones, and the walk visits them all
        let mut steps = 0;
        self.walk(
            &start,
            trace,
            optimal,
            0,
            &mut on_path,
            &mut moves,
            &mut found,
            &mut steps,
            budget,
        )?;
        Ok(Some(found))
    }

    /// Depth-first walk over every simple path that still fits within `optimal`
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        state: &State,
        trace: &[&str],
        optimal: u32,
        spent: u32,
        on_path: &mut HashSet<State>,
        moves: &mut Vec<AlignmentMove>,
        found: &mut Vec<Vec<AlignmentMove>>,
        steps: &mut usize,
        budget: usize,
    ) -> Result<(), Unanswerable> {
        *steps += 1;
        // The walk recurses, so a long path would overflow the stack before the budget bites
        if *steps > budget || found.len() > budget || moves.len() > MAX_DEPTH {
            return Err(Unanswerable::TooLarge);
        }
        // Reaching the goal ends the path: coming back to it would visit it twice
        if self.is_goal(state, trace) {
            if spent == optimal {
                found.push(moves.clone());
            }
            return Ok(());
        }
        for (cost, next, step) in self.successors(state, trace)? {
            if spent + cost > optimal || on_path.contains(&next) {
                continue;
            }
            on_path.insert(next.clone());
            moves.push(step);
            self.walk(
                &next,
                trace,
                optimal,
                spent + cost,
                on_path,
                moves,
                found,
                steps,
                budget,
            )?;
            moves.pop();
            on_path.remove(&next);
        }
        Ok(())
    }
}
