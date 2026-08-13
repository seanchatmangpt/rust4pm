//! Proof of concept: enumerate *every* optimal alignment, not just one.
//!
//! [`super::align`] keeps one predecessor per state, so it can only ever reconstruct a single
//! optimal alignment. This module runs the same [`PetriNetAlignment`] problem through its own
//! Dijkstra, which keeps *all* equally-cheap predecessors, and then walks the resulting
//! predecessor DAG to yield one alignment per path through it.
//!
//! It deliberately lives beside the tuned search rather than inside it:
//!
//! - It is unidirectional. Bidirectional search terminates as soon as the two frontiers can no
//!   longer beat the best meeting, at which point a meeting state may still be missing some of
//!   its optimal in-edges. That is harmless when one path is wanted and silently incomplete when
//!   all of them are.
//! - It does not prune commuting move orders. [`PetriNetAlignment::expand`] skips model moves
//!   directly after a log move, which loses alignments that differ only in that order. The prune
//!   keys off the step a state was reached by, so passing `via: None` turns it off without
//!   touching the tuned search.
//! - It stores predecessors in a growable arena rather than one word per node, which the tuned
//!   search cannot afford.
//!
//! # Zero-cost cycles
//!
//! Silent and synchronous moves may cost `0`, so a tau loop is a zero-cost cycle, and a net with
//! one has infinitely many optimal alignments. Enumeration therefore yields **simple** paths only:
//! no `(marking, trace position)` state is visited twice. Where no zero-cost cycle exists that is
//! the same set, since any repeated state encloses a segment costing either more than `0` (not
//! optimal) or exactly `0` (a zero-cost cycle).
//!
//! Even so the count is exponential in general, so the result hands out a lazy iterator.
use std::{cmp::Ordering, collections::VecDeque};

use super::{
    super::{
        build_model, sync_prod_net::SyncProductNet, AlignmentMove, AlignmentOptions,
        AlignmentResult,
    },
    AlignmentError, PetriNetAlignment, PetriNetAlignmentSpace, PetriNetStep,
};
use crate::{
    utils::dijkstra_search::{NodeID, SearchError, SearchLimits, SearchProblem},
    PetriNet,
};

/// Absent index, used both for "no parent edge" and for the end of a parent list
const NO_EDGE: u32 = u32::MAX;

/// A node of the predecessor DAG
#[derive(Debug)]
struct DagNode<C> {
    /// Shortest known distance from the start
    distance: C,
    /// Whether the distance is final (the node has been taken off the queue)
    finished: bool,
    /// First of its optimal in-edges, or [`NO_EDGE`] for the start node
    parents: u32,
}

/// One optimal in-edge, chained into a per-node list
#[derive(Debug, Clone)]
struct ParentEdge<S> {
    /// The node this edge comes from
    from: NodeID,
    /// The step fired to get from there to the node owning this edge
    step: S,
    /// Next edge of the same node, or [`NO_EDGE`]
    next: u32,
}

/// The optimal in-edges of every reached state
#[derive(Debug)]
struct Dag<S, C> {
    nodes: Vec<DagNode<C>>,
    edges: Vec<ParentEdge<S>>,
}

/// What [`search_all`] found: the DAG plus the goal it ends at
struct AllOptimalPaths<S, C> {
    dag: Dag<S, C>,
    goal: NodeID,
    cost: C,
    states_visited: usize,
}

/// Dijkstra keeping *every* optimal predecessor of each state.
///
/// Differs from [`crate::utils::dijkstra_search::search`] in three points, all of which matter
/// only when all optimal paths are wanted:
///
/// - a state reached again at exactly its known distance records another parent instead of being
///   dropped, including when it is already finished (a zero-cost edge can reach a finished state),
/// - reaching the goal does not end the search: every state at the goal's distance is still
///   expanded, so zero-cost edges into the goal are recorded too,
/// - `via` is passed as `None`, which turns off the commuting-move-order prune.
fn search_all<P: SearchProblem>(
    problem: &mut P,
    limits: SearchLimits,
) -> Result<AllOptimalPaths<P::Step, P::Cost>, SearchError> {
    let start = problem.initial();
    let num_buckets_cost = problem.max_edge_cost() + P::Cost::from(1);
    let num_buckets: usize = num_buckets_cost
        .try_into()
        .map_err(|_| SearchError::MaxEdgeCostTooLarge)?;

    let mut buckets: Vec<VecDeque<NodeID>> = Vec::new();
    buckets.resize_with(num_buckets, VecDeque::new);
    buckets[0].push_back(start);
    let mut bucket = 0usize;
    let mut queued = 1usize;

    let mut dag = Dag {
        nodes: vec![DagNode {
            distance: P::Cost::from(0),
            finished: false,
            parents: NO_EDGE,
        }],
        edges: Vec::new(),
    };

    let limit = limits.max_states.unwrap_or(usize::MAX);
    let queued_limit = limits.max_states_queued.unwrap_or(usize::MAX);
    let mut states_visited: usize = 0;
    let mut goal: Option<(NodeID, P::Cost)> = None;

    loop {
        // Next node to finish, skipping entries a shorter distance has obsoleted
        let next = loop {
            if queued == 0 {
                break None;
            }
            while buckets[bucket].is_empty() {
                bucket = (bucket + 1) % buckets.len();
            }
            let node = buckets[bucket][0];
            if dag.nodes[node as usize].finished {
                buckets[bucket].pop_front();
                queued -= 1;
                continue;
            }
            break Some((dag.nodes[node as usize].distance, node));
        };
        let Some((distance, node)) = next else { break };
        // Everything at the goal's distance still has to be expanded, so that zero-cost edges
        // into the goal are recorded; anything beyond it cannot lie on an optimal path
        if goal.is_some_and(|(_, cost)| distance > cost) {
            break;
        }

        buckets[bucket].pop_front();
        queued -= 1;
        dag.nodes[node as usize].finished = true;
        states_visited += 1;
        if states_visited > limit {
            return Err(SearchError::LimitReached);
        }
        if dag.nodes.len() > queued_limit {
            return Err(SearchError::QueuedLimitReached);
        }

        if problem.is_goal(node) {
            goal = Some((node, distance));
            // Nothing an optimal alignment does continues past the goal
            continue;
        }

        let nodes = &mut dag.nodes;
        let edges = &mut dag.edges;
        let buckets = &mut buckets;
        let queued = &mut queued;
        // `via: None` leaves the commuting-move-order prune off, so both orders are generated
        problem.expand(node, None, |succ, is_new, cost, step| {
            let new_distance = distance + cost;
            let idx = (new_distance % num_buckets_cost).try_into().unwrap_or(0);
            let mut push_edge = |next| {
                edges.push(ParentEdge { from: node, step, next });
                (edges.len() - 1) as u32
            };
            if is_new {
                debug_assert_eq!(succ as usize, nodes.len(), "no NodeID should be skipped");
                let head = push_edge(NO_EDGE);
                nodes.push(DagNode {
                    distance: new_distance,
                    finished: false,
                    parents: head,
                });
                buckets[idx].push_back(succ);
                *queued += 1;
                return;
            }
            let entry = &mut nodes[succ as usize];
            match new_distance.cmp(&entry.distance) {
                Ordering::Greater => {}
                // Another equally cheap way in, i.e. another optimal alignment. A finished node
                // can still gain one, over a zero-cost edge from a state finished after it.
                Ordering::Equal => entry.parents = push_edge(entry.parents),
                Ordering::Less => {
                    debug_assert!(!entry.finished, "a finished node cannot be improved");
                    entry.distance = new_distance;
                    entry.parents = push_edge(NO_EDGE);
                    buckets[idx].push_back(succ);
                    *queued += 1;
                }
            }
        });
    }

    let Some((goal, cost)) = goal else {
        return Err(SearchError::Unreachable);
    };
    Ok(AllOptimalPaths {
        dag,
        goal,
        cost,
        states_visited,
    })
}

/// One optimal in-edge, with the step reduced to the move it stands for
#[derive(Debug, Clone)]
struct Edge {
    from: NodeID,
    /// Index of the fired transition in the synchronous product net
    transition: u32,
    next: u32,
}

/// Every optimal alignment of one trace, as the DAG of optimal moves between reached states.
///
/// The alignments themselves are produced on demand by [`AllOptimalAlignments::iter`], since there
/// may be exponentially many of them. See the [module docs] for which paths are enumerated.
///
/// [module docs]: self
#[derive(Debug, Clone)]
pub struct AllOptimalAlignments {
    cost: u32,
    states_visited: usize,
    /// The move each fired transition stands for, indexed by transition, `None` where unused
    moves: Vec<Option<AlignmentMove>>,
    /// First optimal in-edge per node, or [`NO_EDGE`]
    parents: Vec<u32>,
    edges: Vec<Edge>,
    start: NodeID,
    goal: NodeID,
}

impl AllOptimalAlignments {
    /// Cost shared by every optimal alignment
    pub fn cost(&self) -> u32 {
        self.cost
    }

    /// Number of states visited during the search
    pub fn states_visited(&self) -> usize {
        self.states_visited
    }

    /// Number of states on some optimal path, i.e. nodes of the DAG
    pub fn num_states(&self) -> usize {
        self.parents.len()
    }

    /// Yield the optimal alignments one at a time.
    ///
    /// There may be exponentially many, so bound the walk with
    /// [`Iterator::take`] unless the net is known to be small.
    pub fn iter(&self) -> OptimalAlignments<'_> {
        OptimalAlignments::new(self)
    }

    /// Collect at most `max` optimal alignments as [`AlignmentResult`]s
    pub fn to_results(&self, max: usize) -> Vec<AlignmentResult> {
        self.iter()
            .take(max)
            .map(|moves| AlignmentResult {
                moves,
                cost: self.cost,
                states_visited: self.states_visited,
            })
            .collect()
    }
}

/// One frame of the backwards walk: a state, and which of its in-edges to try next
#[derive(Debug)]
struct Frame {
    node: NodeID,
    cursor: u32,
}

/// Lazily walks the DAG backwards from the goal, yielding one optimal alignment per simple path
#[derive(Debug)]
pub struct OptimalAlignments<'a> {
    result: &'a AllOptimalAlignments,
    stack: Vec<Frame>,
    /// Which states the current path already holds, so it never revisits one
    on_path: Vec<bool>,
    /// The transitions fired along the current path, from the goal backwards
    steps: Vec<u32>,
    /// The goal is the start, so the one optimal alignment is empty
    pending_empty: bool,
}

impl<'a> OptimalAlignments<'a> {
    fn new(result: &'a AllOptimalAlignments) -> Self {
        let mut on_path = vec![false; result.parents.len()];
        on_path[result.goal as usize] = true;
        Self {
            stack: vec![Frame {
                node: result.goal,
                cursor: result.parents[result.goal as usize],
            }],
            on_path,
            steps: Vec::new(),
            pending_empty: result.goal == result.start,
            result,
        }
    }

    /// Turn the current path into moves, which the walk collected goal-first
    fn alignment(&self) -> Vec<AlignmentMove> {
        self.steps
            .iter()
            .rev()
            .map(|transition| {
                self.result.moves[*transition as usize]
                    .clone()
                    .expect("every fired transition is in the move table")
            })
            .collect()
    }
}

impl Iterator for OptimalAlignments<'_> {
    type Item = Vec<AlignmentMove>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pending_empty {
            self.pending_empty = false;
            self.stack.clear();
            return Some(Vec::new());
        }
        loop {
            let frame = self.stack.last_mut()?;
            let edge = frame.cursor;
            if edge == NO_EDGE {
                let done = self.stack.pop().expect("just read the top frame");
                self.on_path[done.node as usize] = false;
                // The bottom frame is the goal, which no step led into
                if !self.stack.is_empty() {
                    self.steps.pop();
                }
                continue;
            }
            let Edge {
                from,
                transition,
                next,
            } = self.result.edges[edge as usize].clone();
            frame.cursor = next;
            // Re-entering a state means a zero-cost cycle, of which there are infinitely many
            if self.on_path[from as usize] {
                continue;
            }
            self.steps.push(transition);
            if from == self.result.start {
                let alignment = self.alignment();
                self.steps.pop();
                return Some(alignment);
            }
            self.on_path[from as usize] = true;
            self.stack.push(Frame {
                node: from,
                cursor: self.result.parents[from as usize],
            });
        }
    }
}

/// Compute *every* optimal alignment of a single trace (given as activity sequence).
///
/// Slower and hungrier than [`crate::conformance::alignments::align_trace`], which returns one
/// optimal alignment: the search runs in one direction only and generates commuting move orders
/// that the tuned search prunes away. Prefer it whenever one alignment is enough.
///
/// Permits at most 255 ([`super::TokenCount::MAX`]) tokens in each place.
pub fn align_trace_all_optimal(
    net: &PetriNet,
    trace: &[&str],
    options: &AlignmentOptions,
) -> Result<AllOptimalAlignments, AlignmentError> {
    let model = build_model(net, &options.cost_fn)?;
    let sp = SyncProductNet::construct(&model, trace, &options.cost_fn);
    let mut space = PetriNetAlignmentSpace::default();
    let mut problem = PetriNetAlignment {
        net: &sp,
        space: &mut space,
        reverse: false,
    };
    let found = search_all(&mut problem, options.limits())?;
    Ok(collect(&sp, found))
}

/// Turn the search's DAG into a self-contained result, resolving steps to moves
fn collect(
    net: &SyncProductNet<'_>,
    found: AllOptimalPaths<PetriNetStep, u32>,
) -> AllOptimalAlignments {
    let edges: Vec<Edge> = found
        .dag
        .edges
        .iter()
        .map(|edge| Edge {
            from: edge.from,
            transition: edge.step.transition() as u32,
            next: edge.next,
        })
        .collect();
    let width = edges
        .iter()
        .map(|edge| edge.transition as usize + 1)
        .max()
        .unwrap_or(0);
    let mut moves = vec![None; width];
    for edge in &edges {
        let index = edge.transition as usize;
        if moves[index].is_none() {
            moves[index] = Some(net.transition(index).move_type.clone());
        }
    }
    AllOptimalAlignments {
        cost: found.cost,
        states_visited: found.states_visited,
        moves,
        parents: found.dag.nodes.iter().map(|node| node.parents).collect(),
        edges,
        start: 0,
        goal: found.goal,
    }
}

#[cfg(test)]
mod reference;

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use super::{align_trace_all_optimal, reference, AlignmentError};
    use crate::{
        conformance::alignments::{align_trace, AlignmentMove, AlignmentOptions},
        core::process_models::petri_net::ArcType,
        utils::dijkstra_search::SearchError,
        PetriNet,
    };

    /// What an alignment costs under [`CostFunction::standard`], to check it against the optimum
    ///
    /// [`CostFunction::standard`]: crate::conformance::alignments::cost::CostFunction::standard
    fn cost_of(net: &PetriNet, alignment: &[AlignmentMove]) -> u32 {
        alignment
            .iter()
            .map(|mv| match mv {
                AlignmentMove::SyncMove { .. } => 0,
                AlignmentMove::ModelMove { transition } => {
                    // Silent transitions are free under the standard cost function
                    let trans = net
                        .transitions
                        .get(&transition.0)
                        .expect("move refers to a transition of the net");
                    u32::from(trans.label.is_some())
                }
                AlignmentMove::LogMove { .. } => 1,
            })
            .sum()
    }

    /// All optimal alignments as comparable move sequences, checked to share the reported cost
    fn all(net: &PetriNet, trace: &[&str]) -> Vec<Vec<AlignmentMove>> {
        let options = AlignmentOptions::default();
        let result = align_trace_all_optimal(net, trace, &options).unwrap();
        let one = align_trace(net, trace, &options).unwrap();
        assert_eq!(
            result.cost(),
            one.cost,
            "enumerating must not change the optimal cost"
        );
        let alignments: Vec<_> = result.iter().take(1000).collect();
        for alignment in &alignments {
            assert_eq!(
                cost_of(net, alignment),
                result.cost(),
                "every alignment costs the optimum"
            );
        }
        for (index, alignment) in alignments.iter().enumerate() {
            assert!(
                !alignments[..index].contains(alignment),
                "no alignment is repeated"
            );
        }
        assert!(
            alignments.contains(&one.moves),
            "the single-alignment search's answer must be among them"
        );
        alignments
    }

    /// `p0 -a-> p1`, with `a` visible
    fn one_step_net() -> PetriNet {
        let mut net = PetriNet::new();
        let p0 = net.add_place(None);
        let p1 = net.add_place(None);
        let a = net.add_transition(Some("a".to_string()), None);
        net.add_arc(ArcType::PlaceTransition(p0.0, a.0), None);
        net.add_arc(ArcType::TransitionPlace(a.0, p1.0), None);
        net.initial_marking = Some([(p0, 1)].into_iter().collect());
        net.final_markings = Some(vec![[(p1, 1)].into_iter().collect()]);
        net
    }

    /// A fitting trace has exactly one optimal alignment
    #[test]
    fn perfectly_fitting_trace_aligns_one_way() {
        assert_eq!(all(&one_step_net(), &["a"]).len(), 1);
    }

    /// A log move and a model move commute, so both orders are optimal alignments. The tuned
    /// search prunes one of them away, which is exactly what this module must not do.
    #[test]
    fn both_orders_of_commuting_moves() {
        let alignments = all(&one_step_net(), &["x"]);
        assert_eq!(alignments.len(), 2, "log-then-model and model-then-log");
        for alignment in &alignments {
            assert_eq!(alignment.len(), 2);
        }
    }

    /// Two ways through the model, each interleaving with the log move in two ways
    #[test]
    fn alternatives_multiply_with_orders() {
        let mut net = PetriNet::new();
        let p0 = net.add_place(None);
        let p1 = net.add_place(None);
        for label in ["a", "b"] {
            let t = net.add_transition(Some(label.to_string()), None);
            net.add_arc(ArcType::PlaceTransition(p0.0, t.0), None);
            net.add_arc(ArcType::TransitionPlace(t.0, p1.0), None);
        }
        net.initial_marking = Some([(p0, 1)].into_iter().collect());
        net.final_markings = Some(vec![[(p1, 1)].into_iter().collect()]);
        assert_eq!(all(&net, &["c"]).len(), 4);
    }

    /// Concurrent branches can be aligned in either order
    #[test]
    fn concurrent_model_moves_interleave() {
        let mut net = PetriNet::new();
        let places: Vec<_> = (0..5).map(|_| net.add_place(None)).collect();
        // split: p0 -> p1 + p2, then a: p1 -> p3 and b: p2 -> p4
        let split = net.add_transition(None, None);
        net.add_arc(ArcType::PlaceTransition(places[0].0, split.0), None);
        net.add_arc(ArcType::TransitionPlace(split.0, places[1].0), None);
        net.add_arc(ArcType::TransitionPlace(split.0, places[2].0), None);
        for (from, to, label) in [(1, 3, "a"), (2, 4, "b")] {
            let t = net.add_transition(Some(label.to_string()), None);
            net.add_arc(ArcType::PlaceTransition(places[from].0, t.0), None);
            net.add_arc(ArcType::TransitionPlace(t.0, places[to].0), None);
        }
        net.initial_marking = Some([(places[0], 1)].into_iter().collect());
        net.final_markings = Some(vec![[(places[3], 1), (places[4], 1)].into_iter().collect()]);
        // Nothing in the log, so both branches are model moves and may fire in either order
        assert_eq!(all(&net, &[]).len(), 2);
    }

    /// A zero-cost cycle admits infinitely many optimal alignments, so only simple paths count
    #[test]
    fn zero_cost_cycle_yields_finitely_many() {
        let mut net = PetriNet::new();
        let places: Vec<_> = (0..3).map(|_| net.add_place(None)).collect();
        // Two silent transitions forming a cycle p0 -> p2 -> p0, both free
        let out = net.add_transition(None, None);
        net.add_arc(ArcType::PlaceTransition(places[0].0, out.0), None);
        net.add_arc(ArcType::TransitionPlace(out.0, places[2].0), None);
        let back = net.add_transition(None, None);
        net.add_arc(ArcType::PlaceTransition(places[2].0, back.0), None);
        net.add_arc(ArcType::TransitionPlace(back.0, places[0].0), None);
        // And a silent way to the final marking
        let done = net.add_transition(None, None);
        net.add_arc(ArcType::PlaceTransition(places[0].0, done.0), None);
        net.add_arc(ArcType::TransitionPlace(done.0, places[1].0), None);
        net.initial_marking = Some([(places[0], 1)].into_iter().collect());
        net.final_markings = Some(vec![[(places[1], 1)].into_iter().collect()]);

        let alignments = all(&net, &[]);
        assert_eq!(
            alignments.len(),
            1,
            "going round the cycle revisits a state, so only firing `done` counts"
        );
    }

    /// Deterministic xorshift, so a failing case can be reproduced from the seed alone
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    /// A small net that can always reach its final marking, with random detours around a chain.
    ///
    /// The chain `p0 -> p1 -> .. -> pn` keeps the final marking reachable, so most cases are worth
    /// comparing. The extra transitions take from one or two places and feed one or two others,
    /// arcs may repeat (their weights then add up), and labels may repeat or be absent, so silent
    /// moves and several transitions per activity both occur.
    fn random_net(rng: &mut Rng) -> PetriNet {
        let num_places = 3 + rng.below(3);
        let mut net = PetriNet::new();
        let places: Vec<_> = (0..num_places).map(|_| net.add_place(None)).collect();
        let label = |rng: &mut Rng| ["a", "b", "c"].get(rng.below(4)).map(|l| l.to_string());
        for step in 0..num_places - 1 {
            let transition = net.add_transition(label(rng), None);
            net.add_arc(ArcType::PlaceTransition(places[step].0, transition.0), None);
            net.add_arc(
                ArcType::TransitionPlace(transition.0, places[step + 1].0),
                None,
            );
        }
        for _ in 0..1 + rng.below(2) {
            let transition = net.add_transition(label(rng), None);
            for _ in 0..1 + rng.below(2) {
                let place = places[rng.below(num_places)];
                net.add_arc(ArcType::PlaceTransition(place.0, transition.0), None);
            }
            for _ in 0..1 + rng.below(2) {
                let place = places[rng.below(num_places)];
                net.add_arc(ArcType::TransitionPlace(transition.0, place.0), None);
            }
        }
        net.initial_marking = Some([(places[0], 1)].into_iter().collect());
        net.final_markings = Some(vec![[(places[num_places - 1], 1)].into_iter().collect()]);
        net
    }

    /// Alignments as comparable keys, since [`AlignmentMove`] is neither `Hash` nor `Ord`
    fn keys(alignments: &[Vec<AlignmentMove>]) -> HashSet<String> {
        alignments
            .iter()
            .map(|alignment| format!("{alignment:?}"))
            .collect()
    }

    /// The real completeness check: on small random nets the enumeration must equal, exactly, what
    /// an independent brute-force walk over the state space finds.
    ///
    /// This is what rules out both missing alignments and invented ones; the reference shares no
    /// code with the search and builds its alignments by firing transitions itself.
    #[test]
    fn matches_a_naive_reference_on_random_nets() {
        // Enough that the reference gives up on the few nets whose state space explodes, without
        // it ever being the reason a real disagreement goes unnoticed
        let budget = 20_000;
        let options = AlignmentOptions::default();
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let (mut compared, mut alignments_checked, mut skipped, mut unreachable) = (0, 0, 0, 0);

        for case in 0..500 {
            let net = random_net(&mut rng);
            let trace: Vec<&str> = (0..rng.below(4))
                .map(|_| ["a", "b", "z"][rng.below(3)])
                .collect();
            let context = || format!("case {case}, trace {trace:?}");

            let reference = reference::Reference::build(&net);
            let Ok(expected) = reference.all_optimal(&trace, budget) else {
                skipped += 1;
                continue;
            };
            let actual = align_trace_all_optimal(&net, &trace, &options);
            match (expected, actual) {
                // The final marking is out of reach, which both have to agree on
                (None, Err(AlignmentError::SearchError(SearchError::Unreachable))) => {
                    unreachable += 1;
                }
                (None, Err(error)) => panic!("{}: unexpected {error:?}", context()),
                (None, Ok(found)) => panic!(
                    "{}: enumerated {} alignments of an unreachable final marking",
                    context(),
                    found.iter().take(1).count()
                ),
                (Some(expected), Err(error)) => panic!(
                    "{}: {error:?}, but {} optimal alignments exist",
                    context(),
                    expected.len()
                ),
                (Some(expected), Ok(found)) => {
                    let mine: Vec<_> = found.iter().take(budget + 1).collect();
                    assert!(mine.len() <= budget, "{}: more than the budget", context());
                    assert_eq!(
                        keys(&mine),
                        keys(&expected),
                        "{}: enumerated alignments differ from the reference's",
                        context()
                    );
                    alignments_checked += mine.len();
                    compared += 1;
                }
            }
        }
        println!(
            "compared {compared} nets ({alignments_checked} alignments), \
             {unreachable} unreachable, {skipped} beyond the reference"
        );
        assert!(
            compared > 100,
            "the generator has to produce enough alignable nets for this to mean anything"
        );
    }

    /// A real net, to check the enumeration holds up outside hand-built examples
    #[test]
    fn sepsis_variants_agree_with_the_single_alignment() {
        use crate::{
            core::event_data::case_centric::utils::activity_projection::log_to_activity_projection,
            test_utils::get_test_data_path, EventLog, Importable,
        };

        let test_path = get_test_data_path();
        let log =
            EventLog::import_from_path(test_path.join("xes").join("Sepsis Cases - Event Log.xes.gz"))
                .unwrap();
        let net =
            PetriNet::import_pnml(test_path.join("petri-net").join("sepsis-DISCovered.apnml"))
                .unwrap();
        let projection = log_to_activity_projection(&log);
        let mut variants: Vec<Vec<&str>> = projection
            .traces
            .iter()
            .map(|(indices, _frequency)| {
                indices
                    .iter()
                    .map(|&index| projection.activities[index].as_str())
                    .collect()
            })
            .collect();
        // Short variants first, so the check stays quick without the tuned search's pruning
        variants.sort_by_key(Vec::len);

        // Enough to hold every optimal alignment of these variants; longer ones have far more
        let cap = 50_000;
        let options = AlignmentOptions::default();
        for trace in variants.iter().take(20) {
            let all = align_trace_all_optimal(&net, trace, &options).unwrap();
            let one = align_trace(&net, trace, &options).unwrap();
            assert_eq!(all.cost(), one.cost);
            let alignments: Vec<_> = all.iter().take(cap).collect();
            assert!(
                alignments.len() < cap,
                "the variant has more optimal alignments than this check enumerates"
            );
            assert!(
                !alignments.is_empty(),
                "an optimal alignment exists, so at least one must be enumerated"
            );
            for alignment in &alignments {
                assert_eq!(
                    cost_of(&net, alignment),
                    all.cost(),
                    "every enumerated alignment costs the optimum"
                );
            }
            assert!(
                alignments.contains(&one.moves),
                "the single-alignment answer must be among them"
            );
        }
    }

    /// A sync move does not commute with a log move, so it pins the order down to one alignment
    #[test]
    fn sync_moves_pin_the_trace_order() {
        // Skipping `x` and then synchronising on `a` costs 1; firing `a` as a model move first
        // leaves both events to log moves, which costs 3
        let alignments = all(&one_step_net(), &["x", "a"]);
        assert_eq!(alignments.len(), 1);
        assert!(matches!(alignments[0][0], AlignmentMove::LogMove { .. }));
        assert!(matches!(alignments[0][1], AlignmentMove::SyncMove { .. }));
    }
}
