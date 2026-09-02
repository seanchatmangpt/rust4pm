//! BPMN export for [`Powl`] models.
//!
//! Independently designed for this fork -- there is no pre-existing BPMN type in this crate to
//! reuse, so this module introduces a new, minimal [`BpmnModel`] scoped to exactly what this
//! fork's [`PowlNode`] constructs can express: [`Task`](BpmnNode::Task) nodes for leaves,
//! [`Gateway`](BpmnNode::Gateway) nodes (exclusive or parallel) for branching/looping/concurrency,
//! a flat list of [`SequenceFlow`] edges for control flow, and a single [`StartEvent`](BpmnNode::StartEvent)/
//! [`EndEvent`](BpmnNode::EndEvent) pair bracketing the whole model.
//!
//! [`Powl::to_bpmn`] mirrors the *structural role* of the real reference translation
//! (`~/POWL/powl/conversion/variants/to_bpmn.py`, read in full this session, not ported): tasks
//! for activities, gateways for branching/looping/concurrency, sequence flows for control-flow
//! edges. The concrete graph construction below is an independent design (a `(before-gateway,
//! node, after-gateway)` triple per [`ChoiceGraphNode`] child rather than the reference's
//! `networkx`-based node/edge composition), not a transliteration of the reference's Python.
//!
//! # Output format
//!
//! [`Powl::to_bpmn`] returns a [`BpmnModel`] (an in-memory graph), not a `String`, so callers can
//! inspect its structure directly (as the tests below do) without round-tripping through XML.
//! [`BpmnModel::to_xml_string`] then serializes that model to real, well-formed BPMN 2.0 XML
//! using `quick-xml`'s `Writer` (already a dependency of this crate, used the same way
//! [`crate::core::process_models::case_centric::petri_net::pnml::export_pnml`] uses it) --
//! attribute values and text content are escaped by `quick-xml` itself (`Attribute::from`
//! escapes the value half of a `(&str, &str)` tuple; `BytesText::new` escapes text content), so
//! no hand-rolled string concatenation is needed and activity labels containing XML-special
//! characters (`&`, `<`, `>`, `"`) are handled correctly -- see the `escaping_is_real` test.

use quick_xml::Writer;

use crate::core::process_models::case_centric::process_tree::{LeafLabel, OperatorType};

use super::{ChoiceGraphEndpoint, ChoiceGraphNode, PartialOrderNode, Powl, PowlNode, PowlOperator};

/// Identifies one node (task, gateway, or start/end event) within a [`BpmnModel`]. Opaque outside
/// this module beyond equality/ordering -- callers get IDs back from `add_*` methods and pass
/// them to [`BpmnModel::add_flow`]; nothing about the numeric value is meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BpmnNodeId(usize);

impl BpmnNodeId {
    /// The XML `id` attribute value for this node -- stable for the lifetime of the model, unique
    /// within it.
    fn xml_id(self) -> String {
        format!("Node_{}", self.0)
    }
}

/// Which of the two gateway routing semantics a [`BpmnNode::Gateway`] uses -- BPMN's own
/// distinction, and exactly the two kinds this fork's POWL constructs ever need: exclusive
/// (`ExclusiveChoice`, `Loop`, and [`ChoiceGraphNode`] branching -- only one outgoing branch is
/// taken) and parallel (`Concurrency` and [`PartialOrderNode`] -- every outgoing branch is taken).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GatewayKind {
    /// BPMN `exclusiveGateway`: routes to exactly one outgoing sequence flow.
    Exclusive,
    /// BPMN `parallelGateway`: routes to every outgoing sequence flow (fork), or waits on every
    /// incoming sequence flow (join).
    Parallel,
}

/// One element of a [`BpmnModel`]: a start/end event, a task (one per [`PowlLeaf`](super::PowlLeaf),
/// silent or not), or a gateway. Kept deliberately minimal -- no BPMN feature this fork's POWL
/// model cannot itself express (no sub-processes, no message flows, no data objects) is
/// represented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BpmnNode {
    /// The single start event bracketing the whole exported model.
    StartEvent,
    /// The single end event bracketing the whole exported model.
    EndEvent,
    /// A task, one per [`PowlLeaf`](super::PowlLeaf). `Some(label)` for a non-silent leaf; `None`
    /// for a silent (tau) leaf -- still a real task node (so the `Task node (for each Leaf)`
    /// requirement holds literally for every leaf), just with no `name` attribute on export.
    Task(Option<String>),
    /// An exclusive or parallel gateway introduced by an operator, [`PartialOrderNode`], or
    /// [`ChoiceGraphNode`] translation.
    Gateway(GatewayKind),
}

/// A directed control-flow edge between two [`BpmnNode`]s, exported as a BPMN `sequenceFlow`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceFlow {
    /// The source node.
    pub from: BpmnNodeId,
    /// The target node.
    pub to: BpmnNodeId,
}

/// A minimal, real BPMN model: a flat node list plus a flat sequence-flow edge list. Node order
/// is insertion order, which [`BpmnModel::to_xml_string`] preserves -- so XML output is
/// deterministic for a deterministic [`Powl`] translation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BpmnModel {
    nodes: Vec<(BpmnNodeId, BpmnNode)>,
    flows: Vec<SequenceFlow>,
}

impl BpmnModel {
    /// Creates a new, empty model (no nodes, no flows).
    pub fn new() -> Self {
        Self::default()
    }

    fn add_node(&mut self, node: BpmnNode) -> BpmnNodeId {
        let id = BpmnNodeId(self.nodes.len());
        self.nodes.push((id, node));
        id
    }

    /// Adds a new start event and returns its id.
    pub fn add_start_event(&mut self) -> BpmnNodeId {
        self.add_node(BpmnNode::StartEvent)
    }

    /// Adds a new end event and returns its id.
    pub fn add_end_event(&mut self) -> BpmnNodeId {
        self.add_node(BpmnNode::EndEvent)
    }

    /// Adds a new task (`Some(label)` for a real activity, `None` for a silent/tau leaf) and
    /// returns its id.
    pub fn add_task(&mut self, label: Option<String>) -> BpmnNodeId {
        self.add_node(BpmnNode::Task(label))
    }

    /// Adds a new gateway of the given kind and returns its id.
    pub fn add_gateway(&mut self, kind: GatewayKind) -> BpmnNodeId {
        self.add_node(BpmnNode::Gateway(kind))
    }

    /// Adds a directed sequence flow from `from` to `to`.
    pub fn add_flow(&mut self, from: BpmnNodeId, to: BpmnNodeId) {
        self.flows.push(SequenceFlow { from, to });
    }

    /// All nodes in the model, in insertion order.
    pub fn nodes(&self) -> &[(BpmnNodeId, BpmnNode)] {
        &self.nodes
    }

    /// All sequence flows in the model, in insertion order.
    pub fn flows(&self) -> &[SequenceFlow] {
        &self.flows
    }

    /// The number of [`BpmnNode::Task`] nodes in the model.
    pub fn task_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|(_, n)| matches!(n, BpmnNode::Task(_)))
            .count()
    }

    /// The number of [`BpmnNode::Gateway`] nodes of the given kind in the model.
    pub fn gateway_count(&self, kind: GatewayKind) -> usize {
        self.nodes
            .iter()
            .filter(|(_, n)| matches!(n, BpmnNode::Gateway(k) if *k == kind))
            .count()
    }

    /// The total number of [`SequenceFlow`]s in the model.
    pub fn flow_count(&self) -> usize {
        self.flows.len()
    }

    /// The number of outgoing [`SequenceFlow`]s from `id` -- a gateway's fan-out (or a task's,
    /// which should always be `<= 1`).
    pub fn out_degree(&self, id: BpmnNodeId) -> usize {
        self.flows.iter().filter(|f| f.from == id).count()
    }

    /// The number of incoming [`SequenceFlow`]s into `id`.
    pub fn in_degree(&self, id: BpmnNodeId) -> usize {
        self.flows.iter().filter(|f| f.to == id).count()
    }

    /// Serializes this model to real, well-formed BPMN 2.0 XML: a `<definitions>` root containing
    /// one `<process>` with a `<startEvent>`, `<endEvent>`, one `<task>`/`<exclusiveGateway>`/
    /// `<parallelGateway>` per node, and one `<sequenceFlow>` per edge. Built with `quick-xml`'s
    /// `Writer` (the same library and calling convention
    /// [`export_pnml`](crate::core::process_models::case_centric::petri_net::pnml::export_pnml)
    /// already uses for PNML export), so every attribute value and text node is escaped by
    /// `quick-xml` itself -- never hand-concatenated.
    pub fn to_xml_string(&self) -> std::io::Result<String> {
        let mut writer = Writer::new_with_indent(Vec::new(), b' ', 2);

        writer
            .create_element("definitions")
            .with_attributes(vec![
                ("xmlns", "http://www.omg.org/spec/BPMN/20100524/MODEL"),
                ("id", "Definitions_1"),
                ("targetNamespace", "https://github.com/rust4pm/powl/bpmn"),
            ])
            .write_inner_content(|writer| {
                writer
                    .create_element("process")
                    .with_attributes(vec![("id", "Process_1"), ("isExecutable", "false")])
                    .write_inner_content(|writer| self.write_nodes_and_flows(writer))?;
                Ok(())
            })?;

        let bytes = writer.into_inner();
        String::from_utf8(bytes).map_err(std::io::Error::other)
    }

    fn write_nodes_and_flows<W: std::io::Write>(
        &self,
        writer: &mut Writer<W>,
    ) -> std::io::Result<()> {
        for (id, node) in &self.nodes {
            let xml_id = id.xml_id();
            match node {
                BpmnNode::StartEvent => {
                    writer
                        .create_element("startEvent")
                        .with_attribute(("id", xml_id.as_str()))
                        .write_empty()?;
                }
                BpmnNode::EndEvent => {
                    writer
                        .create_element("endEvent")
                        .with_attribute(("id", xml_id.as_str()))
                        .write_empty()?;
                }
                BpmnNode::Task(label) => {
                    let el = writer
                        .create_element("task")
                        .with_attribute(("id", xml_id.as_str()));
                    if let Some(name) = label {
                        el.with_attribute(("name", name.as_str())).write_empty()?;
                    } else {
                        el.write_empty()?;
                    }
                }
                BpmnNode::Gateway(GatewayKind::Exclusive) => {
                    writer
                        .create_element("exclusiveGateway")
                        .with_attribute(("id", xml_id.as_str()))
                        .write_empty()?;
                }
                BpmnNode::Gateway(GatewayKind::Parallel) => {
                    writer
                        .create_element("parallelGateway")
                        .with_attribute(("id", xml_id.as_str()))
                        .write_empty()?;
                }
            }
        }

        for (i, flow) in self.flows.iter().enumerate() {
            let flow_id = format!("Flow_{i}");
            let src_id = flow.from.xml_id();
            let tgt_id = flow.to.xml_id();
            writer
                .create_element("sequenceFlow")
                .with_attributes(vec![
                    ("id", flow_id.as_str()),
                    ("sourceRef", src_id.as_str()),
                    ("targetRef", tgt_id.as_str()),
                ])
                .write_empty()?;
        }

        Ok(())
    }
}

impl Powl {
    /// Translates this POWL model into a [`BpmnModel`]: tasks for every leaf, exclusive gateways
    /// for `ExclusiveChoice`/`Loop`/[`ChoiceGraphNode`] branching, parallel gateways for
    /// `Concurrency`/[`PartialOrderNode`] concurrency, and sequence flows mirroring every
    /// control-flow edge -- bracketed by one overall start and end event.
    ///
    /// Frequency tags are expanded first (see [`Powl::expand_frequency_tags`]), matching
    /// [`Powl::to_petri_net`]'s own preprocessing, so a skippable or repeatable node's
    /// multiplicity is represented as real choice-graph gateway structure rather than silently
    /// dropped.
    pub fn to_bpmn(&self) -> BpmnModel {
        let expanded = self.expand_frequency_tags();
        let mut model = BpmnModel::new();
        let start = model.add_start_event();
        let end = model.add_end_event();
        let (entry, exit) = translate_node(&expanded.root, &mut model);
        model.add_flow(start, entry);
        model.add_flow(exit, end);
        model
    }
}

/// Recursively translates one [`PowlNode`] into the BPMN element set, adding nodes and flows to
/// `model` and returning `(entry, exit)`: the node a caller should draw its own incoming flow
/// into, and the node a caller should draw its own outgoing flow from. Every node kind returns
/// real, already-wired ids -- never a placeholder -- so composition is just `add_flow` calls at
/// each level, the same shape as [`PowlNode::add_to_petri_net`]'s `(PlaceID, PlaceID)` return.
fn translate_node(node: &PowlNode, model: &mut BpmnModel) -> (BpmnNodeId, BpmnNodeId) {
    match node {
        PowlNode::Leaf(leaf) => {
            let label = match &leaf.leaf.activity_label {
                LeafLabel::Activity(a) => Some(a.clone()),
                LeafLabel::Tau => None,
            };
            let id = model.add_task(label);
            (id, id)
        }
        PowlNode::Operator(op) => translate_operator(op, model),
        PowlNode::PartialOrder(po) => translate_partial_order(po, model),
        PowlNode::ChoiceGraph(cg) => translate_choice_graph(cg, model),
    }
}

/// Translates a block-structured [`PowlOperator`], reusing the same per-`OperatorType` structural
/// role as [`PowlOperator::add_to_petri_net`] (sequence chains entry->exit; exclusive choice fans
/// out from/into a shared pair of exclusive gateways; concurrency fans out from/into a shared pair
/// of parallel gateways; loop wires a do-part and one or more redo-parts around a converging/
/// diverging exclusive-gateway pair) but over BPMN elements instead of Petri net places/
/// transitions.
fn translate_operator(op: &PowlOperator, model: &mut BpmnModel) -> (BpmnNodeId, BpmnNodeId) {
    if op.children.is_empty() {
        // No real reference case produces this (every real POWL operator has children), but a
        // childless operator must still translate to *something* connectable rather than panic.
        let id = model.add_task(None);
        return (id, id);
    }

    match op.operator_type {
        OperatorType::Sequence => {
            let mut children = op.children.iter();
            // Safe: emptiness handled above.
            let (entry, mut prev_exit) = translate_node(children.next().unwrap(), model);
            for child in children {
                let (child_entry, child_exit) = translate_node(child, model);
                model.add_flow(prev_exit, child_entry);
                prev_exit = child_exit;
            }
            (entry, prev_exit)
        }
        OperatorType::ExclusiveChoice => {
            fan_out_and_in(&op.children, GatewayKind::Exclusive, model)
        }
        OperatorType::Concurrency => fan_out_and_in(&op.children, GatewayKind::Parallel, model),
        OperatorType::Loop => {
            let loop_start = model.add_gateway(GatewayKind::Exclusive); // converging: entry + every redo exit
            let loop_end = model.add_gateway(GatewayKind::Exclusive); // diverging: exit + every redo entry

            let mut children = op.children.iter();
            // Safe: emptiness handled above.
            let (do_entry, do_exit) = translate_node(children.next().unwrap(), model);
            model.add_flow(loop_start, do_entry);
            model.add_flow(do_exit, loop_end);

            for redo in children {
                let (redo_entry, redo_exit) = translate_node(redo, model);
                model.add_flow(loop_end, redo_entry);
                model.add_flow(redo_exit, loop_start);
            }

            (loop_start, loop_end)
        }
    }
}

/// Shared helper for `ExclusiveChoice` and `Concurrency`: a diverging gateway fans out to every
/// child's entry, every child's exit fans into a converging gateway of the same kind. `children`
/// must be non-empty (checked by the caller).
fn fan_out_and_in(
    children: &[PowlNode],
    kind: GatewayKind,
    model: &mut BpmnModel,
) -> (BpmnNodeId, BpmnNodeId) {
    let diverging = model.add_gateway(kind);
    let converging = model.add_gateway(kind);
    for child in children {
        let (entry, exit) = translate_node(child, model);
        model.add_flow(diverging, entry);
        model.add_flow(exit, converging);
    }
    (diverging, converging)
}

/// Translates a [`PartialOrderNode`] into a diverging/converging parallel-gateway pair (the
/// standard BPMN idiom for "every branch runs, order constrained only where specified"), wiring
/// only *direct* order edges (the same redundant-edge-implied-by-transitivity filter
/// [`PartialOrderNode::add_to_petri_net`] uses) so the exported diagram doesn't grow quadratically
/// with transitively-implied edges. A child with no predecessor connects from the diverging
/// gateway directly; a child with no successor connects to the converging gateway directly --
/// mirroring the reference's `__handle_StrictPartialOrder`, which does the same via its
/// `start_powl`/`end_edges` sets.
fn translate_partial_order(po: &PartialOrderNode, model: &mut BpmnModel) -> (BpmnNodeId, BpmnNodeId) {
    let diverging = model.add_gateway(GatewayKind::Parallel);
    let converging = model.add_gateway(GatewayKind::Parallel);

    if po.children.is_empty() {
        model.add_flow(diverging, converging);
        return (diverging, converging);
    }

    let ends: Vec<(BpmnNodeId, BpmnNodeId)> = po
        .children
        .iter()
        .map(|child| translate_node(child, model))
        .collect();

    // Direct edges only: drop (a, c) whenever some b makes it implied by (a, b) and (b, c) --
    // identical filter to PartialOrderNode::add_to_petri_net, so the BPMN diagram has exactly the
    // same edge count as the Petri net translation's synchronizing-transition count.
    let direct_edges: Vec<(usize, usize)> = po
        .order
        .iter()
        .copied()
        .filter(|&(a, c)| {
            !po.order
                .iter()
                .any(|&(x, y)| x == a && y != c && po.order.contains(&(y, c)))
        })
        .collect();

    let has_predecessor: std::collections::BTreeSet<usize> =
        direct_edges.iter().map(|&(_, b)| b).collect();
    let has_successor: std::collections::BTreeSet<usize> =
        direct_edges.iter().map(|&(a, _)| a).collect();

    for (idx, &(entry, _)) in ends.iter().enumerate() {
        if !has_predecessor.contains(&idx) {
            model.add_flow(diverging, entry);
        }
    }
    for (idx, &(_, exit)) in ends.iter().enumerate() {
        if !has_successor.contains(&idx) {
            model.add_flow(exit, converging);
        }
    }
    for (a, b) in direct_edges {
        model.add_flow(ends[a].1, ends[b].0);
    }

    (diverging, converging)
}

/// Translates a [`ChoiceGraphNode`] into BPMN gateways: an "after-start" exclusive gateway
/// (`start_gw`) and a "before-end" exclusive gateway (`end_gw`) stand in for the choice graph's
/// artificial `▷`/`□` boundary nodes, and every real child gets its own `(before, after)`
/// exclusive-gateway pair so that arbitrary in-degree/out-degree (including a self-loop, the
/// generalized-loop case) can be wired without ambiguity. Every graph edge `(from, to)` becomes
/// exactly one [`SequenceFlow`] between the relevant gateways -- a self-loop `Child(i)->Child(i)`
/// becomes `after[i] -> before[i]`, a real BPMN cycle around that child's gateway pair, which is
/// exactly the standard BPMN loop idiom. This mirrors the reference's `__handle_decision_graph`,
/// which does the same "gateway before and after every node, including the artificial start/end"
/// construction over an `nx.DiGraph`, but built directly against [`BpmnModel`] rather than through
/// an intermediate `networkx` graph.
fn translate_choice_graph(cg: &ChoiceGraphNode, model: &mut BpmnModel) -> (BpmnNodeId, BpmnNodeId) {
    use ChoiceGraphEndpoint::{Child, End, Start};

    let start_gw = model.add_gateway(GatewayKind::Exclusive);
    let end_gw = model.add_gateway(GatewayKind::Exclusive);

    if cg.children.is_empty() {
        model.add_flow(start_gw, end_gw);
        return (start_gw, end_gw);
    }

    let ends: Vec<(BpmnNodeId, BpmnNodeId)> = cg
        .children
        .iter()
        .map(|child| translate_node(child, model))
        .collect();

    let mut before = Vec::with_capacity(ends.len());
    let mut after = Vec::with_capacity(ends.len());
    for &(entry, exit) in &ends {
        let b = model.add_gateway(GatewayKind::Exclusive);
        let a = model.add_gateway(GatewayKind::Exclusive);
        model.add_flow(b, entry);
        model.add_flow(exit, a);
        before.push(b);
        after.push(a);
    }

    for &(from, to) in &cg.edges {
        // A valid choice graph (Def. 3.6) has no incoming edge into Start and no outgoing edge
        // out of End; an edge violating that has no real BPMN counterpart to draw, so it is
        // skipped rather than panicking on a malformed (non-`is_valid`) graph.
        let source = match from {
            Start => start_gw,
            Child(i) => after[i],
            End => continue,
        };
        let target = match to {
            Start => continue,
            Child(i) => before[i],
            End => end_gw,
        };
        model.add_flow(source, target);
    }

    (start_gw, end_gw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::process_models::case_centric::process_tree::OperatorType;
    use quick_xml::events::Event;
    use quick_xml::reader::Reader;

    /// Parses `xml` with a real XML reader end-to-end, panicking on any parse error -- proves
    /// well-formedness rather than checking for expected substrings.
    fn assert_well_formed_xml(xml: &str) -> Vec<String> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);
        let mut element_names = Vec::new();
        loop {
            match reader.read_event() {
                Ok(Event::Eof) => break,
                Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                    element_names.push(String::from_utf8(e.name().as_ref().to_vec()).unwrap());
                }
                Ok(_) => {}
                Err(e) => panic!("real XML parser rejected to_xml_string() output: {e}\n---\n{xml}"),
            }
        }
        element_names
    }

    #[test]
    fn simple_sequence_exports_expected_tasks_and_linear_flow_chain() {
        let mut op = PowlOperator::new(OperatorType::Sequence);
        op.children.push(PowlNode::new_leaf(Some("a".into())));
        op.children.push(PowlNode::new_leaf(Some("b".into())));
        op.children.push(PowlNode::new_leaf(Some("c".into())));
        let model = Powl::new(PowlNode::Operator(op));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 3, "one task per leaf");
        assert_eq!(bpmn.gateway_count(GatewayKind::Exclusive), 0);
        assert_eq!(bpmn.gateway_count(GatewayKind::Parallel), 0);

        // Walk the real flow chain start -> a -> b -> c -> end and confirm it is linear (every
        // node in the chain has in-degree <= 1 and out-degree <= 1).
        let start = bpmn
            .nodes()
            .iter()
            .find(|(_, n)| matches!(n, BpmnNode::StartEvent))
            .map(|&(id, _)| id)
            .unwrap();
        let mut current = start;
        let mut visited = 0;
        loop {
            let outs: Vec<BpmnNodeId> = bpmn
                .flows()
                .iter()
                .filter(|f| f.from == current)
                .map(|f| f.to)
                .collect();
            assert!(outs.len() <= 1, "linear sequence must not branch");
            visited += 1;
            match outs.first() {
                Some(&next) => current = next,
                None => break,
            }
        }
        // start, a, b, c, end = 5 nodes visited along the chain.
        assert_eq!(visited, 5);
        assert_eq!(bpmn.flow_count(), 4, "start->a, a->b, b->c, c->end");
    }

    #[test]
    fn exclusive_choice_operator_exports_exclusive_gateway_with_right_fan_out() {
        let mut op = PowlOperator::new(OperatorType::ExclusiveChoice);
        op.children.push(PowlNode::new_leaf(Some("a".into())));
        op.children.push(PowlNode::new_leaf(Some("b".into())));
        op.children.push(PowlNode::new_leaf(Some("c".into())));
        let model = Powl::new(PowlNode::Operator(op));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 3);
        assert_eq!(
            bpmn.gateway_count(GatewayKind::Exclusive),
            2,
            "one diverging + one converging exclusive gateway"
        );
        assert_eq!(bpmn.gateway_count(GatewayKind::Parallel), 0);

        let diverging = bpmn
            .nodes()
            .iter()
            .find(|(_, n)| matches!(n, BpmnNode::Gateway(GatewayKind::Exclusive)))
            .map(|&(id, _)| id)
            .unwrap();
        assert_eq!(bpmn.out_degree(diverging), 3, "fans out to all 3 children");
    }

    #[test]
    fn choice_graph_exclusive_choice_exports_exclusive_gateways_with_right_fan_out() {
        let cg = ChoiceGraphNode::exclusive_choice(vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(Some("b".into())),
        ]);
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 2);
        // start_gw/end_gw for the choice graph itself, plus a before/after pair per child (2
        // children) = 2 + 4 = 6 exclusive gateways, no parallel gateways.
        assert_eq!(bpmn.gateway_count(GatewayKind::Exclusive), 6);
        assert_eq!(bpmn.gateway_count(GatewayKind::Parallel), 0);
    }

    #[test]
    fn concurrency_operator_exports_parallel_gateway() {
        let mut op = PowlOperator::new(OperatorType::Concurrency);
        op.children.push(PowlNode::new_leaf(Some("a".into())));
        op.children.push(PowlNode::new_leaf(Some("b".into())));
        let model = Powl::new(PowlNode::Operator(op));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 2);
        assert_eq!(bpmn.gateway_count(GatewayKind::Exclusive), 0);
        assert_eq!(
            bpmn.gateway_count(GatewayKind::Parallel),
            2,
            "one diverging + one converging parallel gateway"
        );

        let diverging = bpmn
            .nodes()
            .iter()
            .find(|(_, n)| matches!(n, BpmnNode::Gateway(GatewayKind::Parallel)))
            .map(|&(id, _)| id)
            .unwrap();
        assert_eq!(bpmn.out_degree(diverging), 2);
    }

    #[test]
    fn partial_order_exports_parallel_gateways_and_direct_order_edges_only() {
        let children = vec![
            PowlNode::new_leaf(Some("a".into())),
            PowlNode::new_leaf(Some("b".into())),
            PowlNode::new_leaf(Some("c".into())),
        ];
        // a -> b -> c (closes a -> c by transitivity): only 2 direct edges must be drawn, not 3.
        let po = PartialOrderNode::new(children, [(0, 1), (1, 2)]);
        assert!(po.order.contains(&(0, 2)), "sanity: transitive closure ran");

        let model = Powl::new(PowlNode::PartialOrder(po));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 3);
        assert_eq!(bpmn.gateway_count(GatewayKind::Parallel), 2);
        // start->div, div->a (a has no predecessor), a->b, b->c, c->conv (c has no successor),
        // conv->end = 6 flows total -- confirms the transitively-implied a->c edge was NOT drawn.
        assert_eq!(bpmn.flow_count(), 6);
    }

    #[test]
    fn loop_operator_exports_a_real_back_edge() {
        let mut op = PowlOperator::new(OperatorType::Loop);
        op.children.push(PowlNode::new_leaf(Some("do".into())));
        op.children.push(PowlNode::new_leaf(Some("redo".into())));
        let model = Powl::new(PowlNode::Operator(op));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 2);
        assert_eq!(bpmn.gateway_count(GatewayKind::Exclusive), 2);

        let redo_task = bpmn
            .nodes()
            .iter()
            .find(|(_, n)| matches!(n, BpmnNode::Task(Some(l)) if l == "redo"))
            .map(|&(id, _)| id)
            .unwrap();
        let loop_start = bpmn
            .nodes()
            .iter()
            .find(|(_, n)| matches!(n, BpmnNode::Gateway(GatewayKind::Exclusive)))
            .map(|&(id, _)| id)
            .unwrap();
        // The redo task's exit must flow back into the loop-start gateway -- a genuine cycle in
        // the flow graph, not just a linear chain.
        assert!(bpmn
            .flows()
            .iter()
            .any(|f| f.from != loop_start && bpmn.out_degree(f.from) >= 1 && f.to == loop_start && {
                // Confirm this edge originates from (a descendant of) the redo task specifically.
                f.from == redo_task
            }));
    }

    #[test]
    fn choice_graph_self_loop_exports_a_real_cycle_around_one_child() {
        let cg = ChoiceGraphNode::self_looping(PowlNode::new_leaf(Some("a".into())));
        let model = Powl::new(PowlNode::ChoiceGraph(cg));
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 1);
        let a_task = bpmn
            .nodes()
            .iter()
            .find(|(_, n)| matches!(n, BpmnNode::Task(Some(l)) if l == "a"))
            .map(|&(id, _)| id)
            .unwrap();
        // "a"'s after-gateway must flow back into "a"'s own before-gateway -- the self-loop edge
        // survived translation as a real cycle, not silently dropped.
        let after_a: Vec<BpmnNodeId> = bpmn
            .flows()
            .iter()
            .filter(|f| f.from != a_task && bpmn.flows().iter().any(|g| g.from == a_task && g.to == f.from))
            .map(|f| f.from)
            .collect();
        assert!(!after_a.is_empty(), "expected a's after-gateway to be reachable");
        let cycles_back = after_a.iter().any(|&ag| {
            bpmn.flows()
                .iter()
                .any(|f| f.from == ag && bpmn.flows().iter().any(|g| g.from == f.to && g.to == a_task))
        });
        assert!(cycles_back, "expected a real back-edge around the self-looping child");
    }

    #[test]
    fn to_xml_string_produces_real_well_formed_xml_with_expected_element_counts() {
        let mut choice = PowlOperator::new(OperatorType::ExclusiveChoice);
        choice.children.push(PowlNode::new_leaf(Some("a".into())));
        choice.children.push(PowlNode::new_leaf(Some("b".into())));
        let model = Powl::new(PowlNode::Operator(choice));
        let bpmn = model.to_bpmn();
        let xml = bpmn.to_xml_string().expect("real quick-xml Writer must not fail on a Vec<u8> sink");

        let elements = assert_well_formed_xml(&xml);
        assert_eq!(elements.iter().filter(|e| e.as_str() == "task").count(), 2);
        assert_eq!(elements.iter().filter(|e| e.as_str() == "startEvent").count(), 1);
        assert_eq!(elements.iter().filter(|e| e.as_str() == "endEvent").count(), 1);
        assert_eq!(
            elements.iter().filter(|e| e.as_str() == "exclusiveGateway").count(),
            2
        );
        assert_eq!(
            elements.iter().filter(|e| e.as_str() == "sequenceFlow").count(),
            bpmn.flow_count()
        );
        assert!(xml.contains("<definitions"));
        assert!(xml.contains("</definitions>"));
    }

    #[test]
    fn escaping_is_real_not_substring_matching() {
        // A label containing every XML-special character must round-trip through a real parser,
        // not just "look right" as a substring of the raw output.
        let label = "A & B <weird> \"quoted\" 'activity'";
        let leaf = PowlNode::new_leaf(Some(label.to_string()));
        let model = Powl::new(leaf);
        let bpmn = model.to_bpmn();
        let xml = bpmn.to_xml_string().unwrap();

        // The raw serialized form must NOT contain the literal unescaped special characters next
        // to the name attribute value's quote -- i.e. real escaping happened, not string paste.
        assert!(xml.contains("&amp;"), "raw XML must contain the escaped ampersand: {xml}");

        // And a real parser must recover the exact original label.
        let mut reader = Reader::from_str(&xml);
        reader.config_mut().trim_text(true);
        let mut recovered_name = None;
        loop {
            match reader.read_event() {
                Ok(Event::Eof) => break,
                Ok(Event::Empty(e)) if e.name().as_ref() == b"task" => {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"name" {
                            recovered_name =
                                Some(attr.unescape_value().expect("valid escaped attribute value").into_owned());
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => panic!("real XML parser rejected escaped output: {e}\n---\n{xml}"),
            }
        }
        assert_eq!(recovered_name.as_deref(), Some(label));
    }

    #[test]
    fn to_bpmn_expands_frequency_tags_before_translating() {
        // A skippable+repeatable leaf must translate through the same expand_frequency_tags path
        // to_petri_net uses -- i.e. as a real ChoiceGraph do/redo loop, not a bare unadorned task
        // with its multiplicity silently dropped.
        let mut leaf = PowlNode::new_leaf(Some("a".into()));
        leaf.set_freq(super::super::Freq::new(0, None));
        let model = Powl::new(leaf);
        let bpmn = model.to_bpmn();

        assert_eq!(bpmn.task_count(), 2, "the real 'a' task plus the silent do-part task");
        assert!(
            bpmn.gateway_count(GatewayKind::Exclusive) >= 2,
            "expansion wraps the leaf in a ChoiceGraph do/redo loop with real gateway structure"
        );
    }
}
