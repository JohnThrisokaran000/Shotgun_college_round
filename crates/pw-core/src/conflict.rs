//! Conflict detection on resolved elements. Targets are element indices from a
//! snapshot, never selector strings: `.btn` and `#buy` may be the same element.
//!
//! Checks:
//! - write-write: two patches write the same property of one node with different values (hard)
//! - behaviour: two behaviour ops on one node and event (hard)
//! - structural: one patch moves a node that another patch targets (hard)
//! - anchor dependency: one patch changes text or an attribute another patch's fingerprint relies on (soft)
//! - order dependence: applying A then B gives a different tree than B then A (hard).
//!   Simulated on an abstract tree, not in a browser. This is a stronger claim than the
//!   plan's empirical check, and it only covers what the simulator models.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::anchor::Fingerprint;
use crate::patch::Op;
use crate::snapshot::Snapshot;

#[derive(Clone, Debug)]
pub struct Target {
    pub patch: String,
    pub node: u32,
    pub fingerprint: Fingerprint,
    pub ops: Vec<Op>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    WriteWrite,
    Behaviour,
    Structural,
    AnchorDependency,
    OrderDependence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Conflict {
    pub kind: Kind,
    pub hard: bool,
    pub patches: [String; 2],
    pub node: u32,
    pub detail: String,
}

/// A property write: (key, value, is_behaviour). Keys are grouped so that
/// `setText` and `setHTML` on one node count as writing the same content.
fn writes(op: &Op) -> Vec<(String, String, bool)> {
    let w = |k: &str, v: &str, b: bool| vec![(k.to_string(), v.to_string(), b)];
    match op {
        Op::SetText { value } | Op::SetHtml { value } => w("content", value, false),
        Op::SetAttr { name, value } => w(&format!("attr:{name}"), value, false),
        Op::RemoveAttr { name } => w(&format!("attr:{name}"), "<removed>", false),
        Op::SetStyle { prop, value } => w(&format!("style:{prop}"), value, false),
        Op::AddClass { name } => w(&format!("class:{name}"), "add", false),
        Op::RemoveClass { name } => w(&format!("class:{name}"), "remove", false),
        Op::Hide {} => w("visibility", "hidden", false),
        Op::Show {} => w("visibility", "visible", false),
        Op::Disable {} => w("disabled", "true", false),
        Op::Enable {} => w("disabled", "false", false),
        Op::OnClickNavigate { href } => w("click", &format!("navigate:{href}"), true),
        Op::OnClickSubmit {} => w("click", "submit", true),
        Op::OnClickFetch { url } => w("click", &format!("fetch:{url}"), true),
        Op::MoveBefore { .. } | Op::MoveAfter { .. } | Op::InsertBefore { .. } | Op::InsertAfter { .. } => vec![],
    }
}

fn is_move(op: &Op) -> bool {
    matches!(op, Op::MoveBefore { .. } | Op::MoveAfter { .. })
}

fn pair(a: &str, b: &str) -> [String; 2] {
    if a <= b {
        [a.to_string(), b.to_string()]
    } else {
        [b.to_string(), a.to_string()]
    }
}

/// Abstract document: parent -> ordered children, and per-node properties.
#[derive(Clone, Debug, PartialEq)]
struct Tree {
    children: BTreeMap<u32, Vec<u32>>,
    parent: BTreeMap<u32, u32>,
    props: BTreeMap<(u32, String), String>,
}

impl Tree {
    fn from_snapshot(snap: &Snapshot) -> Tree {
        let mut children: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        let mut parent = BTreeMap::new();
        for el in snap.elements() {
            if let Some(p) = el.parent {
                children.entry(p).or_default().push(el.index);
                parent.insert(el.index, p);
            }
        }
        Tree { children, parent, props: BTreeMap::new() }
    }

    fn detach(&mut self, node: u32) {
        if let Some(p) = self.parent.remove(&node) {
            if let Some(list) = self.children.get_mut(&p) {
                list.retain(|&c| c != node);
            }
        }
    }

    fn insert_at(&mut self, node: u32, reference: u32, after: bool) {
        let Some(&p) = self.parent.get(&reference) else { return };
        let list = self.children.entry(p).or_default();
        let pos = list.iter().position(|&c| c == reference).map_or(list.len(), |i| if after { i + 1 } else { i });
        list.insert(pos, node);
        self.parent.insert(node, p);
    }

    fn apply(&mut self, snap: &Snapshot, target: &Target) {
        for (i, op) in target.ops.iter().enumerate() {
            for (key, value, _) in writes(op) {
                self.props.insert((target.node, key), value);
            }
            match op {
                Op::MoveBefore { target: css } | Op::MoveAfter { target: css } => {
                    if let Some(reference) = snap.css(css).ok().and_then(|h| h.first().copied()) {
                        if reference != target.node {
                            self.detach(target.node);
                            self.insert_at(target.node, reference, matches!(op, Op::MoveAfter { .. }));
                        }
                    }
                }
                Op::InsertBefore { target: css, html } | Op::InsertAfter { target: css, html } => {
                    if let Some(reference) = snap.css(css).ok().and_then(|h| h.first().copied()) {
                        let synthetic = synthetic_id(&target.patch, i);
                        self.props.insert((synthetic, "content".into()), html.clone());
                        self.insert_at(synthetic, reference, matches!(op, Op::InsertAfter { .. }));
                    }
                }
                _ => {}
            }
        }
    }
}

/// Stable id for an inserted node: high bit set so it never collides with real indices.
fn synthetic_id(patch: &str, op_index: usize) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in patch.bytes().chain(op_index.to_le_bytes()) {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h | 0x8000_0000
}

fn apply_order(snap: &Snapshot, first: &Target, second: &Target) -> Tree {
    let mut tree = Tree::from_snapshot(snap);
    tree.apply(snap, first);
    tree.apply(snap, second);
    tree
}

fn has_structure(t: &Target) -> bool {
    t.ops.iter().any(|op| is_move(op) || matches!(op, Op::InsertBefore { .. } | Op::InsertAfter { .. }))
}

pub fn detect(targets: &[Target], snap: &Snapshot) -> Vec<Conflict> {
    let mut out = Vec::new();
    for (i, a) in targets.iter().enumerate() {
        for b in &targets[i + 1..] {
            if a.patch == b.patch {
                continue;
            }
            let names = pair(&a.patch, &b.patch);
            let mut flagged_same_node = false;

            // write-write and behaviour: same node, same key, different value
            if a.node == b.node {
                let wa: Vec<(String, String, bool)> = a.ops.iter().flat_map(writes).collect();
                let wb: Vec<(String, String, bool)> = b.ops.iter().flat_map(writes).collect();
                for (ka, va, ba) in &wa {
                    for (kb, vb, _) in &wb {
                        if ka == kb && va != vb {
                            flagged_same_node = true;
                            out.push(Conflict {
                                kind: if *ba { Kind::Behaviour } else { Kind::WriteWrite },
                                hard: true,
                                patches: names.clone(),
                                node: a.node,
                                detail: format!("both write `{ka}` with different values (`{va}` vs `{vb}`)"),
                            });
                        }
                    }
                }
            }

            // structural: one patch moves the node the other targets
            for (mover, other) in [(a, b), (b, a)] {
                let moves = mover.ops.iter().any(is_move);
                if moves && other.node == mover.node {
                    out.push(Conflict {
                        kind: Kind::Structural,
                        hard: true,
                        patches: names.clone(),
                        node: mover.node,
                        detail: format!("`{}` moves the node that `{}` targets", mover.patch, other.patch),
                    });
                }
            }

            // anchor dependency: a text or attribute change breaks the other fingerprint
            for (writer, dependent) in [(a, b), (b, a)] {
                if writer.node != dependent.node {
                    continue;
                }
                let Some(old) = snap.element(writer.node) else { continue };
                for op in &writer.ops {
                    let changes_text = matches!(op, Op::SetText { .. } | Op::SetHtml { .. });
                    if changes_text && dependent.fingerprint.text.as_deref().is_some_and(|t| t == old.text) {
                        out.push(Conflict {
                            kind: Kind::AnchorDependency,
                            hard: false,
                            patches: names.clone(),
                            node: writer.node,
                            detail: format!("`{}` changes text that `{}` relies on; re-fingerprint after ordering", writer.patch, dependent.patch),
                        });
                    }
                    if let Op::SetAttr { name, .. } = op {
                        let stored = dependent.fingerprint.attrs.get(name);
                        if stored.is_some() && stored == old.attrs.get(name) {
                            out.push(Conflict {
                                kind: Kind::AnchorDependency,
                                hard: false,
                                patches: names.clone(),
                                node: writer.node,
                                detail: format!("`{}` changes attribute `{name}` that `{}` relies on", writer.patch, dependent.patch),
                            });
                        }
                    }
                }
            }

            // order dependence: only when structure is involved or the pair shares a node
            // and was not already flagged as a write conflict
            let relevant = has_structure(a) || has_structure(b) || (a.node == b.node && !flagged_same_node);
            if relevant && apply_order(snap, a, b) != apply_order(snap, b, a) {
                out.push(Conflict {
                    kind: Kind::OrderDependence,
                    hard: true,
                    patches: names.clone(),
                    node: a.node,
                    detail: "applying them in either order gives different results".into(),
                });
            }
        }
    }
    dedupe(out)
}

fn dedupe(conflicts: Vec<Conflict>) -> Vec<Conflict> {
    let mut seen: HashMap<String, ()> = HashMap::new();
    conflicts
        .into_iter()
        .filter(|c| seen.insert(format!("{:?}|{:?}|{}|{}", c.kind, c.patches, c.node, c.detail), ()).is_none())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Snapshot;

    const PAGE: &str = r#"<html><body>
        <section id="cart"><p id="total">Order total</p><button id="buy" class="btn">Buy now</button><button id="x">X</button><button id="y">Y</button><button id="z">Z</button></section>
        </body></html>"#;

    fn node(snap: &Snapshot, id: &str) -> u32 {
        snap.css(&format!("#{id}")).unwrap()[0]
    }

    fn target(snap: &Snapshot, patch: &str, id: &str, ops: Vec<Op>) -> Target {
        let el = snap.element(node(snap, id)).unwrap();
        Target { patch: patch.into(), node: el.index, fingerprint: Fingerprint::from_element(el), ops }
    }

    fn style(prop: &str, value: &str) -> Op {
        Op::SetStyle { prop: prop.into(), value: value.into() }
    }

    fn kinds(conflicts: &[Conflict]) -> Vec<Kind> {
        conflicts.iter().map(|c| c.kind).collect()
    }

    #[test]
    fn same_property_different_values_is_a_hard_write_write_conflict() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![style("background-color", "#0a7d32")]),
            target(&snap, "dev-b", "buy", vec![style("background-color", "#c62828")]),
        ];
        let found = detect(&targets, &snap);
        assert_eq!(kinds(&found), vec![Kind::WriteWrite]);
        assert!(found[0].hard);
    }

    #[test]
    fn identical_changes_do_not_conflict() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![style("background-color", "#0a7d32")]),
            target(&snap, "dev-b", "buy", vec![style("background-color", "#0a7d32")]),
        ];
        assert!(detect(&targets, &snap).is_empty());
    }

    #[test]
    fn different_properties_on_one_node_do_not_conflict() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![style("background-color", "#0a7d32")]),
            target(&snap, "dev-b", "buy", vec![style("color", "#ffffff")]),
        ];
        assert!(detect(&targets, &snap).is_empty());
    }

    #[test]
    fn two_behaviours_on_one_click_are_a_hard_conflict() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![Op::OnClickNavigate { href: "/a".into() }]),
            target(&snap, "dev-b", "buy", vec![Op::OnClickNavigate { href: "/b".into() }]),
        ];
        let found = detect(&targets, &snap);
        assert_eq!(kinds(&found), vec![Kind::Behaviour]);
    }

    #[test]
    fn moving_a_node_another_patch_targets_is_structural() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![Op::MoveBefore { target: "#x".into() }]),
            target(&snap, "dev-b", "buy", vec![style("color", "#fff")]),
        ];
        assert!(kinds(&detect(&targets, &snap)).contains(&Kind::Structural));
    }

    #[test]
    fn changing_text_that_another_fingerprint_relies_on_is_soft() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![Op::SetText { value: "Place order".into() }]),
            target(&snap, "dev-b", "buy", vec![style("color", "#fff")]),
        ];
        let found = detect(&targets, &snap);
        assert_eq!(kinds(&found), vec![Kind::AnchorDependency]);
        assert!(!found[0].hard);
    }

    #[test]
    fn order_dependent_moves_are_detected_by_simulation() {
        // X and Y both move before Z. Final order depends on which goes first.
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "x", vec![Op::MoveBefore { target: "#z".into() }]),
            target(&snap, "dev-b", "y", vec![Op::MoveBefore { target: "#z".into() }]),
        ];
        let found = detect(&targets, &snap);
        assert!(kinds(&found).contains(&Kind::OrderDependence), "{found:?}");
    }

    #[test]
    fn independent_moves_are_not_order_dependent() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "x", vec![Op::MoveBefore { target: "#y".into() }]),
            target(&snap, "dev-b", "buy", vec![style("color", "#fff")]),
        ];
        assert!(!kinds(&detect(&targets, &snap)).contains(&Kind::OrderDependence));
    }

    #[test]
    fn a_patch_never_conflicts_with_itself() {
        let snap = Snapshot::parse(PAGE);
        let targets = vec![
            target(&snap, "dev-a", "buy", vec![style("color", "#000")]),
            target(&snap, "dev-a", "buy", vec![style("color", "#fff")]),
        ];
        assert!(detect(&targets, &snap).is_empty());
    }
}
