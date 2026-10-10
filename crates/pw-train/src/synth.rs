//! Synthetic web pages with controlled mutations. Used to make RELOCATE and AUTHOR training
//! examples and the held-out benchmark. Everything is seeded, so the same seed always gives the
//! same dataset.
//!
//! Eight page kinds. Four are storefront and account pages. The other four (table, search
//! results, settings form, modal dialog) repeat identical controls on purpose, so the target can
//! only be told apart by its context. Sixteen mutations change the gold element's identity,
//! its text, its position, its tag, or the page around it. "Twin" adds an identical copy of the
//! target, and that case is labelled as a refusal, because nothing separates the two.

/// xorshift64*: small, deterministic, no dependencies.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    pub fn chance(&mut self, p: f64) -> bool {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }

    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub tag: String,
    pub attrs: Vec<(String, String)>,
    pub text: String,
    pub children: Vec<Node>,
    pub gold: bool,
}

const VOID: &[&str] = &["input", "img", "br", "hr", "meta", "link"];

impl Node {
    pub fn new(tag: &str) -> Node {
        Node { tag: tag.into(), attrs: Vec::new(), text: String::new(), children: Vec::new(), gold: false }
    }

    pub fn with_text(tag: &str, text: &str) -> Node {
        let mut n = Node::new(tag);
        n.text = text.into();
        n
    }

    pub fn attr(mut self, name: &str, value: &str) -> Node {
        self.attrs.push((name.into(), value.into()));
        self
    }

    pub fn child(mut self, c: Node) -> Node {
        self.children.push(c);
        self
    }

    fn set_attr(&mut self, name: &str, value: Option<&str>) {
        self.attrs.retain(|(k, _)| k != name);
        if let Some(v) = value {
            self.attrs.push((name.into(), v.into()));
        }
    }

    /// Serialise to HTML. The gold node gets `data-pwgold`; the marker is never part of a prompt.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.render_into(&mut out);
        out
    }

    fn render_into(&self, out: &mut String) {
        out.push('<');
        out.push_str(&self.tag);
        for (k, v) in &self.attrs {
            out.push(' ');
            out.push_str(k);
            out.push_str("=\"");
            out.push_str(&escape(v, true));
            out.push('"');
        }
        if self.gold {
            out.push_str(" data-pwgold=\"1\"");
        }
        out.push('>');
        if VOID.contains(&self.tag.as_str()) {
            return;
        }
        out.push_str(&escape(&self.text, false));
        for c in &self.children {
            c.render_into(out);
        }
        out.push_str("</");
        out.push_str(&self.tag);
        out.push('>');
    }

    /// Visit every node mutably, depth first.
    fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Node)) {
        f(self);
        for c in &mut self.children {
            c.visit_mut(f);
        }
    }

    fn count_gold(&self) -> usize {
        usize::from(self.gold) + self.children.iter().map(Node::count_gold).sum::<usize>()
    }
}

fn escape(s: &str, attr: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if attr => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}

pub const CLASS_VOCAB: &[&str] = &[
    "btn", "btn-primary", "btn-secondary", "cta", "primary-action", "action", "button-lg", "wrap", "row", "col",
    "card", "panel", "section", "cart", "actions", "footer-links", "hero", "toolbar", "stack", "grid", "item", "box",
];

pub const CONTAINER_TAGS: &[&str] = &["div", "section", "aside", "footer", "nav", "main"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    Checkout,
    Product,
    Login,
    Dashboard,
    Table,
    Search,
    Settings,
    Modal,
}

impl SiteKind {
    pub const ALL: [SiteKind; 8] = [
        SiteKind::Checkout,
        SiteKind::Product,
        SiteKind::Login,
        SiteKind::Dashboard,
        SiteKind::Table,
        SiteKind::Search,
        SiteKind::Settings,
        SiteKind::Modal,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SiteKind::Checkout => "checkout",
            SiteKind::Product => "product",
            SiteKind::Login => "login",
            SiteKind::Dashboard => "dashboard",
            SiteKind::Table => "table",
            SiteKind::Search => "search",
            SiteKind::Settings => "settings",
            SiteKind::Modal => "modal",
        }
    }

    fn title(self) -> &'static str {
        match self {
            SiteKind::Checkout => "Your cart",
            SiteKind::Product => "Trail running shoe",
            SiteKind::Login => "Sign in to continue",
            SiteKind::Dashboard => "Overview",
            SiteKind::Table => "Orders",
            SiteKind::Search => "Results for trail shoes",
            SiteKind::Settings => "Notifications",
            SiteKind::Modal => "Projects",
        }
    }

    fn primary(self) -> &'static [&'static str] {
        match self {
            SiteKind::Checkout => &["Buy now", "Place order", "Checkout", "Pay now"],
            SiteKind::Product => &["Add to cart", "Add to bag", "Buy now"],
            SiteKind::Login => &["Sign in", "Log in", "Continue"],
            SiteKind::Dashboard => &["Export", "Save changes", "Publish"],
            SiteKind::Table => &["Edit"],
            SiteKind::Search => &["Add to cart", "Add to bag"],
            SiteKind::Settings => &["Save preferences", "Save changes"],
            SiteKind::Modal => &["Delete"],
        }
    }

    fn secondary(self) -> &'static [&'static str] {
        match self {
            SiteKind::Checkout => &["Cancel", "Apply coupon", "Remove", "Continue shopping"],
            SiteKind::Product => &["View details", "Compare", "Add to wishlist"],
            SiteKind::Login => &["Forgot password?", "Create account", "Help"],
            SiteKind::Dashboard => &["Refresh", "Filter", "Share"],
            SiteKind::Table => &["Delete", "View", "Archive"],
            SiteKind::Search => &["Save for later", "Compare"],
            SiteKind::Settings => &["Reset to defaults", "Cancel"],
            SiteKind::Modal => &["Cancel"],
        }
    }

    fn labels(self) -> &'static [&'static str] {
        match self {
            SiteKind::Checkout => &["Order total", "Subtotal", "Shipping", "Tax", "Apply coupon"],
            SiteKind::Product => &["In stock", "Free shipping", "Reviews", "Size", "Color"],
            SiteKind::Login => &["Email", "Password", "Remember me"],
            SiteKind::Dashboard => &["Revenue", "Orders", "Visitors", "Conversion"],
            SiteKind::Table => &["Order", "Status", "Total"],
            SiteKind::Search => &["In stock", "Free returns", "Ships today"],
            SiteKind::Settings => &["Email me", "Text me", "Weekly digest"],
            SiteKind::Modal => &["This cannot be undone"],
        }
    }
}

/// Alternative wordings a site might change a button to (the "retext" mutation).
pub fn synonyms(text: &str) -> &'static [&'static str] {
    match text {
        "Buy now" => &["Place order", "Checkout now", "Pay now"],
        "Add to cart" => &["Add to bag", "Add to basket"],
        "Sign in" => &["Log in", "Continue"],
        "Export" => &["Download CSV", "Export data"],
        "Save changes" => &["Save", "Apply changes"],
        "Place order" => &["Buy now", "Confirm order"],
        "Save preferences" => &["Save", "Update preferences"],
        "Edit" => &["Modify", "Change"],
        "Delete" => &["Remove", "Delete item"],
        _ => &[],
    }
}

fn random_classes(rng: &mut Rng) -> String {
    let n = 1 + rng.below(3);
    let mut parts: Vec<&str> = Vec::new();
    for _ in 0..n {
        let c = *rng.pick(CLASS_VOCAB);
        if !parts.contains(&c) {
            parts.push(c);
        }
    }
    parts.join(" ")
}

fn container(rng: &mut Rng) -> Node {
    let tag = *rng.pick(CONTAINER_TAGS);
    let mut n = Node::new(tag).attr("class", &random_classes(rng));
    if rng.chance(0.3) {
        n = n.attr("id", &format!("c{}", rng.below(10_000)));
    }
    n
}

fn action(rng: &mut Rng, text: &str, tag: &str) -> Node {
    let classes = random_classes(rng);
    let mut n = match tag {
        "a" => Node::with_text("a", text).attr("href", "/next"),
        "input" => Node::new("input").attr("type", "submit").attr("value", text),
        _ => Node::with_text("button", text).attr("type", "submit"),
    };
    n = n.attr("class", &classes);
    if rng.chance(0.4) {
        n = n.attr("id", &format!("btn-{}", rng.below(100_000)));
    }
    if rng.chance(0.2) {
        n = n.attr("data-testid", &format!("action-{}", rng.below(1000)));
    }
    n
}

/// Wrap page content in the shared chrome: a heading, a main section, and a footer link.
fn page(title: &str, content: Node) -> Node {
    let section = Node::new("section").attr("class", "content").child(content);
    let body = Node::new("body")
        .child(Node::with_text("h1", title))
        .child(Node::new("main").child(section))
        .child(Node::new("footer").attr("class", "footer").child(Node::with_text("a", "Help").attr("href", "/help")));
    Node::new("html").child(Node::new("head").child(Node::with_text("title", "Shop"))).child(body)
}

/// One page. The primary action is marked gold.
pub fn make_site(rng: &mut Rng, kind: SiteKind) -> Node {
    match kind {
        SiteKind::Table => return build_table(rng),
        SiteKind::Search => return build_search(rng),
        SiteKind::Settings => return build_settings(rng),
        SiteKind::Modal => return build_modal(rng),
        _ => {}
    }
    let mut body = Node::new("body");
    body = body.child(Node::with_text("h1", kind.title()));
    let mut main = Node::new("main");
    let mut section = Node::new("section").attr("class", &random_classes(rng));
    for label in kind.labels().iter().take(2 + rng.below(3)) {
        section = section.child(Node::with_text("p", label));
    }
    let tag = if rng.chance(0.1) { "a" } else if rng.chance(0.1) { "input" } else { "button" };
    let mut actions = container(rng);
    let primary_text = kind.primary()[rng.below(kind.primary().len())];
    let mut gold = action(rng, primary_text, tag);
    gold.gold = true;
    actions = actions.child(gold);
    for _ in 0..rng.below(3) {
        let text = kind.secondary()[rng.below(kind.secondary().len())];
        let secondary_tag = if rng.chance(0.2) { "a" } else { "button" };
        actions = actions.child(action(rng, text, secondary_tag));
    }
    // Sometimes nest the actions in a wrapper, so the gold node is not a direct child of the section.
    let wrapper_depth = rng.below(3);
    let mut inner = actions;
    for _ in 0..wrapper_depth {
        inner = container(rng).child(inner);
    }
    section = section.child(inner);
    for label in kind.labels().iter().skip(2).take(rng.below(3)) {
        section = section.child(Node::with_text("p", label));
    }
    main = main.child(section);
    body = body.child(main);
    body = body.child(Node::new("footer").attr("class", "footer").child(Node::with_text("a", "Help").attr("href", "/help")));
    let head = Node::new("head").child(Node::with_text("title", "Shop"));
    let html = Node::new("html").child(head).child(body);
    debug_assert_eq!(html.count_gold(), 1);
    html
}

/// Orders table: every row has an "Edit" button, and only one row's button is the target.
fn build_table(rng: &mut Rng) -> Node {
    let rows = 3 + rng.below(4);
    let gold_row = rng.below(rows);
    let mut tbody = Node::new("tbody");
    for r in 0..rows {
        let mut edit = action(rng, "Edit", "button");
        edit.gold = r == gold_row;
        let row = Node::new("tr")
            .child(Node::with_text("td", &format!("Order #{}", 1000 + rng.below(9000))))
            .child(Node::with_text("td", SiteKind::Table.labels()[rng.below(3)]))
            .child(Node::new("td").child(edit));
        tbody = tbody.child(row);
    }
    page(SiteKind::Table.title(), Node::new("table").attr("class", &random_classes(rng)).child(tbody))
}

/// Search results: every item has an "Add to cart" button, and only one item's button is the target.
fn build_search(rng: &mut Rng) -> Node {
    let items = 4 + rng.below(3);
    let gold_item = rng.below(items);
    let mut list = Node::new("ul").attr("class", &random_classes(rng));
    for i in 0..items {
        let mut add = action(rng, "Add to cart", "button");
        add.gold = i == gold_item;
        let item = Node::new("li")
            .attr("class", "item")
            .child(Node::with_text("h3", &format!("Trail shoe model {}", 1 + rng.below(40))))
            .child(Node::with_text("p", &format!("${}.{:02}", 40 + rng.below(160), rng.below(100))))
            .child(add);
        list = list.child(item);
    }
    page(SiteKind::Search.title(), list)
}

/// Settings form: a few checkboxes, then "Save preferences" (the target) and a reset button.
fn build_settings(rng: &mut Rng) -> Node {
    let mut form = Node::new("form").attr("action", "/settings");
    for label in SiteKind::Settings.labels().iter().take(2 + rng.below(2)) {
        form = form.child(Node::with_text("label", label).child(Node::new("input").attr("type", "checkbox")));
    }
    let label = SiteKind::Settings.primary()[rng.below(2)];
    let mut save = action(rng, label, "button");
    save.gold = true;
    let mut actions = Node::new("div").attr("class", &random_classes(rng)).child(save);
    if rng.chance(0.7) {
        actions = actions.child(action(rng, "Reset to defaults", "button"));
    }
    page(SiteKind::Settings.title(), form.child(actions))
}

/// Modal dialog: a page-level "Delete" outside the dialog, and the target "Delete" inside it,
/// next to the question that names the project.
fn build_modal(rng: &mut Rng) -> Node {
    let outside = action(rng, "Delete", "button");
    let project = ["Atlas", "Borealis", "Cinder"][rng.below(3)];
    let mut delete = action(rng, "Delete", "button");
    delete.gold = true;
    let dialog = Node::new("div")
        .attr("role", "dialog")
        .attr("class", &random_classes(rng))
        .child(Node::with_text("p", &format!("Delete project {project}?")))
        .child(delete)
        .child(action(rng, "Cancel", "button"));
    page(SiteKind::Modal.title(), Node::new("div").child(outside).child(dialog))
}

/// Mutations that change how the gold element looks, where it sits, or what surrounds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutation {
    RenameClasses,
    DropIds,
    WrapGold,
    MoveGold,
    Retext,
    AddDecoy,
    DeleteGold,
    ShuffleSiblings,
    AddNoise,
    AddTwin,
    ChangeTag,
    AddAriaLabel,
    WrapInForm,
    HashClasses,
    UppercaseText,
    IconPrefix,
}

impl Mutation {
    pub const ALL: [Mutation; 16] = [
        Mutation::RenameClasses,
        Mutation::DropIds,
        Mutation::WrapGold,
        Mutation::MoveGold,
        Mutation::Retext,
        Mutation::AddDecoy,
        Mutation::DeleteGold,
        Mutation::ShuffleSiblings,
        Mutation::AddNoise,
        Mutation::AddTwin,
        Mutation::ChangeTag,
        Mutation::AddAriaLabel,
        Mutation::WrapInForm,
        Mutation::HashClasses,
        Mutation::UppercaseText,
        Mutation::IconPrefix,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Mutation::RenameClasses => "rename-classes",
            Mutation::DropIds => "drop-ids",
            Mutation::WrapGold => "wrap-gold",
            Mutation::MoveGold => "move-gold",
            Mutation::Retext => "retext",
            Mutation::AddDecoy => "add-decoy",
            Mutation::DeleteGold => "delete-gold",
            Mutation::ShuffleSiblings => "shuffle-siblings",
            Mutation::AddNoise => "add-noise",
            Mutation::AddTwin => "add-twin",
            Mutation::ChangeTag => "change-tag",
            Mutation::AddAriaLabel => "add-aria-label",
            Mutation::WrapInForm => "wrap-in-form",
            Mutation::HashClasses => "hash-classes",
            Mutation::UppercaseText => "uppercase-text",
            Mutation::IconPrefix => "icon-prefix",
        }
    }
}

fn find_path_to_gold(node: &Node, path: &mut Vec<usize>) -> bool {
    if node.gold {
        return true;
    }
    for (i, c) in node.children.iter().enumerate() {
        path.push(i);
        if find_path_to_gold(c, path) {
            return true;
        }
        path.pop();
    }
    false
}

fn node_at_mut<'a>(root: &'a mut Node, path: &[usize]) -> &'a mut Node {
    let mut cur = root;
    for &i in path {
        cur = &mut cur.children[i];
    }
    cur
}

/// Remove the gold node, returning it, and the path of its parent. Caller reinserts.
fn take_gold(root: &mut Node) -> Option<(Vec<usize>, Node)> {
    let mut path = Vec::new();
    if !find_path_to_gold(root, &mut path) || path.is_empty() {
        return None;
    }
    let idx = path.pop()?;
    let parent = node_at_mut(root, &path);
    let gold = parent.children.remove(idx);
    Some((path, gold))
}

/// Paths (child indices from the root) of every container element, document order.
fn container_paths(root: &Node) -> Vec<Vec<usize>> {
    fn walk(n: &Node, path: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if CONTAINER_TAGS.contains(&n.tag.as_str()) {
            out.push(path.clone());
        }
        for (i, c) in n.children.iter().enumerate() {
            path.push(i);
            walk(c, path, out);
            path.pop();
        }
    }
    let mut out = Vec::new();
    walk(root, &mut Vec::new(), &mut out);
    out
}

/// Swap the first and last child of some containers. Walks the tree directly: swapping moves
/// subtrees, so paths collected beforehand would go stale.
fn shuffle_containers(n: &mut Node, rng: &mut Rng) -> bool {
    let mut done = false;
    if CONTAINER_TAGS.contains(&n.tag.as_str()) && n.children.len() > 1 && rng.chance(0.5) {
        let last = n.children.len() - 1;
        n.children.swap(0, last);
        done = true;
    }
    for c in n.children.iter_mut() {
        done |= shuffle_containers(c, rng);
    }
    done
}

/// Apply one mutation in place. Returns false when it had nothing to act on.
pub fn apply(rng: &mut Rng, root: &mut Node, m: Mutation) -> bool {
    match m {
        Mutation::RenameClasses => {
            root.visit_mut(&mut |n| {
                if n.attrs.iter().any(|(k, _)| k == "class") && rng.chance(0.8) {
                    let fresh = random_classes(rng);
                    n.set_attr("class", Some(&fresh));
                }
            });
            true
        }
        Mutation::DropIds => {
            root.visit_mut(&mut |n| {
                if n.tag != "html" {
                    n.set_attr("id", None);
                }
            });
            true
        }
        Mutation::WrapGold => {
            let mut path = Vec::new();
            if !find_path_to_gold(root, &mut path) {
                return false;
            }
            let Some((parent_path, gold)) = take_gold(root) else { return false };
            let mut wrapped = container(rng).child(gold);
            if rng.chance(0.5) {
                wrapped = container(rng).child(wrapped);
            }
            node_at_mut(root, &parent_path).children.push(wrapped);
            true
        }
        Mutation::MoveGold => {
            let Some((_, gold)) = take_gold(root) else { return false };
            let paths = container_paths(root);
            if paths.is_empty() {
                return false;
            }
            let target = node_at_mut(root, &paths[rng.below(paths.len())]);
            let at = rng.below(target.children.len() + 1);
            target.children.insert(at, gold);
            true
        }
        Mutation::Retext => {
            let mut changed = false;
            root.visit_mut(&mut |n| {
                if n.gold && !changed {
                    let options = synonyms(&n.text);
                    if !options.is_empty() {
                        n.text = (*rng.pick(options)).to_string();
                        changed = true;
                    }
                }
            });
            changed
        }
        Mutation::AddDecoy => {
            // A look-alike with the same text and classes, placed in another container.
            let mut gold_copy: Option<Node> = None;
            root.visit_mut(&mut |n| {
                if n.gold && gold_copy.is_none() {
                    let mut c = n.clone();
                    c.gold = false;
                    c.set_attr("id", None);
                    c.set_attr("data-testid", None);
                    gold_copy = Some(c);
                }
            });
            let Some(copy) = gold_copy else { return false };
            let paths = container_paths(root);
            if paths.is_empty() {
                return false;
            }
            node_at_mut(root, &paths[rng.below(paths.len())]).children.push(copy);
            true
        }
        Mutation::DeleteGold => take_gold(root).is_some(),
        Mutation::ShuffleSiblings => shuffle_containers(root, rng),
        Mutation::AddNoise => {
            let paths = container_paths(root);
            if paths.is_empty() {
                return false;
            }
            let label = ["Free returns", "Secure checkout", "Gift wrap", "Estimated delivery"][rng.below(4)];
            node_at_mut(root, &paths[rng.below(paths.len())]).children.push(Node::with_text("p", label));
            true
        }
        Mutation::AddTwin => {
            // An identical sibling of the target: same tag, text and classes, no id. Nothing on the
            // page tells the two apart, so the right answer is a refusal (see datagen).
            let mut path = Vec::new();
            if !find_path_to_gold(root, &mut path) || path.is_empty() {
                return false;
            }
            let idx = path.pop().unwrap_or(0);
            let parent = node_at_mut(root, &path);
            let mut twin = parent.children[idx].clone();
            twin.gold = false;
            twin.set_attr("id", None);
            twin.set_attr("data-testid", None);
            parent.children.insert(idx + 1, twin);
            true
        }
        Mutation::ChangeTag => {
            let mut changed = false;
            root.visit_mut(&mut |n| {
                if n.gold && !changed && (n.tag == "button" || n.tag == "a") {
                    let (tag, role) = if n.tag == "button" { ("div", "button") } else { ("span", "link") };
                    n.tag = tag.into();
                    n.set_attr("role", Some(role));
                    changed = true;
                }
            });
            changed
        }
        Mutation::AddAriaLabel => {
            let mut changed = false;
            root.visit_mut(&mut |n| {
                if n.gold && !changed {
                    let text = n.text.clone();
                    n.set_attr("aria-label", Some(text.as_str()));
                    changed = true;
                }
            });
            changed
        }
        Mutation::WrapInForm => {
            let Some((parent_path, gold)) = take_gold(root) else { return false };
            let form = Node::new("form").attr("class", &random_classes(rng)).child(gold);
            node_at_mut(root, &parent_path).children.push(form);
            true
        }
        Mutation::HashClasses => {
            // Build-tool class names, like CSS modules produce. They carry no meaning across releases.
            let mut changed = false;
            root.visit_mut(&mut |n| {
                if n.gold && !changed {
                    let hashed = format!("css-{:06x}", rng.below(0xFF_FFFF));
                    n.set_attr("class", Some(hashed.as_str()));
                    changed = true;
                }
            });
            changed
        }
        Mutation::UppercaseText => {
            let mut changed = false;
            root.visit_mut(&mut |n| {
                if n.gold && !changed && !n.text.is_empty() {
                    n.text = n.text.to_uppercase();
                    changed = true;
                }
            });
            changed
        }
        Mutation::IconPrefix => {
            let mut changed = false;
            root.visit_mut(&mut |n| {
                if n.gold && !changed {
                    n.children.insert(0, Node::with_text("span", "\u{2192}"));
                    changed = true;
                }
            });
            changed
        }
    }
}

/// A mutated page plus what was done to it.
pub struct Variant {
    pub html: String,
    pub mutations: Vec<Mutation>,
    /// False when the gold element was deleted.
    pub gold_present: bool,
}

/// Apply 1 to 3 mutations. Deleting the target is chosen with its own probability.
pub fn variant(rng: &mut Rng, base: &Node) -> Variant {
    let mut root = base.clone();
    let count = 1 + rng.below(3);
    let mut chosen = Vec::new();
    // Deleting the target is the answer `null`. Keep it near the ~12% the plan expects.
    let non_delete: Vec<Mutation> = Mutation::ALL.iter().copied().filter(|m| *m != Mutation::DeleteGold).collect();
    for _ in 0..count {
        let m = if rng.chance(0.12) { Mutation::DeleteGold } else { *rng.pick(&non_delete) };
        if apply(rng, &mut root, m) {
            chosen.push(m);
        }
    }
    let gold_present = root.count_gold() == 1;
    Variant { html: format!("<!doctype html>{}", root.render()), mutations: chosen, gold_present }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pw_core::snapshot::Snapshot;

    #[test]
    fn same_seed_same_page() {
        let a = make_site(&mut Rng::new(7), SiteKind::Checkout).render();
        let b = make_site(&mut Rng::new(7), SiteKind::Checkout).render();
        assert_eq!(a, b);
    }

    #[test]
    fn every_base_page_has_exactly_one_gold_element_and_it_parses() {
        for seed in 0..50 {
            let mut rng = Rng::new(seed);
            for kind in SiteKind::ALL {
                let page = make_site(&mut rng, kind);
                assert_eq!(page.count_gold(), 1, "seed {seed} {kind:?}");
                let snap = Snapshot::parse(&page.render());
                assert_eq!(snap.elements().iter().filter(|e| e.attrs.contains_key("data-pwgold")).count(), 1);
            }
        }
    }

    #[test]
    fn mutations_keep_html_parseable_and_report_deletions() {
        let mut deleted_seen = false;
        for seed in 0..200 {
            let mut rng = Rng::new(seed);
            let base = make_site(&mut rng, SiteKind::Product);
            let v = variant(&mut rng, &base);
            let snap = Snapshot::parse(&v.html);
            let gold_here = snap.elements().iter().any(|e| e.attrs.contains_key("data-pwgold"));
            assert_eq!(gold_here, v.gold_present, "seed {seed}: {:?}", v.mutations);
            deleted_seen |= !v.gold_present;
        }
        assert!(deleted_seen, "delete-gold should occur in 200 variants");
    }

    #[test]
    fn every_mutation_can_apply_and_keeps_one_target() {
        for m in Mutation::ALL {
            if m == Mutation::DeleteGold {
                continue;
            }
            let mut applied = false;
            'search: for kind in SiteKind::ALL {
                for seed in 0..40 {
                    let mut rng = Rng::new(seed);
                    let mut page = make_site(&mut rng, kind);
                    if apply(&mut rng, &mut page, m) {
                        applied = true;
                        assert_eq!(page.count_gold(), 1, "{} lost the target on {kind:?}", m.name());
                        let _ = Snapshot::parse(&format!("<!doctype html>{}", page.render()));
                        break 'search;
                    }
                }
            }
            assert!(applied, "{} never applied on any page kind", m.name());
        }
    }

    #[test]
    fn twin_adds_an_identical_sibling_without_an_id() {
        let mut rng = Rng::new(11);
        let mut page = make_site(&mut rng, SiteKind::Search);
        let before = page.render().matches(">Add to cart<").count();
        assert!(apply(&mut rng, &mut page, Mutation::AddTwin));
        assert_eq!(page.count_gold(), 1);
        let after = page.render().matches(">Add to cart<").count();
        assert_eq!(after, before + 1, "the twin should add one more button with the same text");
    }

    #[test]
    fn escaping_protects_text_and_attributes() {
        let n = Node::with_text("p", "a < b & \"c\"").attr("title", "x\"y");
        let html = n.render();
        assert!(html.contains("a &lt; b &amp; \"c\""));
        assert!(html.contains("title=\"x&quot;y\""));
    }
}
