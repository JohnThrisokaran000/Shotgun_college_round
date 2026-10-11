//! The release gate. A release to a protected channel needs every patch approved in the audit log, and a
//! complete, passing verification of the same patch content. A waiver lets one gate through, with a
//! reason, and the waiver is written to the audit trail.

use anyhow::Result;

use pw_core::review::{self, Approval};
use pw_core::release;
use pw_core::verify::{self, Outcome};

use crate::project::{Project, ReleasePlan};

#[derive(Default, Clone)]
pub struct Waivers {
    pub verification: Option<String>,
    pub review: Option<String>,
}

#[derive(Default)]
pub struct Gate {
    /// Why the release may not publish. Empty when it may.
    pub blockers: Vec<String>,
    /// Gates that a waiver let through, with the reason, for the audit trail.
    pub waived: Vec<(&'static str, String)>,
    /// When the verification that passed was recorded, if one was needed.
    pub verified_ms: Option<u64>,
}

/// Judge `plan` against the gate for `channel`. Other channels have no gate.
pub fn evaluate(project: &Project, channel: &str, plan: &ReleasePlan, waivers: &Waivers) -> Result<Gate> {
    let mut gate = Gate::default();
    if !project.is_protected(channel) {
        return Ok(gate);
    }
    // An approval is only as good as the log that records it, so a broken chain stops the release.
    let audit = project.audit();
    audit.verify()?;
    let events = audit.read_all()?;

    let mut unapproved = Vec::new();
    // Rejected patches are withdrawn from every release, so only the active ones need approval.
    for patch in project.load_patches()?.into_iter().filter(|p| p.status != pw_core::patch::Status::Rejected) {
        match review::approval(&patch, &events)? {
            Approval::Approved => {}
            Approval::NotApproved(status) => unapproved.push(format!("{} is {}", patch.id, status.label())),
            Approval::Unproven(reason) => unapproved.push(format!("{}: {reason}", patch.id)),
        }
    }
    if !unapproved.is_empty() {
        let list = unapproved.join("; ");
        match &waivers.review {
            Some(reason) => gate.waived.push(("review", format!("{reason} [{list}]"))),
            None => gate.blockers.push(format!(
                "not approved for {channel}: {list}. Each patch needs `pw review propose` and then `pw review approve` by someone other than its author"
            )),
        }
    }

    let digest = release::patches_digest(&plan.manifest.patches)?;
    match verify::latest(&project.root, &digest)? {
        Some(record) if record.complete() => gate.verified_ms = Some(record.ts_ms),
        found => {
            let why = match found {
                None => "there is no verification record for this content".to_string(),
                Some(r) => format!(
                    "the newest verification of this content is incomplete ({} passed, {} failed, {} not run)",
                    r.count(Outcome::Pass),
                    r.count(Outcome::Fail),
                    r.count(Outcome::NotRun)
                ),
            };
            match &waivers.verification {
                Some(reason) => gate.waived.push(("verification", format!("{reason} [{why}]"))),
                None => gate.blockers.push(format!(
                    "{why}. Publish this content to a preview channel, then run `pw verify --channel <preview> --base <URL>` until it passes"
                )),
            }
        }
    }
    Ok(gate)
}
