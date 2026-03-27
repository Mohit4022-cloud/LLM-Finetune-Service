from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_finetune_service.config import Settings


INSTRUCTION = (
    "Rewrite the enterprise email as a concise, friendly Slack message. Preserve the core facts, "
    "requested action, and deadlines. Remove formal sign-offs and avoid sounding robotic."
)

FORMAL_OPENERS = [
    "I am writing to inform you that",
    "Please be advised that",
    "I wanted to let you know that",
    "I am writing to confirm that",
    "For the sake of clarity,",
    "Following our prior discussion,",
]

SIGN_OFFS = [
    "Thank you for your attention to this matter.",
    "Please let me know if you require additional information.",
    "Your prompt assistance would be greatly appreciated.",
    "Kind regards,",
]

DEPARTMENTS = ["engineering", "product", "finance", "sales", "marketing", "operations", "hr", "security"]
AUDIENCES = ["the delivery team", "leadership", "the client team", "the recruiting panel", "operations", "the account owners"]
URGENCIES = ["low", "medium", "high", "critical"]
STAKEHOLDERS = ["vendor", "customer", "legal", "security", "finance", "design", "data"]
DEADLINES = [
    "by EOD Friday",
    "before the Monday planning meeting",
    "by tomorrow morning",
    "before the release cut on Wednesday",
    "this afternoon",
    "by next Tuesday",
]
PROJECTS = [
    "the Phoenix migration",
    "the mobile checkout launch",
    "the payroll reporting refresh",
    "the customer onboarding revamp",
    "the retention dashboard rollout",
    "the access review program",
]
DETAILS = [
    "The current status should be reflected in the next cross-functional update.",
    "This may change what we tell stakeholders in the weekly summary.",
    "The team is trying to keep execution risk contained before the next milestone.",
    "This is one of the remaining dependencies before the broader rollout.",
    "We should keep the message tight so the next update is easy to skim.",
]
TOPICS = [
    "the customer migration checklist",
    "the release readiness review",
    "the onboarding workflow",
    "the Q3 planning packet",
    "the security remediation backlog",
    "the procurement approval queue",
    "the analytics dashboard refresh",
    "the mobile login experiment",
    "the billing reconciliation run",
    "the headcount approval process",
    "the partner launch checklist",
    "the support escalation flow",
]

SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_type": "status_update",
        "formal_bodies": [
            "the {department} workstream is on track, but one dependency from {stakeholder} is still outstanding",
            "the rollout is progressing as planned and the team completed the final validation checks",
            "the implementation is complete, and we are preparing a short handoff for {audience}",
        ],
        "slack_openers": ["Quick update:", "Heads up:", "Small update:"],
        "slack_bodies": [
            "the {department} work is mostly on track, but we're still waiting on {stakeholder}",
            "rollout is looking good and the team wrapped final checks",
            "implementation is done and we're putting together a quick handoff for {audience}",
        ],
        "requires_action": False,
    },
    {
        "scenario_type": "escalation",
        "formal_bodies": [
            "a production issue is blocking the team from completing the release and requires immediate attention",
            "multiple stakeholders have raised concerns about the current delay, and we need a decision on mitigation steps",
            "a security finding needs review before we can proceed with deployment",
        ],
        "slack_openers": ["Heads up:", "Need help:", "Important:"],
        "slack_bodies": [
            "a prod issue is blocking the release and needs attention now",
            "a few stakeholders are pushing on the delay, and we need a call on the mitigation plan",
            "we need a quick review on a security finding before deploy",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "meeting_request",
        "formal_bodies": [
            "we should schedule time with {audience} to review the latest project risks and next steps",
            "I would appreciate a short meeting to align on priorities for the next sprint",
            "it would be helpful to meet with {stakeholder} and confirm the launch checklist",
        ],
        "slack_openers": ["Can we set up time", "Can we grab 20 min", "Could we meet"],
        "slack_bodies": [
            "with {audience} to go over the latest risks and next steps?",
            "to align on priorities for the next sprint?",
            "with {stakeholder} to confirm the launch checklist?",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "approval",
        "formal_bodies": [
            "the budget request for the {department} initiative has been approved and procurement may proceed",
            "leadership has approved the revised plan, and we can move into implementation",
            "the travel request has been approved for the dates listed below",
        ],
        "slack_openers": ["Good news:", "FYI:", "Approved:"],
        "slack_bodies": [
            "the budget request for the {department} initiative is approved, so procurement can move forward",
            "leadership approved the revised plan, so we're good to start implementation",
            "the travel request is approved for those dates",
        ],
        "requires_action": False,
    },
    {
        "scenario_type": "decline",
        "formal_bodies": [
            "we cannot move forward with the request at this time because the supporting analysis is incomplete",
            "the proposed scope increase is not approved for the current quarter due to budget constraints",
            "the vendor change request cannot be accepted until legal review is complete",
        ],
        "slack_openers": ["Update:", "Quick note:", "Flagging this:"],
        "slack_bodies": [
            "we can't move on the request yet because the supporting analysis is still incomplete",
            "the scope increase isn't approved this quarter because of budget limits",
            "we can't accept the vendor change until legal wraps review",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "feedback",
        "formal_bodies": [
            "the draft is in a strong place, though it would benefit from a clearer recommendation section",
            "the presentation covered the necessary details, but the narrative could be tighter for leadership",
            "the handoff document is useful, and adding concrete owners would make it easier to execute",
        ],
        "slack_openers": ["Thoughts:", "Feedback:", "Quick feedback:"],
        "slack_bodies": [
            "the draft is in a good spot. It just needs a clearer recommendation section",
            "the presentation had the right info, but the story could be tighter for leadership",
            "the handoff doc is helpful. Adding explicit owners would make it easier to run with",
        ],
        "requires_action": False,
    },
    {
        "scenario_type": "clarification",
        "formal_bodies": [
            "the timeline referenced in the prior note applies to the pilot only and not the full rollout",
            "the budget estimate covers implementation work and does not include ongoing support costs",
            "the review is required for both contractors and full-time employees",
        ],
        "slack_openers": ["Just to clarify:", "Clarification:", "Quick clarification:"],
        "slack_bodies": [
            "the timeline in the last note is just for the pilot, not the full rollout",
            "the budget estimate only covers implementation, not ongoing support",
            "the review applies to contractors and full-time folks",
        ],
        "requires_action": False,
    },
    {
        "scenario_type": "incident",
        "formal_bodies": [
            "we are seeing elevated error rates in the customer workflow and have paused the rollout while the team investigates",
            "an outage in a shared dependency is affecting sign-in performance across the platform",
            "the latest deployment introduced a regression, and we have started mitigation work",
        ],
        "slack_openers": ["Incident update:", "Heads up:", "Current status:"],
        "slack_bodies": [
            "we're seeing higher error rates in the customer flow, so rollout is paused while we investigate",
            "a shared dependency outage is slowing sign-in across the platform",
            "the latest deploy introduced a regression, and mitigation is in flight",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "vendor_coordination",
        "formal_bodies": [
            "the vendor confirmed a delay in delivery and asked for revised implementation dates",
            "we need confirmation from the vendor before finalizing the purchasing timeline",
            "the vendor has sent updated pricing, and finance would like a quick review before approval",
        ],
        "slack_openers": ["Vendor update:", "Quick vendor note:", "Need a callout:"],
        "slack_bodies": [
            "the vendor is delayed and asked for updated implementation dates",
            "we need vendor confirmation before locking the purchasing timeline",
            "the vendor sent updated pricing and finance wants a quick review before approval",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "leadership_update",
        "formal_bodies": [
            "leadership requested a short summary of the launch risks and our mitigation plan",
            "the executive team would like a concise view of the delivery timeline and any open decisions",
            "we need to package the current status into a one-paragraph update for the weekly leadership sync",
        ],
        "slack_openers": ["Leadership ask:", "Quick ask:", "Need this today:"],
        "slack_bodies": [
            "leadership wants a short summary of the launch risks and mitigation plan",
            "the exec team wants a quick view of the timeline plus any open decisions",
            "we need a one-paragraph status update for the weekly leadership sync",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "recruiting",
        "formal_bodies": [
            "the interview schedule has shifted because one panelist is unavailable",
            "we would like to move a candidate to the final round pending hiring manager approval",
            "the recruiting coordinator needs feedback submitted before the decision meeting",
        ],
        "slack_openers": ["Recruiting update:", "Can you take a look:", "Reminder:"],
        "slack_bodies": [
            "the interview schedule moved because one panelist is out",
            "we'd like to move a candidate to final round once the hiring manager signs off",
            "the recruiting coordinator needs feedback in before the decision meeting",
        ],
        "requires_action": True,
    },
    {
        "scenario_type": "finance_ops",
        "formal_bodies": [
            "the invoice is missing a purchase order reference and cannot be processed yet",
            "finance needs confirmation on the cost center before the request can be booked",
            "the monthly spend report is available and includes a variance that should be reviewed",
        ],
        "slack_openers": ["Finance note:", "Quick finance update:", "Need confirmation:"],
        "slack_bodies": [
            "the invoice is missing a PO reference, so finance can't process it yet",
            "finance needs the cost center before they can book the request",
            "the monthly spend report is ready and has a variance we should review",
        ],
        "requires_action": True,
    },
]


@dataclass(slots=True)
class DatasetBuildResult:
    records: list[dict[str, Any]]
    split_counts: dict[str, int]
    output_paths: dict[str, Path]


def _assign_split(group_key: str) -> str:
    bucket = int(hashlib.sha256(group_key.encode("utf-8")).hexdigest(), 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def _formal_email(rng: random.Random, scenario: dict[str, Any], index: int) -> tuple[str, dict[str, Any], int]:
    department = DEPARTMENTS[index % len(DEPARTMENTS)]
    audience = AUDIENCES[(index // len(DEPARTMENTS)) % len(AUDIENCES)]
    stakeholder = STAKEHOLDERS[(index // (len(DEPARTMENTS) * len(AUDIENCES))) % len(STAKEHOLDERS)]
    urgency = rng.choice(URGENCIES)
    deadline = rng.choice(DEADLINES)
    project = PROJECTS[index % len(PROJECTS)]
    topic = TOPICS[(index // len(PROJECTS)) % len(TOPICS)]
    opener = rng.choice(FORMAL_OPENERS)
    template_index = rng.randrange(len(scenario["formal_bodies"]))
    body = scenario["formal_bodies"][template_index].format(
        department=department,
        audience=audience,
        stakeholder=stakeholder,
    )
    context = rng.choice(
        [
            f"This affects {audience} for {project} and should stay visible in the weekly update on {topic}.",
            f"The current status has implications for the {department} roadmap on {project}, especially around {topic}.",
            f"We should keep {stakeholder} informed on {project} so there are no surprises around {topic}.",
            f"Please share any concerns about {project} and {topic} before the next planning cycle.",
        ]
    )
    detail = rng.choice(DETAILS)
    action_sentence = (
        f"Please confirm next steps {deadline}."
        if scenario["requires_action"]
        else f"No immediate action is required, but please flag any concerns {deadline}."
    )
    closing = rng.choice(SIGN_OFFS)
    source_email = f"{opener} {body}. {context} {detail} {action_sentence} {closing}"
    metadata = {
        "urgency": urgency,
        "audience": audience,
        "department": department,
        "tone_constraints": ["friendly", "concise", "Slack-style"],
        "requires_action": scenario["requires_action"],
        "contains_deadline": True,
        "emoji_allowed": urgency in {"low", "medium"},
        "stakeholder": stakeholder,
        "deadline": deadline,
        "project": project,
        "topic": topic,
    }
    return source_email, metadata, template_index


def _slack_message(rng: random.Random, scenario: dict[str, Any], metadata: dict[str, Any], template_index: int) -> str:
    opener = rng.choice(scenario["slack_openers"])
    body = scenario["slack_bodies"][template_index].format(
        department=metadata["department"],
        audience=metadata["audience"],
        stakeholder=metadata["stakeholder"],
    )
    deadline = metadata["deadline"]
    project = metadata["project"]
    topic = metadata["topic"]
    framing = rng.choice(
        [
            f" This is for {project} and touches {topic}.",
            f" Keeping this tied to {project}, especially {topic}.",
            f" This affects {project} and the current work on {topic}.",
            "",
        ]
    )
    ending = (
        f" Can someone confirm next steps {deadline}?"
        if metadata["requires_action"]
        else f" No action needed right now, but flag anything odd {deadline}."
    )
    emoji = ""
    if metadata["emoji_allowed"] and rng.random() > 0.55:
        emoji = rng.choice(["", " 👍", " ✅", " 📌", " 👀"])
    return f"{opener} {body}.{framing}{ending}{emoji}".strip()


def build_dataset(settings: Settings | None = None, output_dir: Path | None = None) -> DatasetBuildResult:
    settings = settings or Settings()
    output_dir = output_dir or settings.dataset_dir
    rng = random.Random(settings.data_seed)

    records: list[dict[str, Any]] = []
    for index in range(settings.dataset_size):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        local_rng = random.Random(rng.randint(0, 10_000_000))
        source_email, metadata, template_index = _formal_email(local_rng, scenario, index)
        target_slack = _slack_message(local_rng, scenario, metadata, template_index)
        group_key = f"{scenario['scenario_type']}:{metadata['department']}:{metadata['audience']}:{index % 5}"
        split = _assign_split(group_key)
        record = {
            "id": f"{scenario['scenario_type']}-{index:04d}",
            "split": split,
            "scenario_type": scenario["scenario_type"],
            "instruction": INSTRUCTION,
            "source_email": source_email,
            "target_slack": target_slack,
            "metadata": metadata,
        }
        records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    output_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "test": output_dir / "test.jsonl",
    }

    split_buckets = {"train": [], "validation": [], "test": []}
    for record in records:
        split_buckets[record["split"]].append(record)
        split_counts[record["split"]] += 1

    for split_name, path in output_paths.items():
        with path.open("w", encoding="utf-8") as handle:
            for record in split_buckets[split_name]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return DatasetBuildResult(records=records, split_counts=split_counts, output_paths=output_paths)
