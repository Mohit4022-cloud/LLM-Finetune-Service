from __future__ import annotations


def render_prompt(record: dict[str, object], include_target: bool = False) -> str:
    prompt = (
        "### Instruction\n"
        f"{record['instruction']}\n\n"
        "### Source Email\n"
        f"{record['source_email']}\n\n"
        "### Slack Message\n"
    )
    if include_target:
        return prompt + f"{record['target_slack']}"
    return prompt
