from __future__ import annotations


FORMAL_REPLACEMENTS = {
    "I am writing to inform you that": "Quick update:",
    "Please be advised that": "FYI:",
    "For the sake of clarity,": "Just to clarify:",
    "Thank you for your attention to this matter.": "",
    "Kind regards,": "",
    "Please confirm next steps": "Can someone confirm next steps",
}


def generate_dev_fallback(text: str) -> str:
    rewritten = text
    for formal, casual in FORMAL_REPLACEMENTS.items():
        rewritten = rewritten.replace(formal, casual)
    rewritten = " ".join(rewritten.split())
    if len(rewritten) > 240:
        rewritten = rewritten[:237] + "..."
    return rewritten
