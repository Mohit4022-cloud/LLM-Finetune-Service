# Model Card

## Model summary

This project fine-tunes a small causal language model with LoRA adapters to rewrite formal enterprise emails into concise Slack-style messages.

## Intended use

- internal productivity tooling
- style transformation prototypes
- demonstrations of supervised fine-tuning and evaluation workflows

## Non-goals

- legal or compliance-sensitive rewriting
- customer-facing message generation without human review
- broad open-domain chat

## Training data

- synthetic, template-generated enterprise communication examples
- structured around realistic scenario families
- deterministic train/validation/test splits

The dataset is meant to demonstrate good fine-tuning hygiene, not to claim production coverage.

## Prompt format

Each example is rendered as:

```text
### Instruction
<task instruction>

### Source Email
<formal email>

### Slack Message
<target output>
```

## Training configuration

Default configuration:

- base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- adapter method: LoRA
- task type: causal LM
- target modules: attention and MLP projection layers

Exact values are saved per run in `artifacts/train_runs/<run_id>/train_config.json`.

## Evaluation approach

Held-out evaluation compares the base model and the adapter on:

- style conformity
- brevity
- action preservation
- Slack-likeness
- lexical divergence from the source email

These are heuristic metrics and should be treated as lightweight regression signals.

## Limitations and risks

- synthetic supervision can encode templated phrasing
- tone conversion may oversimplify nuanced messages
- deadlines and action items can be dropped by weak models
- heuristics do not replace human review

## Safety notes

This project should not be used to rewrite sensitive HR, legal, or incident communication without human approval.
