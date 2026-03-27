# Dataset Notes

The dataset in `data/splits/` is generated deterministically from structured business-communication templates.

Schema:

- `id`
- `split`
- `scenario_type`
- `instruction`
- `source_email`
- `target_slack`
- `metadata`

The generator is intentionally synthetic but not random noise. It combines:

- scenario family
- department
- audience
- stakeholder
- urgency
- deadlines
- action requirements

Regenerate and validate:

```bash
make data
make validate-data
```
