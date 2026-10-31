# Data format

One row per **episode**: a situation, what was done, and what happened.

| column | required | meaning |
|---|---|---|
| workflow `inputs` | yes | what the model sees (numbers, categories, or text) |
| `action` | yes | what was done. `choice` tasks: one of `action_space`. `generation` tasks: the response text |
| `outcome_success` | yes | measured result: did it work? (bool; map your own values with `data.success_values`) |
| `episode_id` | auto | unique id (generated if absent) |
| `<group_by>` | yes | e.g. machine/customer/site - **splits never share a group**, preventing leakage |
| `<slice_by>` | yes | root-cause / segment used for per-slice evaluation (what "failure modes" are named after) |
| `source`, `status` | optional | `history`/`live`/`synthetic`; `accepted`/`overridden`/`failed` |

Columns measured **after** the action (`outcome_fields`) must never be model inputs; this is enforced.

Task types: `choice` (closed action set; scored by log-likelihood; gives a confidence for the human gate) and
`generation` (free text; scored by a **verifier** - exact, regex, JSON fields, or your own function). Generation
workflows evaluate on successful episodes only, since references exist only for good answers.

`autoresearch validate --config c.json` reports missing columns, bad actions, class balance, thin or empty slices,
and group counts before you spend compute.

## How much data?
There is no magic number. As a rule of thumb: hundreds of episodes to see signal, thousands for stable slices.
Rare cases (<1%) usually need synthetic or verified augmentation, which the loop proposes when it finds the failing slice.
