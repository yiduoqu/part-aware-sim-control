# Separate SIMPACT planner environment

The simulator keeps its existing Newton environment. Use a separate Python 3.10+
planner environment on a host that can execute the simulator worker command; an
ARM environment must be created on the GH200 node, not copied from an x86 host.
Only the upstream generator/loop is used. Its Torch, Warp, reconstruction and
simulator extras are not needed by this bridge.

From this repository root, with the intended planner environment activated:

```bash
python -m pip install -r part_aware_control/planner-requirements.txt
python -m pip install --no-deps -r part_aware_control/upstream-requirement.txt
```

The second command installs SIMPACT from commit
`692aab2bb92e4df324d28c76630d17b08e814287`. The bridge checks the imported
revision from the source checkout or pip VCS installation metadata and refuses
another revision. No upstream source is vendored here. The adapter supplies all
three prompt templates, avoiding reliance on upstream prompts being packaged.

The user selected provider `codex` with exact model `gpt-6-sol` for this
integration. This is an explicit model change from the paper's Gemini setting;
it does not change the ten-proposal / three-refinement planning budget.

The `codex` backend uses the installed Codex CLI and its normal saved login
(`codex login status`). It accepts text and rollout images, requests stage-specific
JSON output, and runs each call in a temporary directory with a read-only sandbox,
ignored user config, and no persistent session. No OpenAI Python SDK or Google key
is required for this backend. The bridge never extracts or reuses an OAuth token;
Codex manages authentication. The common planner imports still require NumPy,
SciPy, Pillow and python-dotenv from the requirements above.

Choose `--provider codex --model gpt-6-sol` explicitly. The CLI executable must be
on PATH on the host that runs planning, and its saved login must be available on
that host. Each generation has a 300-second timeout; a missing CLI, unavailable
model, login failure or invalid response is reported as an error. Model calls
consume the selected Codex account's usage allowance.

The optional `google` backend still uses SIMPACT's Gemini client. Install
`google-genai` separately, supply `GOOGLE_API_KEY` through the environment and
select `--provider google --model gemini-2.5-pro` explicitly if that model is
available to your account. Google controls Gemini model availability/deprecation;
this adapter does not replace an unavailable model. Keys are not written to
result files. Neither backend falls back to another model/provider, and API or
schema errors never become evidence of a physical task failure.

Codex CLI structured output and saved-login behavior are described in the
[official non-interactive mode documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

The entry point's worker uses the Newton interpreter. It must run sequentially
inside a GPU Slurm allocation. Its only interchange with the planner is plan JSON,
rollout JSON and PNG files. Use a fresh output directory for each planning run so
prior evidence cannot leak into a different initial state or task. `result.json`
records requested/actual proposal count, completed/attempted planning and
confirmation rollouts, refinement count, provider/model, task/geometry/initial
state, outcome and wall time. Raw model responses live in `vlm/`.

The initial batch always executes all ten plans before verification. The first
verified success in proposal order is selected. Otherwise the imported loop gets
at most three single-plan refinements with accumulated evidence. A plan that
passed verification is independently replayed once; failure of that replay ends
the run. Exhausted planning failures retain the last plan for diagnosis and do not
run confirmation. This follows released-loop early success behavior; it is not a
claim of exact paper-algorithm reproduction.

A small offline check can run in the planner environment, without a GPU or API key:

```bash
python -m unittest discover -s tests -p test_simpact_planner.py
```
