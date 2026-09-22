# Rollback Residue

**Category:** Forensics\
**Track:** Track A --- CTF / SFT + RL Training-Data Design\
**Challenge Type:** ML Security / Optimizer Forensics\
**Difficulty:** Intermediate--Advanced\
**Flag:** `PROVUE{OPTIMIZER_AUTOPSY}`

## Overview

Rollback Residue is a machine-learning forensics CTF challenge built
around an incomplete rollback of a neural-network training process.

The challenge simulates a situation in which a model is restored to an
earlier parameter state, but the optimizer state is not restored.
Although the visible model parameters appear to have been rolled back,
the Adam optimizer still contains information about an event that
occurred immediately before the rollback.

The objective is to investigate the checkpoint, understand the rollback
mechanism, identify the optimizer-state channel carrying the residue,
recover the encoded information, reconstruct the payload, and ultimately
recover the flag.

The challenge is intentionally small and CPU-friendly. It uses a tiny
PyTorch MLP and deterministic artifacts so that the forensic process is
reproducible while still requiring the solver to reason about model
state, optimizer state, rollback semantics, and hidden training
information.

## Why This Challenge Exists

Modern ML systems commonly save model parameters and optimizer state
together in checkpoints. During recovery, experimentation, deployment,
or fault handling, it is possible to restore the model parameters
without correctly restoring the optimizer state.

This challenge turns that operational mistake into a forensic problem.

The important distinction is that the model parameters are not the only
state maintained during training. Adam also maintains internal running
statistics such as first- and second-moment estimates. These statistics
can retain information from a gradient update even when the
corresponding parameter update is subsequently undone.

Rollback Residue uses this property deliberately.

The challenge therefore tests whether a solver can look beyond the
visible neural-network weights and inspect the complete training state.

## Challenge Scenario

A training process is running a small neural network using Adam.

At a hidden event step, an 8-bit secret value is converted into eight
binary values. Those bits influence the gradient of a small auxiliary
parameter vector.

Adam performs its update.

Immediately afterward, the challenge restores the model parameters and
auxiliary parameter to their pre-update values. However, the optimizer
state remains in its post-update condition.

The resulting checkpoint therefore contains two apparently conflicting
pieces of evidence:

-   The model looks as though the event never happened.
-   The optimizer state still reflects the event.

The solver must identify and exploit this discrepancy.

## Core Forensic Insight

The central idea is:

``` text
parameter state
      |
      | rollback
      v
pre-event model parameters

optimizer state
      |
      | NOT rolled back
      v
post-event Adam residue
```

The visible model state alone is insufficient to recover the hidden
event.

The optimizer state is the forensic artifact.

In this implementation, the relevant channel is the Adam `exp_avg` state
associated with the eight-element auxiliary parameter.

## Model Architecture

The challenge uses a deliberately small multilayer perceptron.

The architecture is:

``` text
Input: 16
   |
Linear(16 -> 64)
   |
ReLU
   |
Linear(64 -> 64)
   |
ReLU
   |
Linear(64 -> 4)
   |
Output: 4 classes
```

The model is implemented in `src/model.py`.

The network is not intended to provide meaningful predictive
performance. Its purpose is to create a realistic PyTorch training state
while keeping the challenge fast, deterministic, and easy to reproduce
on CPU.

An additional eight-element learnable auxiliary parameter is trained
alongside the MLP. This auxiliary state provides the controlled channel
through which the hidden byte is encoded.

## Data and Training

The training data is generated deterministically.

The input consists of 256 randomly generated 16-dimensional samples, and
the target labels contain four classes.

Training runs for 100 optimization steps with a batch size of 32.

The training setup uses:

-   PyTorch
-   Adam
-   Learning rate `1e-3`
-   Fixed random seed
-   256 samples
-   16 input features
-   4 output classes
-   100 optimization steps

The deterministic setup ensures that the clean checkpoint and challenge
checkpoints can be reproduced consistently.

## Hidden Event

The hidden event occurs at optimization step 40.

For each challenge checkpoint, one byte of the flag is used as the
secret.

The byte is converted into eight bits:

``` text
b7 b6 b5 b4 b3 b2 b1 b0
```

Those eight bits are applied to the auxiliary parameter through a dot
product.

Conceptually:

``` text
encoded_signal = auxiliary_state · secret_bits
```

The encoded signal is added to the training loss only at the hidden
event step.

The resulting gradient affects Adam's internal state.

## The Rollback

After the event update is performed, the challenge deliberately saves
the pre-update model and auxiliary parameter state and restores them.

The important implementation detail is that only parameter state is
restored.

The optimizer is not restored.

Therefore:

``` text
Model parameters:
pre-event state

Auxiliary parameter:
pre-event state

Adam optimizer:
post-event state
```

This mismatch is the residue that gives the challenge its name.

## Why the Residue Survives

Adam maintains running estimates of gradients.

The first-moment estimate can be represented conceptually as:

``` text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
```

where `g_t` is the current gradient.

Even if the parameter value is later restored, the optimizer's internal
moment estimate is not automatically reversed.

Consequently, the optimizer checkpoint retains evidence of the gradient
event.

Rollback of the model does not imply rollback of the optimizer.

That is the key forensic weakness being exploited.

## Challenge Artifacts

The repository contains one clean checkpoint and one challenge
checkpoint for each flag byte.

The artifact layout is:

``` text
artifacts/
├── clean.pt
├── challenge_000.pt
├── challenge_001.pt
├── ...
└── challenge_024.pt
```

There are 25 challenge checkpoints because the flag contains 25 bytes.

`clean.pt` provides the baseline training state.

Each `challenge_XXX.pt` contains a single hidden byte encoded through
optimizer residue.

## Artifact Comparison

A useful investigation starts by comparing the clean checkpoint with a
challenge checkpoint.

The solver should inspect:

-   model parameters
-   auxiliary parameters
-   optimizer state
-   optimizer moment tensors
-   tensor shapes
-   numerical differences between corresponding states

The important observation is that the model state is restored while a
particular optimizer-state tensor differs.

This establishes the rollback/residue relationship.

## Encoding

Each flag byte is encoded as eight bits.

For example, a byte is represented as:

``` text
b7 b6 b5 b4 b3 b2 b1 b0
```

The implementation uses an eight-element floating-point auxiliary
parameter.

The encoding and decoding helpers are located in:

``` text
src/encoding.py
```

The helper functions are:

``` text
byte_to_bits()
bits_to_byte()
```

The challenge does not require brute-forcing the flag. The intended
approach is to recover the bit pattern from the optimizer residue.

## Recovering the Residue

The challenge evaluator analyzes the relevant optimizer state and
compares the challenge state against the clean baseline.

The eight-element `exp_avg` difference is interpreted as the residue
channel.

The encoded signal produces a distinguishable pattern across the eight
elements.

The solver can therefore:

1.  Load the clean checkpoint.
2.  Load a challenge checkpoint.
3.  Locate the relevant Adam state.
4.  Compute the difference between challenge and clean state.
5.  Identify the eight-element residue.
6.  Threshold the residue to recover the eight bits.
7.  Convert the bits into a byte.
8.  Repeat for all challenge checkpoints.
9.  Concatenate the recovered bytes.

## Recovering the Flag

Each challenge checkpoint corresponds to one byte.

The checkpoints are ordered:

``` text
challenge_000.pt
challenge_001.pt
...
challenge_024.pt
```

The recovered bytes are concatenated in that order.

The resulting payload is:

``` text
PROVUE{OPTIMIZER_AUTOPSY}
```

The final flag is validated by the challenge service.

## Reward Structure

The challenge exposes six progressive milestones.

  ------------------------------------------------------------------------
  Stage                                       Reward Purpose
  --------------------- ---------------------------- ---------------------
  `recon`                                         10 Identify the
                                                     training/checkpoint
                                                     structure

  `rollback`                                      20 Recognize that model
                                                     state was restored
                                                     without optimizer
                                                     state

  `channel`                                       20 Identify the
                                                     optimizer-state
                                                     channel

  `residue`                                       20 Recover the hidden
                                                     bit residue

  `payload`                                       20 Reconstruct the
                                                     encoded payload

  `flag`                                          10 Submit the complete
                                                     flag
  ------------------------------------------------------------------------

**Total: 100 points**

The staged reward structure is intentional.

It provides observable intermediate signals rather than making the task
a single all-or-nothing flag submission. This makes the challenge
suitable as a training-data environment for agents that need to learn
multi-step reasoning and tool interaction.

## Reward Semantics

The reward system is implemented in:

``` text
challenge/reward_state.py
```

Each milestone can only be awarded once.

The service maintains the cumulative reward and the set of reached
milestones.

A successful progression therefore looks like:

``` text
recon       -> 10
rollback    -> 20
channel     -> 20
residue     -> 20
payload     -> 20
flag        -> 10

total       -> 100
```

Invalid evidence receives zero reward.

## Challenge Service

The challenge is exposed through a FastAPI service.

The main server is:

``` text
challenge/server.py
```

The service provides:

``` text
GET  /health
GET  /challenge
GET  /checkpoint
POST /submit
```

### Health Endpoint

``` text
GET /health
```

Returns a simple service health response.

### Challenge Endpoint

``` text
GET /challenge
```

Returns the challenge name, category, and high-level description.

### Checkpoint Endpoint

``` text
GET /checkpoint
```

Returns information about the clean checkpoint and the number of
challenge artifacts.

### Submission Endpoint

``` text
POST /submit
```

Accepts a stage and evidence object.

The evidence is passed to the evaluator, and successful submissions
receive the corresponding reward.

## Evaluator

The evaluator is implemented in:

``` text
challenge/evaluator.py
```

It contains checks corresponding to the six reward milestones.

The evaluator verifies evidence rather than simply awarding points for
reaching an endpoint.

The checks cover:

-   reconstruction of the training setup
-   identification of the rollback condition
-   identification of the optimizer-state channel
-   extraction of the residue
-   recovery of the payload
-   exact flag recovery

The evaluator therefore connects the conceptual forensic stages to
machine-checkable criteria.

## Reference Validation Agent

The repository contains:

``` text
solution/solver_agent.py
```

This is a deterministic reference and validation agent.

It is not a trained reinforcement-learning model.

Its purpose is to demonstrate that an external agent can interact with
the challenge through the HTTP interface, perform the forensic analysis,
submit evidence stage by stage, and receive the expected reward.

The agent:

1.  Connects to the challenge service.
2.  Retrieves challenge metadata.
3.  Retrieves checkpoint metadata.
4.  Loads the local challenge artifacts.
5.  Performs the optimizer-state analysis.
6.  Recovers the encoded bits.
7.  Reconstructs the flag.
8.  Submits evidence for each milestone.
9.  Verifies the final reward.

A successful validation run reaches:

``` text
Final reward: 100
```

and recovers:

``` text
PROVUE{OPTIMIZER_AUTOPSY}
```

## HTTP Interaction

The reference agent communicates with the challenge service instead of
importing the evaluator or reward implementation directly.

The interaction is therefore:

``` text
solver_agent.py
        |
        | HTTP
        v
FastAPI challenge server
        |
        v
Evaluator
        |
        v
RewardState
```

This separation is important because it demonstrates the intended
agent-environment interaction boundary.

The solver does not receive reward by directly modifying internal
challenge state.

## Docker Environment

The challenge is packaged as a Docker service.

The runtime image contains:

``` text
challenge/
src/
artifacts/
requirements.txt
```

The solution, tests, experiments, and documentation are intentionally
excluded from the runtime image.

The container starts the FastAPI service with Uvicorn.

The Docker image does not automatically run the solver, training
experiments, or tests.

This keeps the challenge environment separate from the reference
solution and research environment.

## Running the Challenge

Build and start the challenge:

``` bash
docker compose up --build
```

The service becomes available at:

``` text
http://localhost:8000
```

Check the service:

``` bash
curl http://localhost:8000/health
```

Expected response:

``` json
{"status":"ok"}
```

The challenge can then be queried:

``` bash
curl http://localhost:8000/challenge
```

and:

``` bash
curl http://localhost:8000/checkpoint
```

## Running the Reference Agent

With the challenge service running, execute the reference agent from the
repository environment:

``` bash
python -m tests.test_agent
```

The expected progression is:

``` text
recon        -> reward=10 total=10
rollback     -> reward=20 total=30
channel      -> reward=20 total=50
residue      -> reward=20 total=70
payload      -> reward=20 total=90
flag         -> reward=10 total=100
```

The agent should finish with:

``` text
Recovered flag: PROVUE{OPTIMIZER_AUTOPSY}
Final reward: 100
```
## Running the Solver Agent

```The repository contains a deterministic reference solution at:```

``` bash
python -m solution.solver_agent
```
```The reference solution demonstrates the complete intended CTF solve path, from checkpoint inspection to final flag recovery.

It is a deterministic reference/validation agent, not a trained reinforcement-learning agent. Its purpose is to demonstrate how an agent can interact with the challenge environment, inspect the provided artifacts, recover the optimizer residue, reconstruct the hidden payload, and submit evidence to the challenge service.

A typical inspection includes states such as:

[AGENT] parameter=0 state=exp_avg_sq shape=[64, 16]
[AGENT] parameter=1 state=step shape=[]
[AGENT] parameter=1 state=exp_avg shape=[64]
[AGENT] parameter=1 state=exp_avg_sq shape=[64]
...
[AGENT] parameter=6 state=step shape=[]
[AGENT] parameter=6 state=exp_avg shape=[8]
[AGENT] parameter=6 state=exp_avg_sq shape=[8]
```


## Repository Structure

``` text
Rollback-Residue/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
│
├── challenge/
│   ├── __init__.py
│   ├── server.py
│   ├── evaluator.py
│   └── reward_state.py
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── encoding.py
│   ├── forensic.py
│   └── generate.py
│
├── solution/
│   ├── __init__.py
│   └── solver_agent.py
│
├── artifacts/
│   ├── clean.pt
│   ├── challenge_000.pt
│   ├── ...
│   └── challenge_024.pt
│
├── tests/
├── experiments/
├── experiment_results/
└── artifacts-test/
```

## Source Code Responsibilities

### `src/model.py`

Defines the TinyMLP architecture.

### `src/encoding.py`

Contains byte-to-bit and bit-to-byte conversion utilities.

### `src/forensic.py`

Contains deterministic data generation, model construction, auxiliary
state, and checkpoint-generation logic.

### `src/generate.py`

Generates the clean checkpoint and the 25 challenge checkpoints.

### `challenge/server.py`

Provides the HTTP challenge service.

### `challenge/evaluator.py`

Implements the machine-checkable milestone validation.

### `challenge/reward_state.py`

Tracks milestone completion and cumulative reward.

### `solution/solver_agent.py`

Provides the deterministic reference/validation agent.

## Research and Experimentation

The project also contains an experimental research history.

The experiments explored different ways of creating recoverable
information from optimizer or model-state behavior before the final
mechanism was selected.

Examples include:

-   pulse experiments
-   pulse decay
-   compensated pulses
-   localized events
-   temporal events
-   controlled temporal events
-   channel analysis
-   multichannel validation

These experiments helped determine which mechanisms were reproducible,
distinguishable, and suitable for a competition-style challenge.

The final challenge intentionally uses a simpler and more controlled
mechanism than many of the exploratory experiments.

## Design Decision: Why Optimizer State

A model-weight-only challenge would make rollback residue difficult to
demonstrate because restoring the parameters would remove the most
obvious evidence.

Optimizer state provides a separate persistent state channel.

This creates a clean forensic distinction:

``` text
What the model says happened
versus
What the optimizer remembers happened
```

That distinction is the core of the challenge.

## Design Decision: Why a Tiny MLP

The MLP is intentionally small.

A large architecture would add computational cost without improving the
central forensic problem.

The challenge is about state persistence and optimizer analysis, not
about achieving high model accuracy.

A small model also makes the artifact inspection process easier for a
solver and keeps the Docker environment lightweight.

## Design Decision: Why Deterministic Artifacts

Every challenge checkpoint is generated from a fixed seed and
deterministic training procedure.

This makes the task reproducible across environments and simplifies
evaluator validation.

The challenge is therefore not dependent on a fragile random behavior
that happens to work on one machine.

## Design Decision: Why Multiple Checkpoints

A single hidden byte would demonstrate the mechanism but would not
require a complete recovery workflow.

Using one checkpoint per flag byte creates a repeated forensic task.

The solver must identify the invariant mechanism and apply it
consistently across all artifacts.

This turns the challenge into a small-scale dataset of related forensic
instances.

## Security / ML Forensics Perspective

The challenge models a class of failures that can occur when ML systems
treat model parameters as the entire state of a training process.

For optimizers such as Adam, the optimizer state can contain information
that is not visible in the current parameter values.

Therefore, a rollback procedure that restores only model weights can
leave behind state that influences future training and potentially
reveals information about prior events.

The challenge is a controlled demonstration of this state-consistency
problem.

## Training-Data Design Perspective

Rollback Residue is designed to produce structured intermediate signals
for SFT and RL-style training.

The task has:

-   a clear initial environment
-   a sequence of forensic subtasks
-   machine-checkable evidence
-   sparse but staged rewards
-   a deterministic reference trajectory
-   a final exact-answer condition

The milestone structure allows training examples to represent partial
progress rather than only successful flag recovery.

For example, an agent can demonstrate that it correctly identified the
rollback condition before it has recovered the hidden payload.

## Intended Solver Workflow

The intended reasoning path is:

``` text
Inspect artifacts
      ↓
Understand checkpoint structure
      ↓
Compare clean and challenge states
      ↓
Identify model/optimizer mismatch
      ↓
Locate persistent optimizer state
      ↓
Find the eight-element residue channel
      ↓
Recover the eight encoded bits
      ↓
Convert bits to byte
      ↓
Repeat across challenge checkpoints
      ↓
Reconstruct payload
      ↓
Submit flag
```

The challenge is designed so that each step provides evidence useful for
the next step.

## Reproducibility

The challenge can be regenerated using:

``` bash
python -m src.generate
```

This recreates:

``` text
artifacts/clean.pt
artifacts/challenge_000.pt
...
artifacts/challenge_024.pt
```

The deterministic seed and training configuration are defined in the
source code.

The generated artifacts are the runtime inputs used by the challenge
service.

## Acceptance Criteria & Difficulty Calibration
```
The challenge was evaluated using an agent-based solving setup rather than only the deterministic reference solution. The evaluation agent was given access to the challenge service and checkpoint artifacts and was allowed to interact with the environment through the defined challenge interface.

Initial calibration runs showed that the agent was able to identify the checkpoint structure, inspect the Adam optimizer state, locate the eight-element `exp_avg` residue channel, recover the encoded bytes, reconstruct the payload, and progress through the staged reward system.

The current calibration target is a 16-turn interaction budget. The agent is evaluated across repeated independent rollouts, with each rollout starting from a clean challenge state. The evaluation records successful flag recovery, total reward, number of turns used, and wall-clock solve time. Environment failures are tracked separately from genuine unsuccessful attempts so that infrastructure reliability does not get confused with task difficulty.

The reward structure provides six observable milestones — `recon`, `rollback`, `channel`, `residue`, `payload`, and `flag` — allowing the agent to receive meaningful partial reward before completing the entire challenge. This makes the task suitable for iterative agent training rather than relying only on the final flag as a binary success signal.

 ```



## Testing

The repository contains test and experiment material separately from the
runtime challenge.

Tests can be run from the development environment using the project's
Python test setup.

The Docker runtime intentionally does not execute tests automatically.

This separation prevents challenge startup from becoming dependent on
development-only tooling.

## Expected End-to-End Result

A complete local validation consists of:

``` text
1. Build Docker image
2. Start challenge service
3. Query challenge metadata
4. Run solver agent
5. Submit all six milestones
6. Reach reward 100
7. Recover the flag
```

The expected final result is:

``` text
PROVUE{OPTIMIZER_AUTOPSY}
```

## Final Summary

Rollback Residue is a forensic CTF about incomplete ML rollback.

A neural network and an auxiliary parameter are trained with Adam. A
hidden byte influences a gradient event. The resulting parameter update
is rolled back, but the optimizer state is deliberately left untouched.
The solver must recognize the mismatch, identify the persistent Adam
state, extract the residue, recover the encoded bits, reconstruct the
payload, and submit the final flag through the challenge service.

The challenge combines ML internals, checkpoint analysis, optimizer
behavior, forensic reasoning, deterministic artifact generation,
HTTP-based agent interaction, and staged machine-checkable rewards in a
compact CPU-friendly environment.

The final challenge demonstrates that restoring visible model parameters
is not necessarily equivalent to restoring the complete state of an ML
training process.
