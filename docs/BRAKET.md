# Optional Amazon Braket adapter

This bridge is included in the 1.3.0 release. The published
1.2.0 wheel does not include it. No cloud job is submitted by installation,
imports, preparation, the example, or CI. All development validation uses local
simulation or mocked AWS tasks. Real QPU execution has not been verified.

## Install

Braket's SDK requires Python 3.11+, while NexQus core remains Python 3.10+.
From this checkout, in an activated environment:

```bash
python -m pip install -e '.[braket]'
python examples/braket_local.py
python examples/braket_local.py --local
```

The first example command only prepares a plan. `--local` uses Braket's local
CPU simulator, with no account, credentials, AWS connection or cloud charges.
The comparison SDK is an optional execution dependency only for this bridge;
NexQus's own simulator continues to use its own kernels.

## Prepare and inspect offline

```python
from nexqus import Circuit
from nexqus.braket import prepare_braket

q = Circuit(8).rx(.5, 0).cx(0, 1).ry(.8, 3).cx(3, 7)
plan = prepare_braket(q, shots=1000)
print(plan.gate_counts)
native_circuit = plan.to_braket()  # local conversion; imports optional SDK
result = plan.run_local()         # local CPU simulation only
print(result.counts)
print(result.expectation('ZZIIIIII'))
```

`prepare_braket` works without installing Braket or importing boto3. It takes
an immutable snapshot, including bound parameter values. To bind symbolic
circuits, pass `bindings={'theta': .5}`. Preparation and the backend constructor
never contact AWS. `to_braket()` creates a fresh native circuit on each call.
Every wire is measured at the end; idle wires receive identity instructions.

Supported operations are H/X/Y/Z, RX/RY/RZ, P (including S/T), CX/CZ/CP and
SWAP, including their supported inverses and bound templates. Actual matrices
are verified; custom names or display parameters cannot redefine execution.
Noise, measurement/reset/feed-forward, unresolved parameters, compact
permutations, arbitrary matrices and general multiwire diagonals are rejected.
In particular, NexQus's compact Shor modular permutations need an elementary
gate decomposition before they can run on hardware.

`plan.validate_device(capabilities)` accepts a Braket device-capability dict
offline and checks OpenQASM support, declared operations, capacity and shot
range. It is conservative: unsupported gates are rejected rather than silently
decomposed. Logical wires are not fixed physical qubits; Braket/provider
compilation handles mapping and routing. Passing preflight does not guarantee
physical execution, fidelity or current queue availability.

## Results and bit order

`BraketResult` provides:

- `counts`: sparse bitstring-to-count dictionary, with q0 leftmost as in NexQus.
- `probabilities`: sparse empirical frequencies, not exact state probabilities.
- `most_likely` and `expectation(pauli)`: expectations support only I/Z strings
  measured in the computational basis. X/Y need separate measurement circuits.
- `metadata`: shot count, measured wire ordering and backend/task identifiers.
- `to_result(max_qubits=20)`: explicitly allocate a dense NexQus `Result` for
  compatibility with existing probability/marginal code; no state is invented.

Conversion checks the provider's measured-qubit ordering and total shots. It
rejects missing/duplicate wires and malformed counts. Hardware does not return
a full statevector or density matrix. Sparse results avoid allocating 2**n
probabilities for a large QPU. `run_local(max_qubits=20)` also has an explicit
qubit cap; raising it requires sufficient local simulator memory.

## AWS access and paid submission

This section describes the API for future use. Do not execute it unless you
intend to pay for a cloud task. The example and tests never invoke real AWS.

`BraketBackend(device_arn, aws_session=None)` uses the official AWS credential
chain or a supplied Braket `AwsSession`. The adapter takes no secret-key
arguments. Use your own AWS account and normal IAM permissions for Braket and
the configured S3 destination. No cross-account role for NexQus is needed.

`backend.inspect_device()` is an explicit network read for current device
metadata. It does not submit a task. Its `.properties.dict()` can be passed
to `plan.validate_device`.

For a future QPU submission, call:

```text
job = backend.submit(
    plan,
    allow_paid=True,
    task_price_usd=<current price for this device>,
    shot_price_usd=<current price per shot>,
    max_estimated_cost_usd=<your estimate ceiling>,
    s3_destination_folder=(<bucket>, <prefix>),
)
```

Without the literal `allow_paid=True`, submission raises before importing
Braket or contacting AWS. Missing/invalid prices, an exceeded estimate ceiling,
unsupported gates and incompatible device capabilities prevent task creation.
The circuit is validated locally before device metadata is read. Successful
submission invokes `AwsDevice.run` once and returns a `BraketJob`.

The estimate is **task price + shots × shot price**, using prices supplied by
the caller. No live price is guessed or fetched automatically. For illustration
only, a hypothetical $0.30/task plus $0.001/shot is $1.30 for 1,000 shots; these
are not current prices for any particular QPU. This estimate is not an AWS
spending limit and excludes taxes, S3, reservations and optional services.
Use the provider's current pricing and AWS spending controls before paid use.
Duration-billed cloud simulators (SV1/DM1/TN1) are deliberately rejected until
time-based budgeting is implemented. `run_local()` remains available.

## Job lifecycle

```text
job.id                 # save the quantum-task ARN
job.state()            # provider status, network read
job.result()           # polls up to 300 seconds per SDK task; retrieves from S3
job.cancel()           # cancellation request; billing may already have occurred
backend.get_job(task_arn, n_qubits=8, shots=1000)  # reconnect, never resubmit
```

`get_job` verifies the task device and shots against AWS metadata. Supply the
original logical qubit count; final result conversion validates measured wires.
Failed/cancelled/timed-out tasks do not turn into empty success results. A
missing result raises; inspect status and reconnect/poll when appropriate.
There are no adapter-level retries or automatic device substitutions. If a
network error occurs during submission, a task may already exist: inspect AWS
before submitting again to avoid duplicate charges.

## Verification

```bash
python -m unittest discover -s tests -p test_braket.py -v
python -m unittest discover -s tests/braket -v
```

The integration suite blocks botocore endpoint requests. It compares exact
local Braket statevectors against NexQus for gates, QFT, QML layers, inverses,
nonadjacent wires and idle wires; paid submission/lifecycle tests use mocks.
CI has no AWS credentials and submits no cloud tasks.

References: [Braket tasks](https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks-to-braket.html),
[AWS SDK device API](https://amazon-braket-sdk-python.readthedocs.io/en/stable/_apidoc/braket.aws.aws_device.html),
[task lifecycle](https://amazon-braket-sdk-python.readthedocs.io/en/stable/_apidoc/braket.aws.aws_quantum_task.html),
[cost controls](https://docs.aws.amazon.com/braket/latest/developerguide/braket-pricing.html).
