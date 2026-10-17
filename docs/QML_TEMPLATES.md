# Reusable QML building blocks

These are unreleased source-checkout additions. From this repository, install
`python -m pip install -e '.[tutorial,visualization]'`. They are not present in
the previously published PyPI 1.1.0 wheel.

Every helper appends ordinary NexQus gates and returns the same Circuit. You
can draw, bind, compose, invert or compile the result using the regular APIs.
Angles use radians, and wires retain the order you supply.

| Helper | Input and behavior |
|---|---|
| `basis_encoding(q, bits, wires=None)` | One integer 0/1 per selected wire; applies X for 1. Prepares that basis state when starting in zero. On other inputs it applies XOR. |
| `angle_encoding(q, features, wires=None, rotation='Y', scale=1.0)` | 1..number-of-wires numeric or symbolic features; applies RX/RY/RZ to successive selected wires. Unused wires are untouched. |
| `entangling_layer(q, wires=None, entanglement='linear', entangler='CX')` | CX/CZ with linear, ring, full or explicit ordered wire-pair connectivity. |
| `layer_shape(layers, wires, rotations='Y')` | Returns `(layers, wires, len(rotations))`. |
| `variational_layers(q, weights, wires=None, rotations='Y', entanglement='linear', entangler='CX')` | Each layer applies the specified rotations on each wire, then the entangling edges. |
| `data_reuploading(q, features, weights, wires=None, encoding_rotation='Y', scale=1.0, rotations='Y', entanglement='linear', entangler='CX')` | Each layer encodes the same features again, then applies trainable rotations and entangling edges. |

`rotation`, `scale`, `rotations`, `entanglement` and `entangler` are keyword-only.
Both rotation weights and encoded features accept Parameters and arithmetic
Expressions. `scale` is a finite real constant. The helpers do not fit a scaler;
fit preprocessing on the training data and apply the same transform to test data.

## Build and differentiate a model

```python
import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction, templates

x0, x1 = Parameter('x0'), Parameter('x1')
shape = templates.layer_shape(2, 2, rotations='YZ')
weights = np.array([Parameter(f'w{i}') for i in range(np.prod(shape))],
                   dtype=object).reshape(shape)
q = templates.data_reuploading(Circuit(2), [x0, x1], weights,
                              rotations='YZ', entangler='CZ')
function = QuantumFunction(q, 'ZI')
bindings = {'x0': .2, 'x1': -.4,
            **{f'w{i}': .1 for i in range(weights.size)}}
value, gradients = function.value_and_grad(bindings, wrt=['w0', 'w1'])
print(value, gradients)
print(q.bind(bindings).draw())
print(q.depth(), q.count_ops(gates_only=True))
```

The last weight dimension is present even for a single axis: two Y-rotation
layers on three wires need shape `(2, 3, 1)`. Rotations apply left to right;
`'YZ'` means RY followed by RZ on each wire. Parameter names can be reused across
layers. QuantumFunction sums their gate-local derivatives with the arithmetic
chain rule. Gradients of both symbolic data and weights are supported by that
API; choose `wrt` when only weights should be optimized. This is not a PyTorch
autograd integration.

## Choose connectivity

For selected wires `[3, 0, 2]`:

| Connectivity | Ordered edges |
|---|---|
| `'linear'` | `(3,0), (0,2)` |
| `'ring'` | `(3,0), (0,2), (2,3)` |
| `'full'` | `(3,0), (3,2), (0,2)` |
| `[(2,3)]` | The explicitly supplied directed edge |
| `[]` | No entangling gates |

One wire produces no automatic edges. A two-wire ring has one edge, avoiding
two identical consecutive CX gates. Explicit pairs use actual circuit indices
and must join distinct selected wires without duplicate pairs. These edges
describe the logical circuit; the helpers do not route to a hardware device.

Run `python examples/qml_templates.py` or notebook
[07_inspection_and_qml_templates.ipynb](../notebooks/07_inspection_and_qml_templates.ipynb).
The existing [learning guide](LEARNING.md) covers optimizers and fitted models.

မြန်မာအကျဉ်းချုပ် — Data encoding နှင့် trainable layer များကို helper function
ဖြင့် ပြန်သုံးနိုင်ပါသည်။ Parameter များကို bind လုပ်ပြီး circuit ပုံ၊ depth နှင့်
gate count များကို စစ်ဆေးနိုင်ပါသည်။
