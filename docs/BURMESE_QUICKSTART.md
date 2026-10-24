# NexQus SDK 1.3.0 စတင်အသုံးပြုနည်း

NexQus ဖြင့် noiseless circuit၊ noise ပါသော circuit နှင့် measurement/reset
protocol များကို မိမိကွန်ပျူတာပေါ်တွင် simulation လုပ်နိုင်ပါသည်။ NumPy CPU ကို
ပုံမှန်အသုံးပြုပြီး လိုအပ်ပါက PyTorch CPU သို့မဟုတ် NVIDIA CUDA GPU ကို ရွေးနိုင်ပါသည်။
စတင်လေ့လာရန် quantum hardware သို့မဟုတ် GPU မလိုအပ်ပါ။

## ၁။ Install လုပ်ပါ

Python 3.10 နှင့်အထက်ရှိသော virtual environment တွင် အောက်ပါ command ကို run ပါ။

```bash
python -m pip install -e .
```

## ၂။ ပထမဆုံး circuit ကို run ပါ

```python
from nexqus import Circuit, Simulator

# qubit နှစ်ခုကို entangle လုပ်ပါ။
q = Circuit(2).h(0).cx(0, 1)
result = Simulator().run(q, shots=1000, seed=42)
print(result.probabilities)
print(result.counts)
```

Probability သည် [0.5, 0, 0, 0.5] ဖြစ်ပြီး count တွင် 00 နှင့် 11 ကို
ခန့်မှန်းအားဖြင့် တစ်ဝက်စီရရှိပါမည်။ Count သည် random sampling ကြောင့် အနည်းငယ်ကွာနိုင်ပါသည်။
Bitstring ကို ဘယ်ဘက်မှ q0, q1, ... အစဉ်ဖြင့် ဖတ်ပါ။ Angle များကို radians ဖြင့် သတ်မှတ်ပါ။

## ၃။ ပြန်သုံးနိုင်သော function ရေးပါ

```python
from nexqus import circuit

@circuit(2)
def bell_pair(q):
    # ပထမ qubit ကို superposition ဖြစ်စေပါ။
    q.h(0)
    q.cx(0, 1)

result = Simulator().run(bell_pair(), shots=1000, seed=42)
print(result.counts)
```

## ၄။ Noise ပါသော simulation ကို စမ်းပါ

```python
from nexqus import Noise, NoiseModel

noise = NoiseModel(Noise.depolarizing(0.02))
exact = Simulator().run(bell_pair(), noise=noise, shots=0)
sampled = Simulator(method='trajectory').run(
    bell_pair(), noise=noise, trajectories=1000,
    batch_size=32, shots=1000, seed=42,
)
print(exact.probabilities)
print(sampled.probabilities)
```

Noise ရှိသော default simulation သည် exact density matrix ကို အသုံးပြုပါသည်။
Trajectory mode သည် sampled estimate ဖြစ်ပြီး exact ရလဒ်နှင့် အနည်းငယ်ကွာနိုင်ပါသည်။
Trajectory အရေအတွက်များလာလျှင် Monte Carlo error လျော့နိုင်ပါသည်။ Batch size ကြီးလျှင်
memory ပိုလိုအပ်ပါသည်။ shots တိုးခြင်းသည် trajectory အရေအတွက်ကို မတိုးစေပါ။

## ၅။ Parameter များကို ပြောင်းလဲစမ်းသပ်ပါ

```python
from nexqus import Parameter

theta = Parameter('theta')
template = Circuit(2).ry(theta, 0).cx(0, 1)
sim = Simulator()
results = sim.run_sweep(template, [{'theta': 0.2}, {'theta': 0.8}], seed=42)
for result in results:
    print(result.metadata['bindings'], result.counts)
```

Circuit ကို တစ်ကြိမ်ရေးပြီး parameter တန်ဖိုးများကို ပြောင်းလဲ run နိုင်ပါသည်။
ထပ်ခါတလဲလဲ run မည့် circuit အတွက် compile ပြုလုပ်ထားနိုင်ပါသည်။

```python
prepared = sim.compile(template.bind({'theta': 0.8}))
print(prepared.run(shots=1000, seed=42).counts)
```

## ၆။ Gradient နှင့် learning ကို အသုံးပြုပါ

```python
from nexqus import QuantumFunction

function = QuantumFunction(Circuit(1).ry(theta, 0), 'Z')
value, gradients = function.value_and_grad({'theta': 0.4})
print(value, gradients)
```

QuantumFunction သည် expectation value နှင့် parameter-shift gradient ကို တွက်ပေးပါသည်။
QuantumRegressor နှင့် QuantumBinaryClassifier ကို Adam သို့မဟုတ် SGD ဖြင့် train နိုင်ပါသည်။
အသေးစိတ်ကို [Learning guide](LEARNING.md) တွင် ကြည့်ပါ။

## ၇။ Measurement၊ reset နှင့် feed-forward

```python
q = Circuit(1).h(0).measure(0, 'm').x(0).c_if('m', 1)
result = Simulator().run(q, shots=1000, seed=42)
print(result.counts)             # {'0': 1000}
print(result.classical_counts)   # intermediate measurement ရလဒ်များ

reset = Simulator().run(Circuit(1).h(0).reset(0), shots=0)
print(reset.probabilities)       # [1, 0]
```

Measurement ရလဒ်ကို classical bit တွင် သိမ်းပြီး နောက် gate ကို ထိန်းချုပ်နိုင်ပါသည်။
Reset သည် ရွေးချယ်ထားသော qubit ကို |0> သို့ ပြန်ထားပါသည်။ Teleportation နှင့်
အသေးစိတ်ကို [Dynamic circuits guide](DYNAMIC_CIRCUITS.md) တွင် ကြည့်ပါ။

## ၈။ Algorithm နှင့် GPU

```python
from nexqus import algorithms

result = Simulator().run(algorithms.grover(3, marked='101'), seed=42)
print(result.most_likely)
```

Bell၊ GHZ၊ QFT၊ Grover၊ Bernstein–Vazirani၊ Deutsch–Jozsa၊ phase estimation၊
QAOA MaxCut နှင့် teleportation circuit များ ပါဝင်ပါသည်။ Version 1.1.0 တွင်
Shor factoring နှင့် quantum order finding ကိုလည်း အသုံးပြုနိုင်ပါသည်။

```python
print(algorithms.shor_resources(35))
factored = algorithms.shor(35, base=2, seed=42)
print(factored.factors)  # (5, 7)
print(factored.method)   # quantum
```

Number ကြီးလာလျှင် statevector memory သည် အဆတိုးလာပါသည်။ မစမ်းမီ
`shor_resources()` ဖြင့် စစ်ဆေးပါ။ ဤဥပမာသည် RSA-2048 ကို factor လုပ်နိုင်သည်ဟု မဆိုလိုပါ။

မိမိတည်ဆောက်ထားသော circuit ကို simulation မလုပ်ဘဲ ပုံဆွဲနိုင်ပါသည်။
`cx` သည် CNOT ဖြစ်ပြီး gate မရှိသော wire များကိုလည်း ပုံတွင် မြင်ရပါမည်။

```python
custom = Circuit(8).rx(.5, 0).cx(0, 1).ry(.8, 3).cx(3, 7)
print(custom.draw())
custom.draw('svg', filename='custom.svg')  # Jupyter တွင် SVG ပုံကိုလည်း ပြပါသည်။
```

Symbolic gate တွင် parameter expression ကို ပြပြီး concrete gate တွင် gate name
ကို ပြပါသည်။ မူလ numerical angle ကို circuit representation တွင် မသိမ်းထားပါ။

GPU အသုံးပြုရန် မိမိ CUDA environment နှင့်ကိုက်ညီသော PyTorch ကို
[တရားဝင် PyTorch လမ်းညွှန်](https://pytorch.org/get-started/locally/) အတိုင်း install လုပ်ပါ။
ထို့နောက် backend နှင့် device ကို အောက်ပါအတိုင်း ရွေးပါ။

```python
# CUDA-enabled PyTorch နှင့် NVIDIA GPU ရှိသောစက်တွင်သာ run ပါ။
gpu = Simulator(backend='torch', device='cuda:0')
result = gpu.run(Circuit(2).h(0).cx(0, 1), shots=1000, seed=42)
print(result.counts)
```

Qubit တိုးလာသည့်အခါ memory လိုအပ်ချက်သည် အဆတိုးလာပါသည်။ GPU သည် circuit
အားလုံးအတွက် CPU ထက် ပိုမြန်မည်ဟု မယူဆပါနှင့်။ မိမိ workload ကို တိုင်းတာပါ။
[Performance guide](PERFORMANCE.md) တွင် method၊ memory နှင့် benchmark အသုံးပြုပုံကို ကြည့်နိုင်ပါသည်။

ဆက်လက်လေ့လာရန် [API reference](API.md) နှင့်
[English/Burmese notebooks](../notebooks/README.md) ကို ဖတ်ပါ။
