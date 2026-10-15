# NexQus SDK 0.4 စတင်အသုံးပြုနည်း

v0.4 တွင် circuit parameter များ၏ gradient ကို parameter-shift နည်းလမ်းဖြင့်
တွက်ချက်ပြီး Adam သို့မဟုတ် SGD ဖြင့် train နိုင်ပါသည်။

```bash
python examples/gradients.py
python examples/train_sine.py
python examples/train_classifier.py
```

အသုံးပြုပုံနှင့် ကန့်သတ်ချက်များကို `V04_GUIDE.md` တွင် ကြည့်ပါ။

v0.3 တွင် noise sampling ကို ရွေးချယ်ထားသော CPU သို့မဟုတ် GPU ပေါ်တွင်
တိုက်ရိုက်လုပ်ဆောင်ပါသည်။ Performance နှင့် validation command အသစ်များအတွက်
`V03_GUIDE.md` ကို ကြည့်ပါ။

## v0.2 တွင် parameter များကို ပြောင်းလဲစမ်းသပ်ပါ

```python
from nexqus import Circuit, Parameter, Simulator

theta = Parameter('theta')
template = Circuit(2).ry(theta, 0).cx(0, 1)
sim = Simulator()
results = sim.run_sweep(template, [{'theta': 0.2}, {'theta': 0.8}], seed=42)
for result in results:
    print(result.metadata['bindings'], result.counts)
```

Circuit ကို တစ်ကြိမ်ရေးပြီး parameter တန်ဖိုးများကို ပြောင်းလဲအသုံးပြုနိုင်ပါသည်။
`sim.compile(template.bind({'theta': 0.8}))` ဖြင့် device tensor များကို
ကြိုတင်ပြင်ဆင်ထားနိုင်ပါသည်။ ရရှိလာသော object ၏ `.run()` ကို ပြန်လည်ခေါ်နိုင်ပါသည်။

Noisy trajectory များကို တစ်စုစီတွက်ရန် `batch_size=32` ကို သုံးနိုင်ပါသည်။
Batch size ကြီးလျှင် memory ပိုလိုအပ်ပါသည်။ အသေးစိတ်နှင့် benchmark command များကို
README တွင် ကြည့်ပါ။

ဤ SDK ဖြင့် မိမိကွန်ပျူတာပေါ်တွင် quantum circuit များကို simulation လုပ်နိုင်ပါသည်။
စတင်အသုံးပြုရန် quantum hardware မလိုအပ်ပါ။

## ၁။ Install လုပ်ပါ

ZIP file ကို ဖြည်ပြီး `nexqus-sdk-v0.4.0` folder ထဲတွင် terminal ဖွင့်ပါ။

```bash
python -m pip install -e .
python examples/quickstart.py
```

## ၂။ Function တစ်ခုရေးပါ

```python
from nexqus import circuit, Simulator

@circuit(2)
def bell_pair(q):
    # qubit 0 ကို superposition ဖြစ်စေပါ။
    q.h(0)
    # qubit 0 နှင့် qubit 1 ကို entangle လုပ်ပါ။
    q.cx(0, 1)

result = Simulator().run(bell_pair(), shots=1000, seed=42)
print(result.counts)
```

`@circuit(2)` သည် qubit နှစ်ခုပါသော circuit ကို ဖန်တီးပေးပါသည်။
`shots=1000` သည် terminal measurement ကို အကြိမ် ၁၀၀၀ sample ယူခြင်းဖြစ်ပါသည်။
`seed` သတ်မှတ်ထားပါက တူညီသော environment တွင် random ရလဒ်များကို ပြန်လည်စမ်းသပ်နိုင်ပါသည်။
ဤ circuit တွင် `00` နှင့် `11` ကို ခန့်မှန်းအားဖြင့် တစ်ဝက်စီ ရရှိပါမည်။

Bitstring ကို ဘယ်ဘက်မှ `q0, q1, ...` အစဉ်ဖြင့် ဖတ်ပါ။

## ၃။ Noise ထည့်ပါ

```python
from nexqus import Noise, NoiseModel

noise = NoiseModel(Noise.depolarizing(0.02))
result = Simulator().run(bell_pair(), noise=noise, seed=42)
print(result.probabilities)
```

Noise model ကို unitary gate တစ်ခုစီပြီးနောက် သက်ဆိုင်ရာ qubit တစ်ခုစီတွင် ထည့်ပါသည်။
Default mode သည် noise ရှိပါက density matrix ကို အသုံးပြုပါသည်။

Qubit အများအပြားနှင့် noise စမ်းသပ်လိုပါက memory သက်သာသော trajectory mode ကို ရွေးနိုင်ပါသည်။
၎င်းသည် sampled estimate ဖြစ်ပြီး exact density matrix နှင့် ရလဒ် အနည်းငယ်ကွာနိုင်ပါသည်။

```python
result = Simulator(method="trajectory").run(
    bell_pair(), noise=noise, trajectories=1000, seed=42
)
```

## ၄။ ပြင်ဆင်ပြီးသား algorithm ကို အသုံးပြုပါ

```python
from nexqus import algorithms

result = Simulator().run(algorithms.grover(3, marked="101"), seed=42)
print(result.most_likely)
```

QFT, inverse QFT, GHZ, Bernstein–Vazirani, Deutsch–Jozsa, phase estimation နှင့်
QAOA MaxCut circuit များလည်း ပါဝင်ပါသည်။ အသေးစိတ်ကို README တွင် ကြည့်ပါ။

## ၅။ GPU အသုံးပြုပါ

```python
sim = Simulator(backend="torch", device="cuda:0", precision="single")
```

CUDA အသုံးပြုနိုင်သော PyTorch နှင့် NVIDIA GPU လိုအပ်ပါသည်။
ဤ release ပြင်ဆင်ရာ environment တွင် GPU စမ်းသပ်မှု မပြုလုပ်နိုင်ခဲ့ပါ။
မိမိစက်တွင် test များကို အရင် run ပါ။ CPU သုံးရန် `Simulator()` ကိုသာ အသုံးပြုပါ။
