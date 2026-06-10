# CEEUS 2026 — ModulUS Power & Architecture Sandbox

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/luuleitner/test_colab_ceeus/blob/main/ceeus2026_modulus_power.ipynb)

Teaching sandbox for the **IEEE CEEUS 2026 Wearables Workshop** (Warsaw, 22–24 Jun 2026),
by Christoph Leitner (ETH Zurich, Integrated Systems Laboratory).

It makes the *system–power–data* trade-off of wearable ultrasound **computable**: starting
from one real ModulUS measurement, you walk the spine

```
resolution → frequency → Nyquist → data rate → power → battery → wearability
```

forward (what it costs) and backward (given a wrist budget, what architecture survives).

## Run it

Click the **Open in Colab** badge above — the first cell installs everything and clones
this repo. No local setup needed.

To run locally instead:

```
pip install -r requirements.txt
jupyter lab ceeus2026_modulus_power.ipynb
```

## Contents

| File | What |
|---|---|
| `ceeus2026_modulus_power.ipynb` | the guided exercise (Acts 1–3) |
| `modulus.py` | the ModulUS digital twin (Ping / Echo / Core / System + Transducer, Radio, Battery) |
| `requirements.txt` | dependencies (all pre-installed on Colab) |

The notebook also imports the [dasIT](https://github.com/luuleitner/dasIT) toolbox for
signal-domain helpers.

## License

Intended: Apache License 2.0 (matching [dasIT](https://github.com/luuleitner/dasIT)).
`LICENSE` file to be added.
