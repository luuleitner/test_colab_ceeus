"""Builder for the CEEUS 2026 workshop Colab notebook (notebook-as-code).

Run:  py -3.12 build_notebook.py
Output: 2026ceeus_student.ipynb + 2026ceeus_teacher.ipynb at the repo root.

Cells are appended in order; extend CELLS as we build act by act. Keeping the
notebook generated (not hand-edited JSON) gives clean diffs and one source.
"""
import nbformat as nbf
from pathlib import Path

SLUG = "luuleitner/test_colab_ceeus"             # GitHub slug (Colab mirror)
NB_NAME = "2026ceeus_student.ipynb"              # student version (TODO blanks)
NB_TEACHER = "2026ceeus_teacher.ipynb"           # teacher version (blanks filled)
RAW = f"https://raw.githubusercontent.com/{SLUG}/main"   # raw base for assets/ images
REPO_ROOT = Path(__file__).resolve().parent

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
CELLS = []

# ── C0 · front matter ────────────────────────────────────────────────────
CELLS.append(md(f"""\
# Anatomy of a Wearable Ultrasound System

### From Components to Signals
#### *Marco Giordano and Dr. Christoph Leitner (ETH Zurich, Switzerland)*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{SLUG}/blob/main/{NB_NAME})

<p align="left"><img src="{RAW}/assets/modulUS.jpg" width="520" alt="The ModulUS wearable-ultrasound platform"></p>

A wearable ultrasound system must do the job of a cart-sized scanner, but using only a patch on the skin, powered by a battery small enough to wear. 
<br> This workshop takes a wearable ultrasound system apart, from front-end components to the digitized signal, to expose **where the energy actually goes, and for what**.
```
size you start from ──────────────────────────► size you design toward
   cart scanner            handheld probe            wearable patch
   ~200 L                     ~200 mL                  ~3 mL
   ≈ a bathtub                ≈ a coffee mug           ≈ a teaspoon
        └─────────  same job, a bathtub poured into a teaspoon (~70,000×)  ─────────┘
```
<br>We run a live acquisition on our **ModulUS** system (a sandbox for wearable-ultrasound development). You then take a hands-on look at the recorded RF and envelope data.
<br>Using that data, you analyse **why receive-channel scaling must be approached carefully** in wearable ultrasound, and what impact excitation frequency has beyond improved resolution.

#### What you will do
```
echo → frequency → sampling → data rate → power → battery → wearable
└─ Exercise 1 ─┘   └─ Exercise 2 ─────┘   └─ Exercise 3 ───────────┘
```
1. ***Exercise 1***: read one real ModulUS echo as **RF** and as **envelope**, and find the frequency it carries.
2. ***Exercise 2***: digitize that echo, compute its **data rate**, and watch it slam into the **on-chip ADC** and **wireless-link** walls.
3. ***Exercise 3***: weigh what to transmit (**RF**, **envelope**, or **on-device features**) to bring **power** and **battery** within a **small wearable's budget** — and settle on a **receive-channel count it can sustain**.
"""))

# ── C1 · bootstrap (robust on Colab, also runs locally) ──────────────────
CELLS.append(code(f"""\
# === Bootstrap — run me first ===============================================
# Robust on Google Colab (ephemeral VM) and when run locally from the repo.
import sys, os

IN_COLAB = "google.colab" in sys.modules
SLUG = "{SLUG}"                       # GitHub repo (Colab mirror)
REPO = SLUG.split("/")[1]

if IN_COLAB:
    # external toolbox: dasIT signal/plot functions (install once per session)
    try:
        import dasIT  # noqa: F401
    except ImportError:
        !pip install -q git+https://github.com/luuleitner/dasIT
    # our sandbox repo: clone so you can browse modulus.py and the demo data
    if not os.path.isdir(REPO):
        !git clone -q https://github.com/{{SLUG}}
    sys.path.insert(0, REPO)
    DATA = os.path.join(REPO, "example_data", "modulus_demo.npz")
else:
    sys.path.insert(0, ".")          # local: repo root on the path
    DATA = os.path.join("example_data", "modulus_demo.npz")

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False}})
from modulus import System, Acq, load_traces, fom_mw_per_mhz
from dasIT.features.signal import fftsignal, analytic_signal, envelope

print("ready —", "Colab" if IN_COLAB else "local")
"""))

# ── C2 · tools & repositories ────────────────────────────────────────────
CELLS.append(md(f"""\
### Tools & repositories

This notebook stands on two pieces, both pulled in by the bootstrap above:

| what | role here | link |
|---|---|---|
| **dasIT** | signal toolbox — spectra, analytic signal, envelope (`fftsignal`, `analytic_signal`, `envelope`) | [github.com/luuleitner/dasIT](https://github.com/luuleitner/dasIT) |
| **ModulUS sandbox** (this repo) | the system digital-twin you design against — `System`, `Acq`, `modulus.py`, and the demo echo in `example_data/` | [github.com/{SLUG}](https://github.com/{SLUG}) |

Everything runs on Colab as-is — nothing to install by hand. The full method is in the ModulUS paper (see **References** at the end).
"""))

# ── EXERCISE 1 · SIGNAL — the recorded echo ──────────────────────────────
CELLS.append(md("""\
## Exercise 1 · The signal — *what the front-end produces*

Ultrasound is an echo game. The probe sends a short **pulse** (a few cycles of sound) then listens for what bounces back from each tissue boundary. The later an echo arrives, the deeper the reflector that made it.

```
        pulse out  ∿∿►
 ┌────┐ ───────────────────────────────────────────►
 │ Tx │           ║          ║                ║
 │ Rx │ ◄───────────────────────────────────────────
 └────┘  echoes in  ∿         ∿∿               ∿
                   skin     muscle            bone
 t = 0 ───────────────────────────────────────────►  time  (depth = c·t / 2)
```

ModulUS hands you the *same* echo in two forms:

- **RF** — the raw radio-frequency waveform straight off the transducer (Pulse → Core), carrier and all.
- **Envelope** — that same RF after the analog **Echo** board strips the carrier, leaving only the echo's shape.

Both place the reflectors at the same depth. What sets them apart is **bandwidth** — and that single difference is the lever the rest of this notebook turns on. Open one real measurement and look.
"""))

CELLS.append(code("""\
# Load ONE ModulUS measurement (real .npz if present, else a labeled synthetic stand-in)
rf, env, fs = load_traces(DATA)
t_us = np.arange(len(rf)) / fs * 1e6          # time axis [µs]

plt.figure(figsize=(8, 3))
plt.plot(t_us, rf, lw=0.8)
plt.xlabel("time [µs]"); plt.ylabel("amplitude")
plt.title(f"Raw RF echo   (fs = {fs/1e6:.0f} Msps,  {len(rf)} samples)")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""\
# Same echo, two representations — in time and in frequency.
# Spectra via dasIT's Welch helper (returns frequency already in MHz).
f_rf,  P_rf  = fftsignal(rf,  fs)
f_env, P_env = fftsignal(env, fs)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(t_us, rf,  lw=0.7, label="RF")
ax[0].plot(t_us, env, lw=1.6, label="envelope")
ax[0].set_xlabel("time [µs]"); ax[0].set_ylabel("amplitude")
ax[0].set_title("time domain"); ax[0].legend()
ax[1].plot(f_rf,  P_rf,  label="RF")
ax[1].plot(f_env, P_env, label="envelope")
ax[1].set_xlabel("frequency [MHz]"); ax[1].set_ylabel("power")
ax[1].set_title("spectrum"); ax[1].set_xlim(0, fs/2e6); ax[1].legend()
plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""\
# TODO (a) — compute the power-weighted MEAN frequency of each spectrum.
# Formula (ModulUS paper, eq. 2):   f_mean = sum(f_i * P_i) / sum(P_i)
# Use the arrays from the previous cell: f_rf, P_rf  and  f_env, P_env.
raw_mean = ...     # <-- replace ... using f_rf and P_rf
env_mean = ...     # <-- replace ... using f_env and P_env

# ---- self-check (do not edit) -------------------------------------------
assert raw_mean is not ... and env_mean is not ..., "fill in raw_mean and env_mean above"
assert env_mean < raw_mean, "the envelope should sit at LOWER frequency than the RF"
ratio = raw_mean / env_mean
print(f"RF mean ~ {raw_mean:.2f} MHz    envelope mean ~ {env_mean:.2f} MHz")
print(f"bandwidth reduction ~ {ratio:.1f}x   (ModulUS paper: 2.9 -> 0.7 MHz ~ 4x)")
assert ratio > 2, "expected a clear (>2x) reduction - check your formula"
print("OK - spectral check passed")
"""))

CELLS.append(md("""\
### Resolution — what can this pulse *resolve*?

Exercise 1 paid for bandwidth in data; the transducer **frequency** buys something back — **spatial resolution**. A shorter pulse (higher frequency) splits two reflectors that a long pulse smears into one:

```
 two reflectors a hair apart:   ║║
   low  f  (long pulse):   ∿∿∿∿∿∿   →  one blob    (can't separate them)
   high f  (short pulse):  ∿∿  ∿∿   →  two echoes  (resolved)
```

Axial resolution is about half the spatial pulse length:

$$ \\text{axial\\_res} = n_\\text{cycles}\\cdot\\frac{\\lambda}{2}, \\qquad \\lambda = \\frac{c}{f_\\text{Tx}}, \\qquad c = 1540\\ \\text{m/s} $$

So higher $f_\\text{Tx}$ → shorter $\\lambda$ → finer detail. The catch, waiting in **Exercise 2**: finer detail is **more data**.

| $f_\\text{Tx}$ | $\\lambda$ | axial res (5 cyc) | resolves about |
|---|---|---|---|
| 1 MHz  | 1.54 mm  | 3.9 mm  | a grape |
| 5 MHz  | 0.31 mm  | 0.77 mm | a sesame seed |
| 10 MHz | 0.154 mm | 0.38 mm | a human hair (~0.07 mm) |
| 15 MHz | 0.103 mm | 0.26 mm | a dust mite |

"""))

CELLS.append(code("""\
# The resolution ladder, straight from the Transducer twin in modulus.py
from modulus import Transducer
print("f_Tx [MHz]   axial res [mm]")
for f in (1e6, 5e6, 10e6, 15e6):
    print(f"   {f/1e6:>5.0f}        {Transducer().axial_res(f)*1e3:.2f}")
"""))

# ── EXERCISE 2 · COST — digitize and move the echo ──────────────────────
CELLS.append(md("""\
## Exercise 2 · The cost — *digitizing and moving the echo*

A clean echo is worthless until you can get it off the probe. Digitizing it sets the **data rate** — and in a wearable, that data rate is a firehose aimed at a drinking straw. Start with **Nyquist**: to capture a signal you must sample at least twice its top frequency.

```
fs        = 2 · f_Tx                 (RF, full bandwidth)
t_acq     = 2 · D / c                (round trip to depth D)
N         = fs · t_acq               (samples per A-line)
data_rate = N · bits · PRF · nRx     (bits per second)   ← the number that matters
```

Between that firehose and the wearable stand two hard walls:

| wall | limit | cross it and... |
|---|---|---|
| **ADC wall** | on-chip ADC ~ **5 Msps** | you need an external converter + FPGA (bigger, hungrier) |
| **link wall** | usable BLE ~ **300 kb/s** | the stream will not fit the radio — the straw |

The analog **Echo** board (envelope) already cut bandwidth ~4× in Exercise 1, dropping fs to
5 Msps — just under the ADC wall. But does it clear the **link** wall? Compute it and see.
"""))

CELLS.append(code("""\
# TODO (b) — the data rate decides whether a wireless wearable can even exist.
#   data_rate = N · bits · PRF · nRx       [bits/s]
device = System()
acq = Acq(f_Tx=10e6, bits=12, nRx=8, PRF=100, D=0.03, mode="RF")
d = device.run(acq)       # d.N is computed by the model; acq.bits/PRF/nRx are knobs
data_rate = ...              # <-- replace ... using d.N, acq.bits, acq.PRF, acq.nRx

# ---- self-check (do not edit) -------------------------------------------
assert data_rate is not ..., "fill in data_rate above"
assert abs(data_rate - d.data_rate) < 1, "should match the model (Core.data_rate)"
wall = device.radio.throughput_max_bps
print(f"RF, 8 channels, 100 Hz  ->  {data_rate/1e6:.2f} Mb/s")
print(f"BLE ceiling {wall/1e3:.0f} kb/s   ->  {data_rate/wall:.0f}x OVER the wall" if data_rate > wall else "fits")
print("OK - data-rate check passed")
"""))

CELLS.append(code("""\
# The naive choice (RF) vs the analog trick (BWR envelope) — same 10 MHz, 8 ch, 100 Hz.
device = System()
print(f"{'mode':9s}{'fs':>9s}{'data rate':>12s}   ADC wall   BLE wall")
for mode in ("RF", "BWR"):
    d = device.run(Acq(10e6, 12, 8, 100, 0.03, mode))
    print(f"{mode:9s}{d.fs/1e6:7.1f}M {d.data_rate/1e6:9.2f} Mb/s   "
          f"{'fits ' if d.fits_onchip else ' EXT ':>5s}     "
          f"{'fits' if d.fits_ble else 'OVER'}")
print()
print("The Echo board fixed the ADC wall (envelope -> 5 Msps). The radio is still flooded.")
"""))

CELLS.append(code("""\
# STAGE 1 — you are LOCKED in RF mode. Try to get the data rate under the BLE wall
# by changing anything you like... including the battery. (Drag the sliders.)
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
device = System()
WALL = device.radio.throughput_max_bps / 1e6     # BLE wall [Mb/s], read from the model

def stage1(f_Tx_MHz=10.0, nRx=8, PRF=100, battery_days=1):
    d = device.run(Acq(f_Tx_MHz * 1e6, 12, nRx, PRF, 0.03, "RF"))   # mode LOCKED = RF
    dr = d.data_rate / 1e6
    b = device.battery.size(d.P_avg, days=battery_days)
    plt.figure(figsize=(7, 1.6))
    plt.barh([0], [dr], color=("seagreen" if d.fits_ble else "crimson"))
    plt.axvline(WALL, color="k", ls="--"); plt.text(WALL * 1.05, 0, f"BLE {WALL*1e3:.0f} kb/s", va="center")
    plt.yticks([]); plt.xlabel("data rate [Mb/s]"); plt.xlim(0, max(WALL * 2, dr * 1.1))
    plt.title(f"{dr:.2f} Mb/s  ->  {'FITS' if d.fits_ble else 'OVER'}      "
              f"battery {battery_days} d = {b['vol_cm3']:.2f} cm3")
    plt.tight_layout(); plt.show()

interact(stage1,
         f_Tx_MHz=FloatSlider(value=10, min=1, max=15, step=1, description="f_Tx [MHz]"),
         nRx=Dropdown(options=[1, 8, 16, 32], value=8, description="nRx"),
         PRF=IntSlider(value=100, min=25, max=1000, step=25, description="PRF [Hz]"),
         battery_days=IntSlider(value=1, min=1, max=7, description="battery [d]"));
"""))

CELLS.append(md("""\
### The battery was never the problem

You just felt it: **no battery size changes the BLE verdict.** Locked in RF, the only
way under the ~300 kb/s line is to throw away resolution (`f_Tx`), coverage (`nRx`), or
frame rate (`PRF`) — i.e. to stop doing the thing you came to do.

The data rate is set by how you **represent** the signal, not how big a battery you
carry. In **Exercise 3** we unlock that representation — and watch the wall move.
"""))

# ── EXERCISE 3 · SYSTEM — scaling receive channels ──────────────────────
CELLS.append(md("""\
## Exercise 3 · The system — *scaling receive channels*

You proved it: in RF the radio floods the link no matter the battery. Now unlock the
one knob we held back — how the echo is **represented** before it reaches the radio.
Think of it as how you mail a statue:

- **RF** — crate up the whole statue: every sample, the full waveform.
- **BWR** — ship a lightweight cast: the analog envelope, ~4× lighter, shape kept but fine detail (phase) lost.
- **features** — post its dimensions on a card: a handful of numbers per A-line, no waveform at all.

```
per A-line, what you put on the radio:
  RF        ████████████████   the whole statue
  BWR       ████               a light cast — ~4× less
  features  ▌                  dimensions on a card
```

Watch what each does to the **power breakdown**, the **link wall**, and the **battery**
you would have to wear.
"""))

CELLS.append(code("""\
# STAGE 2 — the mode is now UNLOCKED. Flip it to 'features' and watch the radio.
from ipywidgets import interact, FloatSlider, IntSlider, Dropdown
device = System()
CR2032_CM3 = 3.3        # reference coin-cell volume [cm3]

def stage2(f_Tx_MHz=10.0, nRx=8, PRF=100, mode="RF", days=1):
    d = device.run(Acq(f_Tx_MHz * 1e6, 12, nRx, PRF, 0.03, mode))
    P = d.power; b = device.battery.size(d.P_avg, days=days)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(list(P.keys()), [P[k] * 1e3 for k in P],
              color=["#888", "#4a90d9", "#7ab648", "#d9534f"])
    ax[0].set_ylabel("power [mW]"); ax[0].set_title(f"P_avg = {d.P_avg*1e3:.1f} mW")
    ax[1].bar(["this design", "CR2032"], [b["vol_cm3"], CR2032_CM3],
              color=["#d9534f", "#888"])
    ax[1].set_ylabel("volume [cm3]")
    ax[1].set_title(f"{b['vol_cm3']:.2f} cm3  =  {b['n_cr2032']:.1f} CR2032  ({days} d)")
    ble = "FITS" if d.fits_ble else f"{d.data_rate/1e6:.2f} Mb/s OVER"
    adc = "on-chip" if d.fits_onchip else "external/FPGA"
    fig.suptitle(f"mode = {mode}   |   BLE {ble}   |   ADC {adc}   |   "
                 f"FoM {d.fom:.2f} mW/MHz", fontsize=11)
    plt.tight_layout(); plt.show()

interact(stage2,
         f_Tx_MHz=FloatSlider(value=10, min=1, max=15, step=1, description="f_Tx [MHz]"),
         nRx=Dropdown(options=[1, 8, 16, 32], value=8, description="nRx"),
         PRF=IntSlider(value=100, min=25, max=1000, step=25, description="PRF [Hz]"),
         mode=Dropdown(options=["RF", "BWR", "features"], value="RF", description="mode"),
         days=IntSlider(value=1, min=1, max=7, description="battery [d]"));
"""))

CELLS.append(code("""\
# TODO (c) — even the envelope (BWR) floods the radio once you add channels.
# Find the SMALLEST nRx at which BWR breaks the BLE wall (10 MHz, PRF=100 Hz).
device = System()
breaking_nRx = None
for nRx in [1, 2, 4, 8, 16, 32]:
    d = device.run(Acq(10e6, 12, nRx, 100, 0.03, "BWR"))
    if ...:                      # <-- replace ... with the condition "no longer fits BLE"
        breaking_nRx = nRx
        break

# ---- self-check (do not edit) -------------------------------------------
assert breaking_nRx is not None, "fill in the condition above"
print(f"BWR breaks the BLE wall at nRx = {breaking_nRx} channels")
print("Your escape from there? -> features mode: ship numbers, not the waveform.")
"""))

CELLS.append(code("""\
# INVERT THE CHAIN — the design exercise.
# Given a SMALL-WEARABLE budget, which architectures actually survive every constraint:
# fits BLE  AND  fits on-chip ADC  AND  nRx <= 8 (hardware)  AND  battery <= budget.
BUDGET_CM3 = 3.0       # a coin-cell-sized wearable budget
DAYS = 1
device = System(); survivors = []
for mode in ("RF", "BWR", "features"):
    for f_MHz in (2, 5, 10, 15):
        for nRx in (1, 8, 16, 32):
            for PRF in (25, 100, 500, 1000):
                d = device.run(Acq(f_MHz * 1e6, 12, nRx, PRF, 0.03, mode))
                b = device.battery.size(d.P_avg, days=DAYS)
                if (d.fits_ble and d.fits_onchip and d.within_channels
                        and b["vol_cm3"] <= BUDGET_CM3):
                    survivors.append((mode, f_MHz, nRx, PRF,
                                      d.axial_res_mm, b["vol_cm3"]))

print(f"{len(survivors)} architectures fit a {BUDGET_CM3} cm3 wearable budget for {DAYS} day\\n")
print(f"{'mode':9s}{'f_Tx':>6s}{'nRx':>5s}{'PRF':>7s}{'res[mm]':>9s}{'vol[cm3]':>10s}")
for s in sorted(survivors, key=lambda s: (s[4], s[5]))[:15]:
    print(f"{s[0]:9s}{s[1]:>5d}M{s[2]:>5d}{s[3]:>7d}{s[4]:>9.2f}{s[5]:>10.2f}")
print("\\nWhich knob did you have to give up? (Hint: look at how few RF rows survive.)")
"""))

CELLS.append(md("""\
## Recap — and what is still unsolved

We went both ways — forward as a cost, backward as a design:
```
forward (the cost):    resolution → frequency → Nyquist → data → power → battery → wearable
inverse (the design):  wearable → battery → power → data → ... → architecture
```

**Three escape routes from the link wall**

| route | what it buys | what it costs |
|---|---|---|
| lower `f_Tx` | fewer samples | axial resolution |
| analog **BWR** / envelope | ~4× fewer samples, resolution kept | phase (no Doppler) — the ModulUS path |
| on-device **features** | radio collapses | MCU compute budget (edge-AI / PULP) |

**Still open (the next few years):** multi-channel *low-power* acquisition;
phase-preserving BWR (I/Q) for Doppler and displacement; ASIC integration
(→ Costa, Wed); transducer + edge-AI co-design. *This is where your research comes in.*
"""))

CELLS.append(md("""\
## References

1. C. Leitner, M. Giordano, M. Tanner, F. Villani, M. Magno and L. Benini,
   "ModulUS: A Sandbox for High-Resolution Wearable Ultrasound Development,"
   *2025 IEEE International Ultrasonics Symposium (IUS)*, Utrecht, Netherlands,
   2025, pp. 1-4, doi: [10.1109/IUS62464.2025.11201551](https://doi.org/10.1109/IUS62464.2025.11201551).
"""))

# ── Emit two notebooks from the same cells: student (blanks) + solutions ──
import copy

# map each TODO blank line -> its filled solution
SOLUTIONS = {
    "raw_mean = ...     # <-- replace ... using f_rf and P_rf":
        "raw_mean = float(np.sum(f_rf * P_rf) / np.sum(P_rf))",
    "env_mean = ...     # <-- replace ... using f_env and P_env":
        "env_mean = float(np.sum(f_env * P_env) / np.sum(P_env))",
    "data_rate = ...              # <-- replace ... using d.N, acq.bits, acq.PRF, acq.nRx":
        "data_rate = d.N * acq.bits * acq.PRF * acq.nRx",
    "    if ...:                      # <-- replace ... with the condition \"no longer fits BLE\"":
        "    if not d.fits_ble:",
}

def write_nb(cells, path):
    n = nbf.v4.new_notebook(); n.cells = cells
    nbf.write(n, str(path))
    print(f"wrote {path}  ({len(cells)} cells)")

# student version (blanks)
write_nb(CELLS, REPO_ROOT / NB_NAME)

# solutions version (filled; TODO -> SOLUTION)
sol_cells = copy.deepcopy(CELLS)
for c in sol_cells:
    if c.cell_type != "code":
        continue
    for blank, sol in SOLUTIONS.items():
        c.source = c.source.replace(blank, sol)
    c.source = c.source.replace("# TODO (", "# SOLUTION (")
write_nb(sol_cells, REPO_ROOT / NB_TEACHER)
