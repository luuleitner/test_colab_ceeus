# CEEUS 2026 Wearables Workshop — Project Brief

> Handoff document for continuing in Claude Code CLI.
> Workshop by Christoph Leitner (ETH Zurich, IIS). IEEE CEEUS 2026, Warsaw, 22–24 Jun 2026.
> Format: ~20 min talk + live ModulUS acquisition + ~30 min Google Colab (battery/power estimation).

---

## 0. What this doc is

A consolidated brief from the design session. Three deliverables remain to build:
1. **Slide deck** (~14 spoken slides + backup) — content specified in §4.
2. **Google Colab notebook** (`.ipynb`) — the power/battery model — spec + skeleton in §5. **PRIMARY remaining work.**
3. **Four diagram slides** (1, 4, 11, 17) — listed in §6.

---

## 1. Conference context (grounding)

- IEEE CEEUS 2026, 22–24 Jun 2026, Warsaw. ~3 weeks out.
- Sheng Xu plenary (Mon): "Wearable ultrasound technology" → covers transducer/patch side. **Do not re-teach transducers.**
- Sergei Vostrikov (Tue): "Open-source wearable ultrasound platforms" → WULPUS & TinyProbe (two rows of Table I). **Reference, don't re-introduce.**
- Tiago Costa (Wed): miniaturized power-efficient focused-US microsystems → the ASIC/integration endgame (nod to in slide 16).
- Christoph co-chairs the Tue plenary.
- **Positioning:** this workshop owns the *system–power–data* axis, which is exactly the ModulUS thesis.

---

## 2. The structural fix (highest priority)

Terminal objective ("sketch a wearable US architecture") and the Colab ("estimate battery size") were disconnected. Unify them with one spine:

> **resolution → frequency → Nyquist/data rate → power → battery → wearability**

- Talk **teaches** the chain.
- Colab makes them **compute** the chain.
- Closing exercise **inverts** it: given a wrist battery budget, sketch the architecture (fs, bits, nRx, data mode) that fits.

This turns the battery calculator into an architecture-design tool and makes the terminal objective assessable.

---

## 3. Critical evaluation summary

**Learning objectives** — replace "understand" with assessable verbs (identify, classify, estimate, compare, justify). Give the "what's unsolved in the next few years" objective a real slide (16).

**Narrative arc** — talk must build *tension* toward the Colab punchline: data representation (RF vs BWR vs features) is the dominant power knob (= mW/MHz thesis). Slides 6–8 should escalate (ADC deluge → BLE wall → battery).

**Scope/time** — cut to ~14 content slides + backup (18 in 20 min is too dense). Colab is *guided exploration*, not build-from-scratch: model pre-written, students move sliders + fill 2–3 TODO cells.

**Colab discovery** — engineer a genuine "oh": a hard BLE throughput ceiling (~1 Mbit/s usable) that single-channel RF already breaks. The lesson is "this architecture cannot exist," not "use a bigger battery."

---

## 4. Restructured syllabus (5-act arc)

| # | Topic | Key message | Time (min) |
|---|---|---|---|
| 1 | What is wearable US | Co-design triangle Algorithm–System–Transducer; one failure-consequence per vertex | 1.5 |
| 2 | Why it's hard | Promise vs wearability gap (size, power, untethered) | 1.5 |
| 3 | Systems & FoM | MCU vs FPGA templates + Table I; mW/MHz metric; note Vostrikov Tue | 2 |
| 4 | Pulse–echo & the chain | A-mode; introduce resolution→frequency→data spine | 2 |
| 5 | Tx path | Pulser, HV, PRF; Tx is low-duty, NOT the power problem | 1.5 |
| 6 | Rx path | T/R switch, AFE, high-impedance piezo interface | 1.5 |
| 7 | Constraint 1: ADC & Nyquist | Full-BW = data deluge; samples/A-line math | 1.5 |
| 8 | Constraint 2: wireless wall | BLE usable ceiling; raw RF doesn't fit | 1.5 |
| 9 | Constraint 3: power profile | Radio dominates streaming power | 1.5 |
| 10 | Battery reality | Wh → cm³/g; CR2032 as unit; ModulUS 1200 cm³ as anti-example | 1.5 |
| 11 | The lever | RF vs BWR-envelope vs features; mW/MHz (contribution) | 2 |
| 12 | ModulUS | Sandbox concept; analog BWR; R²=0.88, ~4× BW, 0.6 mW/MHz | 2 |
| 13 | Live acquisition | Demo: raw RF + analog envelope on 10 MHz transducer | demo |
| 14 | Hand-off to Colab | "You have the data. Compute the battery. Then design the architecture." | 0.5 |
| 15 | Colab reference | Static model + three escape routes (leave on screen) | — |
| 16 | What's unsolved | Multi-channel low power; phase-preserving BWR; ASIC (→Costa); transducer + edge-AI | 2 |
| 17 | Synthesis | Invert the chain: given wrist budget, what survives? | 1 |
| 18 | Backup | AFE schematic, beamforming/Doppler, refs | — |

---

## 5. Colab spec — PRIMARY REMAINING WORK

First-order, defensible (not SPICE-accurate). One model function, four power blocks. Consistent with the ModulUS paper (PuLsE ~0.6 mW/MHz; raw 2.9 MHz → BWR 0.7 MHz ≈ 4×; 5 Msps on-chip ADC).

### Constants
- `c = 1540`  # m/s
- BLE usable ceiling `R_ble_max ≈ 1e6`  # bit/s (flag red above)
- `E_bit ≈ 10e-9 … 50e-9`  # J/bit effective BLE (default 20e-9)
- ADC Walden `FOM ≈ 50e-15`  # J/conv-step
- `rho_vol ≈ 250…500`  # Wh/L ; `rho_grav ≈ 200…250`  # Wh/kg
- CR2032 reference: ~0.65 Wh, ~3.3 cm³
- Wrist budget: ~3–5 cm³, ~15 g

### Model (pseudocode skeleton)
```python
def system_power(fs, bits, nRx, PRF, D, mode,
                 c=1540.0, FOM=50e-15, E_bit=20e-9,
                 P_afe_on=3e-3, P_afe_leak=50e-6,
                 P_pulser=1e-3, P_idle=1e-3, P_feat=2e-3,
                 n_features=16):
    ENOB = bits  # first-order: assume ENOB ~ resolution
    t_acq = 2 * D / c                       # round-trip window [s]
    N = fs * t_acq                          # samples per A-line
    duty = t_acq * PRF                      # active fraction

    # data rate depends on mode
    if mode == "features":
        R_dot = n_features * bits * PRF * nRx
    else:  # "RF" or "BWR" differ only via fs the caller passes in
        R_dot = N * bits * PRF * nRx

    P_adc   = FOM * (2**ENOB) * fs * duty * nRx
    P_afe   = P_afe_on * duty * nRx + P_afe_leak
    P_radio = R_dot * E_bit
    P_mcu   = P_idle + (P_feat if mode == "features" else 0.0)
    P_avg   = P_pulser + P_adc + P_afe + P_radio + P_mcu

    fits_ble = R_dot <= 1e6
    return dict(N=N, R_dot=R_dot, P_adc=P_adc, P_afe=P_afe,
                P_radio=P_radio, P_mcu=P_mcu, P_avg=P_avg,
                fits_ble=fits_ble)

def battery(P_avg, days=1, rho_vol=400.0, rho_grav=230.0):
    Wh = P_avg * 86400 * days / 3600.0
    return dict(Wh=Wh, vol_cm3=Wh / rho_vol * 1000.0,
                mass_g=Wh / rho_grav * 1000.0)
```

### Sliders (ipywidgets)
`fs`, `bits ∈ {8,10,12,14}`, `nRx ∈ {1,8,16,32}`, `PRF ∈ {25…1000} Hz`,
`D ∈ {1…6} cm`, **`mode ∈ {RF, BWR-envelope, features}`** (the star variable).

Defaults: RF → fs=40 Msps; BWR → fs=5 Msps; features → fs=5 Msps but R_dot collapses ~100–1000×.

### Resolution feedback (for escape route 1)
```python
def axial_res(f_hz, n_cycles=5, c=1540.0):
    lam = c / f_hz
    return n_cycles * lam / 2.0   # [m]
```

### The designed discovery
1. Start everyone at `RF, nRx=8, PRF=100`. Ask them to get under the BLE line by only shrinking the battery target → impossible.
2. Unlock `mode`. The "oh": architecture (not battery) was the problem. → feeds slide 17.

### Three escape routes (the takeaway)
1. **Lower f_Tx** — fewer samples, costs axial resolution (show `axial_res`).
2. **BWR / analog envelope** — ~4–8× fewer samples, resolution kept, phase lost (ModulUS path).
3. **On-device features** — radio term collapses, needs MCU compute budget (edge-AI / PULP path; the real SoA answer).

### TODO cells (keep 30 min feasible)
- (a) Load demo RF + BWR traces; confirm ~4× spectral ratio (ties to Fig. 3).
- (b) Fill the `R_dot` line.
- (c) Reflection: at what nRx does BWR also hit the wall? what's your escape?

### Demo data loader
Save both raw RF and analog-envelope traces from the live acquisition to a file the notebook loads (npz/csv). Spectral check should reproduce raw mean ≈ 2.9 MHz, BWR mean ≈ 0.7 MHz.

---

## 6. Diagram slides to produce
- **Slide 1** — co-design triangle (Algorithm–System–Transducer).
- **Slide 4** — the spine box: resolution → frequency → Nyquist → data → power → battery.
- **Slide 11** — three-column lever (RF / BWR / features) with mW/MHz each.
- **Slide 17** — the inverted chain (read right-to-left from wrist budget).

---

## 7. Open questions to resolve
1. **Audience floor** — EE/embedded (comfortable Python) vs mixed bio/clinical? Sets scaffolding depth in TODO cells.
2. **Lock the power-model numbers** before quoting on slides 7–12, so deck and Colab agree.

---

## 8. Suggested Claude Code next steps
1. `build the Colab notebook` from §5 (model + widgets + staged discovery + demo-data loader). Verify defaults reproduce paper figures (0.6 mW/MHz ballpark, 4× BW ratio).
2. Generate the four diagram slides in §6 (SVG or your deck tool).
3. Draft TODO-cell scaffolding at the chosen audience level.
4. Wire the demo acquisition export format to the notebook loader.

## Key paper anchors (ModulUS, IUS 2025)
- 4-board sandbox; STHVUP32 pulser, AFE BWR board, STM32L496 (dual 5 Msps 12-bit ADC), NRF53 BLE.
- AFE reduces effective bandwidth ~4× (raw mean 2.9 MHz → 0.7 MHz), envelope fidelity R²=0.88 vs Hilbert.
- mW/MHz figure of merit; PuLsE 0.6 mW/MHz (best); USoP 102.3, TinyProbe 2.0, WULPUS 5.5.
- Limits: single-channel at a time; envelope path discards phase (no Doppler/displacement); ModulUS is a sandbox (260×115×40 mm), not a wearable.
