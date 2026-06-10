# CEEUS 2026 Workshop — Session Log (2026-06-06)

Design + build session for the ModulUS power/battery Colab. Captures decisions,
artifacts, and the resume point. (Reconstructed transcript, not verbatim.)

---

## Context established
- Read `dev_material/ceeus2026_workshop_brief.md` (full) and the ModulUS IUS-2025
  paper (PDF was password-protected → extracted text to
  `dev_material/modulus_ius2025_extracted.txt`).
- Claude wears 4 expert hats for this work: wearable-US domain, pedagogy/syllabus,
  research-code, software-engineer-for-teaching.
- Workshop: IEEE CEEUS 2026, Warsaw, 22–24 Jun 2026. ~20 min talk + live ModulUS
  acquisition + ~30 min Colab. Owns the system–power–data axis.

## Working style agreed
- Go slow, step by step: **zoom-out concept → review together → zoom-in → review**.
  Not big-bang builds. User is architect; surface trade-offs, let user decide.
- Use tables + ASCII to structure ideas.
- Cite or discuss provenance for every "acknowledged value".

## Narrative locked — 3-act arc (inductive, grounded in the real signal)
```
ACT 1 SIGNAL            ACT 2 COST              ACT 3 SYSTEM
"what's in here?"   →  "what's it cost      →  "can this be
                        to capture & move?"      a wearable?"
resolution/freq         Nyquist→data→walls       power→battery→INVERT
SEE it                  PAY for it               LIVE with it
```
Spine: `resolution → frequency → Nyquist → data rate → power → battery → wearability`.
- Plant the envelope in Act 1, cash it in Act 3.
- Act 3 ends by INVERTING the chain (given wrist budget → what survives) = the
  gradeable terminal exercise.
- Designed "oh": single-channel raw RF (realistic PRF/nRx) breaks the ~1 Mb/s BLE
  wall → "this architecture cannot exist", not "use a bigger battery".

## Model decisions locked
- **Option B**: `f_Tx` is the ONE primary knob; `fs` and resolution are DERIVED
  (one connected story). Rejected fs-as-primary (breaks the spine).
- `K_NYQUIST=2` (fs_RF = 2·f_Tx → 10 MHz = 20 Msps). `BWR_FACTOR=4`
  (fs_BWR = 5 Msps, lands on STM32L496 on-chip ADC). 20→5 = 4× ties to paper.
- **Two walls, three tiers**: ADC wall (5 Msps) splits RF (external ADC/FPGA) from
  BWR/features (MCU class) = Table I's FPGA-vs-MCU divide. BLE wall (~1 Mb/s) splits
  features (survives) from BWR (floods radio at nRx≈8). RF breaks both.
- **3 primary sliders**: f_Tx (axial res), PRF (temporal res), nRx (lateral). bits, D secondary.
- **In-sandbox power = generic ~20 mW static** (NOT the PuLsE paper number):
  Ping 1 + Echo 4 + Core 15 mW; +2 mW features. Lesson: device idles at floor,
  radio explodes toward real imaging.
- **mW/MHz**: notebook outputs total power (drives battery) AND the paper's FoM
  (power/Rx-ch/f_Tx) as a secondary line.
- Rename `R_dot → data_rate`. Synthetic demo RF set to ~2 MHz placeholder.

## Hardware digital-twin = software architecture
ModulUS = 4 modules. Twin classes in `modulus.py`:
- **Motherboard** ≡ `System` (interconnect / composition root).
- **Ping** = pulser board (STHVUP32 + T/R switch) — send the pulse. 8 ch exposed.
- **Echo** = AFE board (envelope detector) — catch/condition the echo.
- **Core** = control board (STM32L496 + dual 5 Msps ADC) — acquire + compute.
- External (NOT ModulUS): `Transducer` (front), `Radio`/BLE (downstream), `Battery`.
Board naming = Scheme A (the pulse-echo physics): **Ping / Echo / Core** on the
**Motherboard**; daughterboards branded **Modules**. Use this vocabulary across
code + notebook + deck.
`Acq` context object filled as signal walks the stack (per-call, fresh). Stateless
`System`. `nRx>8` flagged as beyond-hardware extrapolation.

## Colab engineering (SoA-checked)
- `%%writefile`-style single source NOT used; instead **clone our repo** in Colab
  (sandbox, students see modulus.py) + **pip install git+ dasIT** (external lib).
- Bootstrap: `IN_COLAB` guard (runs locally too), idempotent clone, data path
  resolved for both envs. ipywidgets `interact` **sliders only** (well supported);
  Colab `@param` forms as documented fallback. Add Open-in-Colab badge + nbstripout.
- Fail-soft `load_traces` (synthetic fallback, never crash live). Self-check asserts
  after each TODO. Deps: numpy/scipy/matplotlib/ipywidgets (Colab-stock).

## dasIT reuse (https://github.com/luuleitner/dasIT — user's own toolbox)
- dasIT = SIGNAL domain (Act 1); `modulus.py` = SYSTEM spine (Acts 2-3).
- Reuse: `features.signal.fftsignal` (Welch PSD, MHz) ✓ used; `analytic_signal`/
  `envelope`; `RFfilter` (EE-tier). Skipped: RFDataloader, transducer/medium classes.
  (Dropped `amp_freq_1channel` — its "raw vs filtered" labels mislabel the envelope;
  used a custom plot + fftsignal instead.)

## Artifacts created
- `modulus.py` (repo ROOT) — the digital twin. Validated; `__main__` sanity check
  reproduces: op-point 10 MHz → BWR fits on-chip, RF needs external ADC; wall demo
  RF & BWR break BLE, features survive; battery sizing. Runs with `py -3.12 modulus.py`.
- `ceeus2026_modulus_power.ipynb` (repo ROOT) — 8 cells: front matter + bootstrap
  (C0-C1) + Act 1 (C2-C7). **Executes headless end-to-end, exit 0.**
- `dev_material/build_notebook.py` — nbformat builder (notebook-as-code, the source
  for the .ipynb; extend CELLS to add cells). Run: `py -3.12 dev_material/build_notebook.py`.
- `requirements.txt`, `.gitignore` (strong) at root.
- `dev_material/modulus_ius2025_extracted.txt` — extracted paper text.

## Sanity numbers (current, generic 20 mW budget)
```
OP-POINT 10 MHz, nRx=1, PRF=25 Hz:  RF 24.7mW FoM2.47 EXT | BWR 21.2mW 2.12 on | feat 22.1mW 2.21 on
WALL 10 MHz, nRx=8, PRF=100 Hz:     RF 169.8mW OVER EXT | BWR 57.5mW OVER on | feat 25.2mW OK on
Battery (BWR op, 1 day): 1.27 cm³, 2.2 g, 0.78 CR2032
```

---

## RESUME POINT — next actions (in order)
1. **Build Act 2 (C8–C11)** in `build_notebook.py`: Nyquist → data_rate, the two
   walls; TODO(b) = fill the `data_rate` line + self-check; Stage-1 `interact`
   (mode LOCKED=RF) → "can't beat BLE by shrinking the battery". Rebuild + execute
   headless to verify. Review with user.
2. **Build Act 3 (C12–C16)**: power→battery widget (Stage-2, mode unlocked = the
   "oh"); TODO(c) reflection; INVERT exercise (wrist budget → what survives); recap.
3. **Backup cell (C17)**: EE-tier RFfilter AFE model, dual-ADC multiplexing note, refs.
4. Lock open items: GitHub **slug** (placeholder `luuleitner/ceeus2026-wearables`);
   create + commit `example_data/modulus_demo.npz` and un-ignore `example_data/`;
   wire user's REAL ModulUS RF+envelope traces into `load_traces`.
5. Lock BLE literature citations (`E_BIT`, `R_BLE_MAX`) via WebSearch (Gomez 2012 /
   Tosi 2017 throughput; Siekkinen 2012 energy/bit). Also Walden FoM, Li energy density.
6. **Later deliverables**: 4 diagram slides (1,4,11,17); deck text (Ping/Echo/Core
   vocab); placative images with citations (hair/seed/cell, battery sizes).

## How to resume
- Memory index: `MEMORY.md` in the project memory dir holds pointers to all decisions.
- Rebuild notebook: `py -3.12 dev_material/build_notebook.py`.
- Verify: `py -3.12 modulus.py` (model) and headless-execute the .ipynb.
- Python: use `py -3.12` (3.6 too old). dasIT installed locally for testing.
