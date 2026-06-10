"""ModulUS digital twin — lean software mirror of the 4-module sandbox.

Hardware module          -> software twin
  Motherboard (base)     -> System    interconnect / composition root
  Pulse (pulser + T/R)   -> Pulse     send the pulse        (8 ch exposed)
  Echo  (AFE envelope)   -> Echo      catch the echo        (RF passthru | envelope)
  Core  (STM32 + 2x ADC) -> Core      digitize, sequence, compute

External to the sandbox (modeled, NOT ModulUS boards):
  Transducer (front)  |  Radio/BLE (downstream)  |  Battery (power source)

f_Tx is the ONE primary knob; fs and axial resolution are DERIVED (the spine):
  f_Tx -> resolution -> fs (Nyquist) -> N -> data_rate -> power -> battery

Research-tier: lean, modular, constants up top, type hints only at edges.
Provenance flags:  OK solid · ~ cite TBD · # datasheet/measure
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml


def load_config(path=None):
    """Load the ModulUS system definition from config.yaml (the twin's spec)."""
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _v(leaf):
    """Value of a config leaf: a {value, ref} mapping, or a bare scalar."""
    return leaf["value"] if isinstance(leaf, dict) and "value" in leaf else leaf


def config_references(cfg=None):
    """Flatten the config into (param, value, ref, status) rows for provenance."""
    cfg = cfg or CFG
    rows = []
    for board, params in cfg.items():
        if not isinstance(params, dict):
            continue
        for name, leaf in params.items():
            ref = leaf.get("ref", "") if isinstance(leaf, dict) else ""
            status = leaf.get("status", "") if isinstance(leaf, dict) else ""
            rows.append((f"{board}.{name}", _v(leaf), ref, status))
    return rows


# ── Constants: loaded from config.yaml (value + ref per entry) ───────────
CFG = load_config()
_phys, _pulse, _echo = CFG["physics"], CFG["pulse"], CFG["echo"]
_core, _radio, _batt = CFG["core"], CFG["radio"], CFG["battery"]

C_SOUND    = _v(_phys["speed_of_sound_ms"])   # m/s    soft-tissue speed of sound
N_CYCLES   = _v(_phys["pulse_cycles"])        # excitation pulse length
K_NYQUIST  = _v(_core["adc_nyquist_factor"])  # fs_RF = K_NYQUIST * f_Tx
BWR_FACTOR = _v(_echo["bandwidth_reduction"]) # envelope bandwidth reduction
N_CH_MAX   = _v(_pulse["channels_exposed"])   # channels exposed by ModulUS
N_FEATURES = _v(_core["feature_count"])       # features-mode payload / A-line

ADC_FOM    = _v(_core["adc_fom_j"])           # Walden ADC FoM [J/conv-step]
FS_ONCHIP  = _v(_core["adc_fs_max_hz"])       # on-chip ADC ceiling [samp/s]
E_BIT      = _v(_radio["energy_per_bit_j"])   # effective BLE energy [J/bit]
R_BLE_MAX  = _v(_radio["throughput_max_bps"]) # usable BLE throughput [bit/s]
RHO_VOL    = _v(_batt["density_vol_wh_l"])    # Li-ion volumetric density [Wh/L]
RHO_GRAV   = _v(_batt["density_grav_wh_kg"])  # Li-ion gravimetric density [Wh/kg]
CR2032_WH  = _v(_batt["reference_cr2032_wh"]) # reference coin cell [Wh]

P_PULSE    = _v(_pulse["power_w"])            # Tx pulser housekeeping [W]
P_ECHO     = _v(_echo["power_bias_w"])        # always-on analog AFE bias [W]
P_CORE     = _v(_core["power_idle_w"])        # MCU + housekeeping floor [W]
P_ECHO_DYN = _v(_echo["power_active_w"])      # AFE active, duty-scaled [W]
P_FEAT     = _v(_core["power_feature_w"])     # extra MCU draw in features mode [W]


# ── Acquisition context: state filled as the signal walks the stack ───────
@dataclass
class Acq:
    f_Tx: float; bits: int; nRx: int; PRF: float; D: float; mode: str
    fs: float = 0.0          # filled by Core.acquire
    N: float = 0.0           # filled by Core.acquire
    duty: float = 0.0        # filled by Pulse.configure
    data_rate: float = 0.0   # filled by Core.compute

    @property
    def t_acq(self):
        return 2 * self.D / C_SOUND      # round-trip window to depth D [s]


# ── External front: the transducer (sets resolution, the source) ──────────
@dataclass
class Transducer:
    n_cycles: int = N_CYCLES

    def axial_res(self, f_Tx):
        lam = C_SOUND / f_Tx             # wavelength [m]
        return self.n_cycles * lam / 2.0 # half the spatial pulse length [m]


# ── Pulse = pulser board (STHVUP32 + T/R switch): send the pulse ──────────
@dataclass
class Pulse:
    p_pulser: float = P_PULSE            # W  generic, low-duty Tx
    n_ch_max: int = N_CH_MAX

    def configure(self, acq):
        acq.duty = acq.t_acq * acq.PRF   # active fraction

    def power(self, acq):
        return self.p_pulser             # low-duty Tx; ~constant first order


# ── Echo = AFE board (envelope detector): catch & condition the echo ──────
@dataclass
class Echo:
    bwr_factor: int = BWR_FACTOR
    p_bias: float = P_ECHO               # W  always-on analog bias
    p_on: float = P_ECHO_DYN             # W  active, duty-scaled

    def bandwidth_factor(self, mode):
        return 1 if mode == "RF" else self.bwr_factor   # how much it eases the ADC

    def process(self, rf):              # real-signal twin: |Hilbert| envelope
        from scipy.signal import hilbert
        return np.abs(hilbert(rf))

    def power(self, acq):
        return self.p_bias + self.p_on * acq.duty * acq.nRx


# ── Core = control board (STM32 + dual ADC): digitize, sequence, compute ──
@dataclass
class Core:
    fs_max: float = FS_ONCHIP
    fom: float = ADC_FOM
    p_idle: float = P_CORE               # W  MCU + housekeeping (the floor)
    p_feat: float = P_FEAT               # W  on-device feature compute
    n_features: int = N_FEATURES

    def acquire(self, acq, echo):       # derive fs, N from transducer + Echo
        nyq = K_NYQUIST * acq.f_Tx
        acq.fs = nyq / echo.bandwidth_factor(acq.mode)
        acq.N  = acq.fs * acq.t_acq
        return acq.fs <= self.fs_max    # fits on-chip ADC? (else needs external/FPGA)

    def compute(self, acq):             # MCU decides the payload to ship
        if acq.mode == "features":
            acq.data_rate = self.n_features * acq.bits * acq.PRF * acq.nRx
        else:                           # RF / BWR differ only via fs -> N
            acq.data_rate = acq.N * acq.bits * acq.PRF * acq.nRx

    def power(self, acq):
        p_adc = self.fom * (2**acq.bits) * acq.fs * acq.duty * acq.nRx
        p_mcu = self.p_idle + (self.p_feat if acq.mode == "features" else 0.0)
        return p_adc + p_mcu


# ── External downstream: BLE radio (NOT a ModulUS board) ──────────────────
@dataclass
class Radio:
    e_bit: float = E_BIT
    r_max: float = R_BLE_MAX

    def fits(self, acq):
        return acq.data_rate <= self.r_max

    def power(self, acq):
        return acq.data_rate * self.e_bit


# ── External power source: battery sizing ─────────────────────────────────
@dataclass
class Battery:
    rho_vol: float = RHO_VOL
    rho_grav: float = RHO_GRAV

    def size(self, P_avg, days=1):
        Wh = P_avg * 86400 * days / 3600.0
        return dict(Wh=Wh, vol_cm3=Wh / self.rho_vol * 1000.0,
                    mass_g=Wh / self.rho_grav * 1000.0, n_cr2032=Wh / CR2032_WH)


def fom_mw_per_mhz(P_mW, nRx, f_Tx):    # paper FoM: avg power / Rx ch / f_Tx
    return P_mW / nRx / (f_Tx / 1e6)


# ── Motherboard = System: interconnect the twins, walk the spine ──────────
class System:
    """The Motherboard. Holds the twins; walk() runs the signal down the stack."""

    def __init__(self):
        self.transducer = Transducer()
        self.pulse, self.echo, self.core = Pulse(), Echo(), Core()
        self.radio, self.battery = Radio(), Battery()

    def walk(self, acq):                 # signal flows down the board stack
        self.pulse.configure(acq)        # -> duty
        fits_adc = self.core.acquire(acq, self.echo)   # -> fs, N
        self.core.compute(acq)           # -> data_rate
        P = {"Pulse": self.pulse.power(acq),
             "Echo":  self.echo.power(acq),
             "Core":  self.core.power(acq),
             "Radio": self.radio.power(acq)}
        P_avg = sum(P.values())
        return dict(acq=acq, P=P, P_avg=P_avg,
                    axial_res_mm=self.transducer.axial_res(acq.f_Tx) * 1e3,
                    fits_onchip=fits_adc,
                    fits_ble=self.radio.fits(acq),
                    within_hw=acq.nRx <= N_CH_MAX,         # beyond 8 ch = extrapolation
                    fom=fom_mw_per_mhz(P_avg * 1e3, acq.nRx, acq.f_Tx))


# ── Demo-data twin: real ModulUS traces, or loud synthetic fallback ───────
def synth_rf_envelope(fs=20e6, f_Tx=2e6, depths_mm=(20.0, 21.5), n_cycles=N_CYCLES):
    """Two-echo RF trace (disc upper/lower boundary) + its envelope.
    Generic ~2 MHz placeholder until the real ModulUS traces are provided;
    swapped out by load_traces() once the acquired .npz is present."""
    t = np.arange(0, 60e-6, 1 / fs)
    rf = np.zeros_like(t)
    for d_mm in depths_mm:
        t0 = 2 * (d_mm * 1e-3) / C_SOUND
        win = np.exp(-((t - t0) ** 2) / (2 * (n_cycles / f_Tx / 2) ** 2))
        rf += win * np.sin(2 * np.pi * f_Tx * (t - t0))
    from scipy.signal import hilbert
    return rf, np.abs(hilbert(rf)), fs


def load_traces(path="modulus_demo.npz"):
    """Real ModulUS traces if present; loud synthetic fallback otherwise.
    NEVER crash in front of the class."""
    from pathlib import Path
    if Path(path).exists():
        d = np.load(path)
        return d["rf"], d["env"], float(d["fs"])
    print("=" * 60)
    print("  SYNTHETIC DATA  - real ModulUS file not found:")
    print(f"    {path}")
    print("  (load the acquired .npz to use measured RF + envelope)")
    print("=" * 60)
    return synth_rf_envelope()


# ── Sanity check (proves twin reproduces the flat-model numbers) ──────────
if __name__ == "__main__":
    sys = System()

    def row(tag, f_Tx, bits, nRx, PRF, D, mode):
        r = sys.walk(Acq(f_Tx, bits, nRx, PRF, D, mode))
        a = r["acq"]
        print(f"{tag:12s} fs={a.fs/1e6:5.1f}M  N={a.N:6.0f}  "
              f"dr={a.data_rate/1e6:7.3f}Mb/s  P={r['P_avg']*1e3:7.2f}mW  "
              f"FoM={r['fom']:5.2f}  BLE={'OK ' if r['fits_ble'] else 'OVER'}  "
              f"ADC={'on ' if r['fits_onchip'] else 'EXT'}  "
              f"res={r['axial_res_mm']:.3f}mm")

    print("=== Operating point: 10 MHz, nRx=1, PRF=25 Hz, D=3 cm ===")
    for m in ("RF", "BWR", "features"):
        row(m, 10e6, 12, 1, 25, 0.03, m)

    print("\n=== Wall demo: 10 MHz, nRx=8, PRF=100 Hz, D=3 cm ===")
    for m in ("RF", "BWR", "features"):
        row(m, 10e6, 12, 8, 100, 0.03, m)

    print("\n=== Battery (BWR op-point, 1 day) ===")
    r = sys.walk(Acq(10e6, 12, 1, 25, 0.03, "BWR"))
    b = sys.battery.size(r["P_avg"])
    print(f"  P_avg={r['P_avg']*1e3:.2f} mW -> {b['vol_cm3']:.3f} cm3, "
          f"{b['mass_g']:.2f} g, {b['n_cr2032']:.3f} CR2032")
