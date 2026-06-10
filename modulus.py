"""ModulUS digital twin — lean software mirror of the 4-module sandbox.

              ┌──────────────┐
              │  Transducer  │  external front · f_Tx is the runtime knob
              └──────┬───────┘
                RF / echo                                       class
   ┌─────────────────┼──────────── ModulUS sandbox ──────────┐
   │          ┌──────┴───────┐                                │
   │          │    Pulse     │  pulser STHVUP32 + T/R         │   slot 'pulse'
   │          └──────┬───────┘                                │
   │          ┌──────┴───────┐                                │
   │          │    Echo      │  analog front-end (AFE)        │   slot 'echo'
   │          └──────┬───────┘                                │
   │          ┌──────┴───────┐                                │
   │          │    Core      │  STM32L496 + dual 5 Msps ADC   │   slot 'core'
   │          └──────┬───────┘                                │
   │           Motherboard   (interconnects the 3 boards)     │   System
   └─────────────────┼───────────────────────────────────────┘
                 data │
        ┌─────────────┴┐                     ┌──────────────┐
        │Wireless link │  Radio (e.g. BLE)   │   Battery    │   slots
        └──────────────┘                     └──────────────┘   'radio' / 'battery'

Each slot is filled by a swappable board, chosen in config.yaml (`type:`) and
built from a registry. Boards obey a Protocol contract (structural typing) and
exchange a standardized `Signal`. Add a board = one class + one @register line;
select it = one word in config.yaml. Transducer, Radio, Battery are modeled but
are NOT ModulUS boards.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
import numpy as np
import yaml


# ── Config loading (value / ref / status per entry) ──────────────────────
def load_config(path=None):
    """Load the ModulUS system definition from config.yaml (the twin's spec)."""
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _v(leaf):
    """Value of a config leaf: a {value, ref, status} mapping, or a bare scalar."""
    return leaf["value"] if isinstance(leaf, dict) and "value" in leaf else leaf


def config_references(cfg=None):
    """Flatten the config into (param, value, ref, status) rows for provenance."""
    cfg = cfg or CFG
    rows = []
    for section, body in cfg.items():
        if not isinstance(body, dict):
            continue
        leaves = body.get("params", body)        # board slots nest under 'params'
        for name, leaf in leaves.items():
            if isinstance(leaf, dict) and "value" in leaf:
                rows.append((f"{section}.{name}", _v(leaf),
                             leaf.get("ref", ""), leaf.get("status", "")))
    return rows


CFG = load_config()
C_SOUND  = _v(CFG["physics"]["speed_of_sound_ms"])   # m/s soft-tissue speed of sound
N_CYCLES = _v(CFG["physics"]["pulse_cycles"])        # excitation pulse length


# ── Standardized IO: a tagged signal passed between boards (ports) ────────
@dataclass
class Signal:
    data: np.ndarray
    fs: float
    domain: str          # 'RF' | 'envelope' | 'IQ' | 'features'


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


# ── Board contracts (Protocol = structural typing, no inheritance) ────────
# Standardized IO/parameters: same-slot boards are interchangeable iff they
# honor the same Protocol. Behavior lives in the concrete board classes.
@runtime_checkable
class Board(Protocol):
    def power(self, acq) -> float: ...                 # every board reports power [W]

class PulseBoard(Protocol):
    channels_exposed: int
    def configure(self, acq) -> None: ...              # sets duty
    def power(self, acq) -> float: ...

class AFEBoard(Protocol):
    in_domain: str; out_domain: str
    def bandwidth_factor(self, mode) -> float: ...     # how much it eases the ADC
    def process(self, sig: Signal) -> Signal: ...      # real-signal transform
    def power(self, acq) -> float: ...

class CoreBoard(Protocol):
    def acquire(self, acq, afe) -> bool: ...           # derive fs, N; fits on-chip?
    def compute(self, acq) -> None: ...                # set data_rate
    def power(self, acq) -> float: ...


# ── Registry: boards self-register per slot; build() picks from config ────
BOARDS = {}
_CONTRACT = {                                          # required methods per slot
    "pulse":   ("configure", "power"),
    "echo":    ("bandwidth_factor", "process", "power"),
    "core":    ("acquire", "compute", "power"),
    "radio":   ("fits", "power"),
    "battery": ("size",),
}

def register(slot, name):
    """Register a board class under a slot; validate the contract at import."""
    def deco(cls):
        for m in _CONTRACT[slot]:
            assert hasattr(cls, m), f"{cls.__name__} (slot '{slot}') missing {m}()"
        BOARDS[(slot, name)] = cls
        return cls
    return deco

def build(slot, name, params):
    """Construct board `name` for `slot` with `params` (value-only)."""
    if (slot, name) not in BOARDS:
        avail = [n for (s, n) in BOARDS if s == slot]
        raise ValueError(f"no board '{name}' for slot '{slot}'; available: {avail}")
    return BOARDS[(slot, name)](**{k: _v(v) for k, v in params.items()})


# ── External front: the transducer (sets resolution, the source) ──────────
@dataclass
class Transducer:
    n_cycles: int = N_CYCLES

    def axial_res(self, f_Tx):
        lam = C_SOUND / f_Tx             # wavelength [m]
        return self.n_cycles * lam / 2.0 # half the spatial pulse length [m]


# ── Pulse boards (slot 'pulse') — send the pulse ──────────────────────────
@register("pulse", "standard_pulser")
@dataclass
class StandardPulser:
    """STHVUP32 pulser + T/R switch."""
    channels_exposed: int = 8
    power_w: float = 1e-3

    def configure(self, acq):
        acq.duty = acq.t_acq * acq.PRF   # active fraction

    def power(self, acq):
        return self.power_w              # low-duty Tx; ~constant first order


# ── AFE boards (slot 'echo') — condition the echo ─────────────────────────
@register("echo", "envelope")
@dataclass
class EnvelopeAFE:
    """Analog front-end as an envelope detector (RF passthru | envelope)."""
    bandwidth_reduction: int = 4
    power_bias_w: float = 4e-3
    power_active_w: float = 3e-3
    in_domain: str = "RF"
    out_domain: str = "envelope"

    def bandwidth_factor(self, mode):
        return 1 if mode == "RF" else self.bandwidth_reduction

    def process(self, sig):              # real-signal twin: |Hilbert| envelope
        from scipy.signal import hilbert
        return Signal(np.abs(hilbert(sig.data)), sig.fs, "envelope")

    def power(self, acq):
        return self.power_bias_w + self.power_active_w * acq.duty * acq.nRx


# ── Core boards (slot 'core') — digitize, sequence, compute ───────────────
@register("core", "stm32_dual_adc")
@dataclass
class STM32DualADC:
    """STM32L496 + dual 5 Msps 12-bit on-chip ADC."""
    adc_nyquist_factor: int = 2
    adc_fs_max_hz: float = 5e6
    adc_fom_j: float = 50e-15
    power_idle_w: float = 15e-3
    power_feature_w: float = 2e-3
    feature_count: int = 16

    def acquire(self, acq, afe):         # derive fs, N from transducer + AFE
        nyq = self.adc_nyquist_factor * acq.f_Tx
        acq.fs = nyq / afe.bandwidth_factor(acq.mode)
        acq.N = acq.fs * acq.t_acq
        return acq.fs <= self.adc_fs_max_hz          # fits on-chip ADC?

    def compute(self, acq):              # MCU decides the payload to ship
        if acq.mode == "features":
            acq.data_rate = self.feature_count * acq.bits * acq.PRF * acq.nRx
        else:                            # RF / BWR differ only via fs -> N
            acq.data_rate = acq.N * acq.bits * acq.PRF * acq.nRx

    def power(self, acq):
        p_adc = self.adc_fom_j * (2 ** acq.bits) * acq.fs * acq.duty * acq.nRx
        p_mcu = self.power_idle_w + (self.power_feature_w if acq.mode == "features" else 0.0)
        return p_adc + p_mcu


# ── Wireless link (slot 'radio') — downstream, NOT a ModulUS board ────────
@register("radio", "ble")
@dataclass
class BLELink:
    energy_per_bit_j: float = 20e-9
    throughput_max_bps: float = 1e6

    def fits(self, acq):
        return acq.data_rate <= self.throughput_max_bps

    def power(self, acq):
        return acq.data_rate * self.energy_per_bit_j


# ── Battery (slot 'battery') — external power source ──────────────────────
@register("battery", "liion")
@dataclass
class LiIonBattery:
    density_vol_wh_l: float = 400.0
    density_grav_wh_kg: float = 230.0
    reference_cr2032_wh: float = 0.65

    def size(self, P_avg, days=1):
        Wh = P_avg * 86400 * days / 3600.0
        return dict(Wh=Wh, vol_cm3=Wh / self.density_vol_wh_l * 1000.0,
                    mass_g=Wh / self.density_grav_wh_kg * 1000.0,
                    n_cr2032=Wh / self.reference_cr2032_wh)


def fom_mw_per_mhz(P_mW, nRx, f_Tx):     # paper FoM: avg power / Rx ch / f_Tx
    return P_mW / nRx / (f_Tx / 1e6)


# ── Motherboard = System: build the boards from config, walk the spine ────
class System:
    """The Motherboard. Builds each slot's board from config; walk() runs the
    signal down the stack."""

    def __init__(self, config=None):
        cfg = config or CFG
        b = cfg["boards"]                # the ModulUS definition: slot -> board name
        self.transducer = Transducer()
        self.pulse   = build("pulse",   b["pulse"],   cfg["pulse"]["params"])
        self.echo    = build("echo",    b["echo"],    cfg["echo"]["params"])
        self.core    = build("core",    b["core"],    cfg["core"]["params"])
        self.radio   = build("radio",   b["radio"],   cfg["radio"]["params"])
        self.battery = build("battery", b["battery"], cfg["battery"]["params"])

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
                    within_hw=acq.nRx <= self.pulse.channels_exposed,
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
    if Path(path).exists():
        d = np.load(path)
        return d["rf"], d["env"], float(d["fs"])
    print("=" * 60)
    print("  SYNTHETIC DATA  - real ModulUS file not found:")
    print(f"    {path}")
    print("  (load the acquired .npz to use measured RF + envelope)")
    print("=" * 60)
    return synth_rf_envelope()


# ── Sanity check (proves the registry/board refactor preserves the numbers)
if __name__ == "__main__":
    sys = System()
    print("registered boards:", sorted(BOARDS))

    def row(tag, f_Tx, bits, nRx, PRF, D, mode):
        r = sys.walk(Acq(f_Tx, bits, nRx, PRF, D, mode))
        a = r["acq"]
        print(f"{tag:12s} fs={a.fs/1e6:5.1f}M  N={a.N:6.0f}  "
              f"dr={a.data_rate/1e6:7.3f}Mb/s  P={r['P_avg']*1e3:7.2f}mW  "
              f"FoM={r['fom']:5.2f}  BLE={'OK ' if r['fits_ble'] else 'OVER'}  "
              f"ADC={'on ' if r['fits_onchip'] else 'EXT'}  "
              f"res={r['axial_res_mm']:.3f}mm")

    print("\n=== Operating point: 10 MHz, nRx=1, PRF=25 Hz, D=3 cm ===")
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
