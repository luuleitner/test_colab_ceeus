"""ModulUS digital twin — lean software model of the modular US sandbox.

SOFTWARE STRUCTURE  (how this file is wired; the HARDWARE diagram is in config.yaml)

   config.yaml ──load_config()──► CFG          (board parameters + refs)
                                    │ _params(section)
                                    ▼
   Acq(f_Tx, bits, nRx, PRF, D, mode, V_pp, n_cycles,        ──► System  (builds
       duty_cycled, rx_window_s)   ── the run-time knobs ──        the boards)
                                           │
                                           ▼  System.walk(acq) — Acq passes through
                                                the boards, each filling its part:
                                                Pulse.configure → duty
                                                Core.acquire    → fs, N
                                                Core.compute    → data_rate
                                                Σ power ; test BLE & ADC walls
                                                     │
                                                     ▼
                                             result dict:
                                               P, P_avg, fits_ble, fits_onchip,
                                               within_hw, fom, axial_res_mm

SCOPE — this is a first-order POWER / DATA-RATE / FEASIBILITY model. For a given
configuration it estimates: sampling (fs, N), data rate, average power per board,
battery size, axial resolution, and whether the design fits the on-chip ADC and
the wireless link. It does NOT simulate the ultrasound signal, image, beamforming,
SNR, or circuit-level electronics. (synth_rf_envelope / load_traces below are demo
traces for the notebook, not part of this model.)

Each board (Pulser, EnvelopeAFE, MixedSignalSoC, BLELink, LiIonBattery) is a
plain class with a few short methods — read any one to see the pattern. System
names each board's class; swap one by changing its class there. Transducer, Radio,
Battery are modeled but are NOT ModulUS boards.
"""
from pathlib import Path
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


CFG = load_config()
C_SOUND  = _v(CFG["medium"]["speed_of_sound_ms"])    # m/s soft-tissue speed of sound
N_CYCLES = _v(CFG["excitation"]["pulse_cycles"])     # default excitation pulse length


# ── Acquisition context (Acq): fields filled as it passes through the boards
class Acq:
    def __init__(self, f_Tx, bits, nRx, PRF, D, mode, V_pp=15.0, n_cycles=N_CYCLES,
                 duty_cycled=True, rx_window_s=None):
        self.f_Tx = f_Tx            # transducer centre frequency [Hz]
        self.bits = bits            # ADC resolution [bits]
        self.nRx = nRx              # parallel Rx channels
        self.PRF = PRF              # pulse repetition frequency [Hz]
        self.D = D                  # imaging depth [m]
        self.mode = mode            # 'RF' | 'BWR' | 'features'
        self.V_pp = V_pp            # excitation voltage [Vpp]; default 15 V (ModulUS)
        self.n_cycles = n_cycles    # excitation pulse length [cycles]; default 5
        self.duty_cycled = duty_cycled  # disable duty-cycleable blocks between acquisitions?
        self.rx_window_s = rx_window_s  # RX-on time per pulse [s]; None -> echo window 2D/c
        self.fs = 0.0               # filled by Core.acquire
        self.N = 0.0                # filled by Core.acquire
        self.duty = 0.0             # filled by Pulse.configure
        self.data_rate = 0.0        # filled by Core.compute

    @property
    def t_acq(self):
        return 2 * self.D / C_SOUND      # echo window: round-trip to depth D [s]

    @property
    def rx_window(self):
        # how long RX is enabled per pulse: explicit setting, else the echo window
        return self.rx_window_s if self.rx_window_s is not None else self.t_acq


# ── External front: the transducer (sets resolution; drives the load) ─────
class Transducer:
    """The piezo transducer. f_Tx sets the resolution; the element capacitance
    sets the transmit (load) energy the pulser must deliver."""
    def __init__(self, capacitance_f=1e-9):
        self.capacitance_f = capacitance_f

    def axial_res(self, f_Tx, n_cycles=N_CYCLES):
        lam = C_SOUND / f_Tx             # wavelength [m]
        return n_cycles * lam / 2.0      # half the spatial pulse length [m]

    def transmit_power(self, acq):
        # charge/discharge the element each cycle, summed over active channels
        #   ~ C * Vpp^2 * n_cycles * PRF  per channel
        return acq.nRx * self.capacitance_f * acq.V_pp ** 2 * acq.n_cycles * acq.PRF


# ── Pulse board — send the pulse ──────────────────────────────────────────
class Pulser:
    """Pulse board: a generic multi-channel HV pulser + T/R switch. Models the
    chip electronics only (per active channel); the transmit/load energy belongs
    to the Transducer (see Transducer.transmit_power)."""
    def __init__(self, channels_exposed=8, power_per_channel_w=0.1e-3):
        self.channels_exposed = channels_exposed
        self.power_per_channel_w = power_per_channel_w

    def configure(self, acq):
        acq.duty = min(acq.rx_window * acq.PRF, 1.0)   # on fraction = RX window x PRF

    def power(self, acq):
        return acq.nRx * self.power_per_channel_w   # chip, per active channel


# ── AFE board — condition the echo ────────────────────────────────────────
class EnvelopeAFE:
    """AFE board: envelope detector (amp -> rectifier -> low-pass), ~4x bandwidth
    cut. Power is the op-amps' always-on quiescent draw, per channel (the diode
    and RC filter are negligible by comparison)."""
    def __init__(self, bandwidth_reduction=4, amplifiers=4, amp_quiescent_a=1.0e-3,
                 amp_disabled_a=2.4e-6, supply_voltage_v=10.0):
        self.bandwidth_reduction = bandwidth_reduction
        self.amplifiers = amplifiers
        self.amp_quiescent_a = amp_quiescent_a
        self.amp_disabled_a = amp_disabled_a
        self.supply_voltage_v = supply_voltage_v

    def bandwidth_factor(self, mode):
        return 1 if mode == "RF" else self.bandwidth_reduction

    def power(self, acq):
        # amps enabled the on-fraction (duty) if duty-cycled, else continuously;
        # disabled the rest. Per active channel.
        on = acq.duty if acq.duty_cycled else 1.0
        i_avg = self.amp_quiescent_a * on + self.amp_disabled_a * (1 - on)
        return acq.nRx * self.amplifiers * i_avg * self.supply_voltage_v


# ── Core board — digitize, sequence, compute ──────────────────────────────
class MixedSignalSoC:
    """Core board: a mixed-signal SoC (MCU + on-chip ADC; here dual 5 Msps,
    12-bit). An FPGA back-end (external ADC, higher fs) would be a sibling class."""
    def __init__(self, adc_nyquist_factor=2, adc_fs_max_hz=5e6, adc_fom_j=50e-15,
                 mcu_clock_mhz=250.0, mcu_current_per_mhz_a=70.1e-6,
                 mcu_stop_current_a=90e-6, mcu_compute_s=100e-6,
                 supply_voltage_v=3.3, feature_count=16, feature_bits=16):
        self.adc_nyquist_factor = adc_nyquist_factor
        self.adc_fs_max_hz = adc_fs_max_hz
        self.adc_fom_j = adc_fom_j
        self.mcu_clock_mhz = mcu_clock_mhz
        self.mcu_current_per_mhz_a = mcu_current_per_mhz_a
        self.mcu_stop_current_a = mcu_stop_current_a
        self.mcu_compute_s = mcu_compute_s
        self.supply_voltage_v = supply_voltage_v
        self.feature_count = feature_count
        self.feature_bits = feature_bits

    def acquire(self, acq, afe):         # derive fs, N from transducer + AFE
        nyq = self.adc_nyquist_factor * acq.f_Tx
        acq.fs = nyq / afe.bandwidth_factor(acq.mode)
        acq.N = acq.fs * acq.t_acq
        return acq.fs <= self.adc_fs_max_hz          # fits on-chip ADC?

    def compute(self, acq):              # MCU decides the payload to ship
        if acq.mode == "features":       # a feature = one derived scalar (feature_bits wide)
            acq.data_rate = self.feature_count * self.feature_bits * acq.PRF * acq.nRx
        else:                            # RF / BWR differ only via fs -> N
            acq.data_rate = acq.N * acq.bits * acq.PRF * acq.nRx

    def power(self, acq):
        p_adc = self.adc_fom_j * (2 ** acq.bits) * acq.fs * acq.duty * acq.nRx
        run = self.mcu_current_per_mhz_a * self.mcu_clock_mhz * self.supply_voltage_v
        if not acq.duty_cycled:
            return p_adc + run                       # always-on: full run power
        # heavily duty-cycled (STOP between pulses): MCU wakes at run power to acquire
        # (+ extract features in features mode), and sits in STOP the rest of the period.
        compute = self.mcu_compute_s * acq.PRF if acq.mode == "features" else 0.0
        awake = min(acq.duty + compute, 1.0)         # acquire window + DSP, per period
        stop = self.mcu_stop_current_a * self.supply_voltage_v
        return p_adc + run * awake + stop * (1.0 - awake)


# ── Wireless link — downstream, NOT a ModulUS board ───────────────────────
class BLELink:
    """Wireless link: Bluetooth Low Energy."""
    def __init__(self, energy_per_bit_j=20e-9, throughput_max_bps=1e6):
        self.energy_per_bit_j = energy_per_bit_j
        self.throughput_max_bps = throughput_max_bps

    def fits(self, acq):
        return acq.data_rate <= self.throughput_max_bps

    def power(self, acq):
        return acq.data_rate * self.energy_per_bit_j


# ── Battery — external power source ───────────────────────────────────────
class LiIonBattery:
    """Battery: sizes the wearable from the average power."""
    def __init__(self, density_vol_wh_l=400.0, density_grav_wh_kg=230.0,
                 reference_cr2032_wh=0.65):
        self.density_vol_wh_l = density_vol_wh_l
        self.density_grav_wh_kg = density_grav_wh_kg
        self.reference_cr2032_wh = reference_cr2032_wh

    def size(self, P_avg, days=1):
        Wh = P_avg * 86400 * days / 3600.0
        return dict(Wh=Wh, vol_cm3=Wh / self.density_vol_wh_l * 1000.0,
                    mass_g=Wh / self.density_grav_wh_kg * 1000.0,
                    n_cr2032=Wh / self.reference_cr2032_wh)


def _params(section):
    """The {param: value} dict for a config section (drops ref / status)."""
    return {k: _v(v) for k, v in CFG[section]["params"].items()}


def fom_mw_per_mhz(P_mW, nRx, f_Tx):     # paper FoM: avg power / Rx ch / f_Tx
    return P_mW / nRx / (f_Tx / 1e6)


# ── Motherboard = System: construct the boards, then pass Acq through them ─
class System:
    """The Motherboard. Constructs each board from its config params; walk()
    passes the Acq through the boards. Swap a board by changing its class here."""

    def __init__(self):
        self.transducer = Transducer(**_params("transducer"))
        self.pulse   = Pulser(**_params("pulse"))
        self.echo    = EnvelopeAFE(**_params("echo"))
        self.core    = MixedSignalSoC(**_params("core"))
        self.radio   = BLELink(**_params("radio"))
        self.battery = LiIonBattery(**_params("battery"))

    def walk(self, acq):                 # Acq passes through the boards (signal-chain order)
        self.pulse.configure(acq)        # -> duty
        fits_adc = self.core.acquire(acq, self.echo)   # -> fs, N
        self.core.compute(acq)           # -> data_rate
        P = {"Pulse": self.pulse.power(acq) + self.transducer.transmit_power(acq),
             "Echo":  self.echo.power(acq),
             "Core":  self.core.power(acq),
             "Radio": self.radio.power(acq)}
        P_avg = sum(P.values())
        return dict(acq=acq, P=P, P_avg=P_avg,
                    axial_res_mm=self.transducer.axial_res(acq.f_Tx, acq.n_cycles) * 1e3,
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


# ── Sanity check (proves the plain-class refactor preserves the numbers) ──
if __name__ == "__main__":
    sys = System()
    print("boards:", ", ".join(b.__class__.__name__ for b in
          (sys.pulse, sys.echo, sys.core, sys.radio, sys.battery)))

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
