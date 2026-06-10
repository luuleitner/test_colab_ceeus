"""ModulUS digital twin — lean software model of the modular US sandbox.

SOFTWARE STRUCTURE  (how this file is wired; the HARDWARE diagram is in config.yaml)

   config.yaml ──load_config()──► CFG          (board parameters + refs)
                                    │ _params(section)
                                    ▼
   System  ── builds one board object per config section ──┐
                                                            │
   Acq(f_Tx, bits, nRx, PRF, D, mode, ...)  ── the knobs ──┤
                                                            ▼
                                       Run(system, acq)
                                         the per-configuration results, each one a
                                         property derived from the knobs:
                                            d.duty  d.fs  d.N  d.data_rate
                                            d.power  d.P_avg
                                            d.fits_onchip  d.fits_ble
                                            d.within_channels  d.within_depth
                                            d.axial_res_mm  d.fom

   usage:  d = System().run(Acq(...));  then  d.P_avg, d.fits_ble, d.power

SCOPE — this is a first-order POWER / DATA-RATE / FEASIBILITY model. For a given
configuration it estimates: sampling (fs, N), data rate, average power per board,
battery size, axial resolution, and whether the design fits the on-chip ADC and
the wireless link. It does NOT simulate the ultrasound signal, image, beamforming,
SNR, or circuit-level electronics. (synth_rf_envelope / load_traces below are demo
traces for the notebook, not part of this model.)

Each board (Pulser, EnvelopeAFE, MixedSignalSoC, BLELink, LiPoBattery) is a plain
class that owns its parameters and a short power() method. Run asks each board for
its contribution and assembles the per-configuration results. System names each
board's class — swap one by changing its class there. Transducer, Radio (BLE),
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


# ── Acq: the run-time knobs that define one acquisition ──────────────────────
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
        assert mode in ("RF", "BWR", "features"), f"mode must be RF/BWR/features, got {mode!r}"

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
        # energy to charge/discharge the element each cycle, summed over active
        # channels: ~ C * Vpp^2 * n_cycles * PRF (a proportionality; this transmit
        # term is small next to the pulser chip power).
        return acq.nRx * self.capacitance_f * acq.V_pp ** 2 * acq.n_cycles * acq.PRF


# ── Pulse board — send the pulse ──────────────────────────────────────────
class Pulser:
    """Pulse board: a generic multi-channel HV pulser + T/R switch. Models the
    chip electronics only (per active channel); the transmit/load energy belongs
    to the Transducer (see Transducer.transmit_power)."""
    def __init__(self, channels_exposed=8, power_per_channel_w=0.1e-3):
        self.channels_exposed = channels_exposed
        self.power_per_channel_w = power_per_channel_w

    def duty(self, acq):
        return min(acq.rx_window * acq.PRF, 1.0)   # on fraction = RX window x PRF

    def power(self, acq):
        return acq.nRx * self.power_per_channel_w   # chip, per active channel


# ── AFE board — condition the echo ────────────────────────────────────────
class EnvelopeAFE:
    """AFE board: envelope detector (amp -> rectifier -> low-pass), ~4x bandwidth
    cut. Power is the op-amps' quiescent draw per channel -- enabled the on-fraction
    (duty) if duty-cycled, else continuously (the diode and RC are negligible)."""
    def __init__(self, bandwidth_reduction=4, amplifiers=4, amp_quiescent_a=1.0e-3,
                 amp_disabled_a=2.4e-6, supply_voltage_v=10.0):
        self.bandwidth_reduction = bandwidth_reduction
        self.amplifiers = amplifiers
        self.amp_quiescent_a = amp_quiescent_a
        self.amp_disabled_a = amp_disabled_a
        self.supply_voltage_v = supply_voltage_v

    def bandwidth_factor(self, mode):    # narrows the band so the Core can sample slower
        return 1 if mode == "RF" else self.bandwidth_reduction

    def power(self, acq, duty):          # op-amp current, summed over the Rx channels
        on = duty if acq.duty_cycled else 1.0       # fraction of the period the amps are on
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

    def sample_rate(self, acq, afe):     # fs from transducer freq + AFE bandwidth cut
        nyq = self.adc_nyquist_factor * acq.f_Tx
        return nyq / afe.bandwidth_factor(acq.mode)

    def data_rate(self, acq, N):         # bits/s the MCU ships, per mode
        if acq.mode == "features":       # a feature = one derived scalar (feature_bits wide)
            return self.feature_count * self.feature_bits * acq.PRF * acq.nRx
        return N * acq.bits * acq.PRF * acq.nRx     # RF / BWR differ only via fs -> N

    def power(self, acq, duty, fs):
        p_adc = self.adc_fom_j * (2 ** acq.bits) * fs * duty * acq.nRx
        run = self.mcu_current_per_mhz_a * self.mcu_clock_mhz * self.supply_voltage_v
        if not acq.duty_cycled:
            return p_adc + run                       # always-on: full run power
        # heavily duty-cycled (STOP between pulses): MCU wakes at run power to acquire
        # (+ extract features in features mode), and sits in STOP the rest of the period.
        dsp = self.mcu_compute_s * acq.PRF if acq.mode == "features" else 0.0
        awake = min(duty + dsp, 1.0)                 # acquire window + DSP, per period
        stop = self.mcu_stop_current_a * self.supply_voltage_v
        return p_adc + run * awake + stop * (1.0 - awake)


# ── Wireless link — downstream, NOT a ModulUS board ───────────────────────
class BLELink:
    """Wireless link: a low-power BLE SoC, downstream of ModulUS -- NOT a sandbox
    board. energy/bit = the radio's TX draw spread over the PHY bit rate, inflated by
    RX/protocol overhead; data_rate must stay under the practical throughput ceiling."""
    def __init__(self, tx_current_a=3.2e-3, supply_voltage_v=3.0, phy_bitrate_bps=1.0e6,
                 overhead_factor=2.0, throughput_max_bps=300e3):
        self.tx_current_a = tx_current_a
        self.supply_voltage_v = supply_voltage_v
        self.phy_bitrate_bps = phy_bitrate_bps
        self.overhead_factor = overhead_factor
        self.throughput_max_bps = throughput_max_bps

    def energy_per_bit(self):
        # effective J per delivered bit: TX draw / PHY rate, x protocol + RX overhead
        return self.overhead_factor * self.tx_current_a * self.supply_voltage_v / self.phy_bitrate_bps

    def power(self, acq, data_rate):
        return data_rate * self.energy_per_bit()


# ── Battery — external power source ───────────────────────────────────────
class LiPoBattery:
    """Battery: a Li-polymer cell, sized from the average power (also reported as
    an equivalent CR2032 coin-cell count for intuition)."""
    def __init__(self, density_vol_wh_l=313.0, density_grav_wh_kg=202.0,
                 reference_cr2032_wh=0.65):
        self.density_vol_wh_l = density_vol_wh_l
        self.density_grav_wh_kg = density_grav_wh_kg
        self.reference_cr2032_wh = reference_cr2032_wh

    def size(self, P_avg, days=1):       # energy for `days` of runtime -> cell size
        Wh = P_avg * 86400 * days / 3600.0          # average power held over the period
        return dict(Wh=Wh, vol_cm3=Wh / self.density_vol_wh_l * 1000.0,
                    mass_g=Wh / self.density_grav_wh_kg * 1000.0,
                    n_cr2032=Wh / self.reference_cr2032_wh)


def _params(section):
    """The {param: value} dict for a config section (drops ref / status)."""
    return {k: _v(v) for k, v in CFG[section]["params"].items()}


def fom_mw_per_mhz(P_mW, nRx, f_Tx):     # paper FoM: avg power / Rx ch / f_Tx
    return P_mW / nRx / (f_Tx / 1e6)


# ── Run: one run of the model (a System + an Acq); results are properties ─────
class Run:
    """The model's results for one configuration: a System (the assembled boards)
    plus one Acq (the knobs). It follows the acquisition chain and exposes each
    stage as a property — the RX duty, the sampling rate fs and sample count N, the
    link data_rate, each board's power and the total P_avg, the four feasibility
    walls (on-chip ADC, BLE link, channel count, PRF-vs-depth), and the axial
    resolution and mW/MHz figure of merit."""
    def __init__(self, system, acq):
        self.system = system
        self.acq = acq

    # the signal + timing chain: knobs -> duty -> fs -> N -> data_rate ────
    @property
    def duty(self):       return self.system.pulse.duty(self.acq)
    @property
    def fs(self):         return self.system.core.sample_rate(self.acq, self.system.echo)
    @property
    def N(self):          return self.fs * self.acq.t_acq
    @property
    def data_rate(self):  return self.system.core.data_rate(self.acq, self.N)

    # power: each board's draw for this configuration, keyed by board ──────
    @property
    def power(self):
        a, b = self.acq, self.system          # a = the knobs, b = the boards
        return {"Pulse": b.pulse.power(a) + b.transducer.transmit_power(a),
                "Echo":  b.echo.power(a, self.duty),
                "Core":  b.core.power(a, self.duty, self.fs),
                "Radio": b.radio.power(a, self.data_rate)}

    @property
    def P_avg(self):      return sum(self.power.values())

    # feasibility walls + readouts ───────────────────────────────────────
    @property
    def fits_onchip(self):      return self.fs <= self.system.core.adc_fs_max_hz
    @property
    def fits_ble(self):         return self.data_rate <= self.system.radio.throughput_max_bps
    @property
    def within_channels(self):  return self.acq.nRx <= self.system.pulse.channels_exposed
    @property
    def within_depth(self):     return self.acq.rx_window * self.acq.PRF <= 1.0   # PRF <= c/2D
    @property
    def axial_res_mm(self):     return self.system.transducer.axial_res(self.acq.f_Tx, self.acq.n_cycles) * 1e3
    @property
    def fom(self):              return fom_mw_per_mhz(self.P_avg * 1e3, self.acq.nRx, self.acq.f_Tx)


# ── Motherboard = System: construct the boards; run(acq) -> a Run ──────────
class System:
    """The Motherboard. Constructs each board from its config params. Call
    device.run(acq) to get a Run whose properties are the results. Swap a board by
    changing its class here."""

    def __init__(self):
        self.transducer = Transducer(**_params("transducer"))
        self.pulse   = Pulser(**_params("pulse"))
        self.echo    = EnvelopeAFE(**_params("echo"))
        self.core    = MixedSignalSoC(**_params("core"))
        self.radio   = BLELink(**_params("radio"))
        self.battery = LiPoBattery(**_params("battery"))

    def run(self, acq):                  # this System evaluated at these knobs
        return Run(self, acq)


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
    """Return (rf, env, fs): the measured ModulUS .npz if present, otherwise a
    synthetic trace (with a loud notice) so a live demo never stalls on a missing file."""
    if Path(path).exists():
        d = np.load(path)
        return d["rf"], d["env"], float(d["fs"])
    print("=" * 60)
    print("  SYNTHETIC DATA  - real ModulUS file not found:")
    print(f"    {path}")
    print("  (load the acquired .npz to use measured RF + envelope)")
    print("=" * 60)
    return synth_rf_envelope()


# ── Self-test: run this file to print the model's outputs at a few op-points ─
if __name__ == "__main__":
    device = System()
    print("boards:", ", ".join(b.__class__.__name__ for b in
          (device.pulse, device.echo, device.core, device.radio, device.battery)))

    def row(tag, f_Tx, bits, nRx, PRF, D, mode):
        d = device.run(Acq(f_Tx, bits, nRx, PRF, D, mode))
        print(f"{tag:12s} fs={d.fs/1e6:5.1f}M  N={d.N:6.0f}  "
              f"dr={d.data_rate/1e6:7.3f}Mb/s  P={d.P_avg*1e3:7.2f}mW  "
              f"FoM={d.fom:5.2f}  BLE={'OK ' if d.fits_ble else 'OVER'}  "
              f"ADC={'on ' if d.fits_onchip else 'EXT'}  "
              f"res={d.axial_res_mm:.3f}mm")

    print("\n=== Operating point: 10 MHz, nRx=1, PRF=25 Hz, D=3 cm ===")
    for m in ("RF", "BWR", "features"):
        row(m, 10e6, 12, 1, 25, 0.03, m)

    print("\n=== Wall demo: 10 MHz, nRx=8, PRF=100 Hz, D=3 cm ===")
    for m in ("RF", "BWR", "features"):
        row(m, 10e6, 12, 8, 100, 0.03, m)

    print("\n=== Battery (BWR op-point, 1 day) ===")
    d = device.run(Acq(10e6, 12, 1, 25, 0.03, "BWR"))
    b = device.battery.size(d.P_avg)
    print(f"  P_avg={d.P_avg*1e3:.2f} mW -> {b['vol_cm3']:.3f} cm3, "
          f"{b['mass_g']:.2f} g, {b['n_cr2032']:.3f} CR2032")
