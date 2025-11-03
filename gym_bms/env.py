import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from .params import ecm_params

# Utility: seeded RNG wrapper
class RNG:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
    def uniform(self, low, high, size=None):
        return self.rng.uniform(low, high, size=size)
    def choice(self, a):
        return self.rng.choice(a)
    def integers(self, low, high=None, size=None):
        return self.rng.integers(low, high, size=size)

def load_ocv_table(csv_path):
    df = pd.read_csv(csv_path)
    soc = df['soc'].to_numpy()
    ocv = df['ocv_v'].to_numpy()
    # ensure sorted by soc
    idx = np.argsort(soc)
    return soc[idx], ocv[idx]

class BatteryPackEnv(gym.Env):
    """5 cells in series, each with a parallel 6Ω shunt + switch.
    ΔT = 30s per step. Action: 5-bit vector (0/1). Some actions ruled invalid
    (e.g., switching ON the lowest-SoC cell during discharge).
    State uses 'retracing': {t, t-ΔT, t-2ΔT} of [SoC(5), SwitchOn(5), SoH(5), I_avg].
    Reward: r = -log(Var(SoC_t))^2 - 4 - β * (#switch_changes) + 0.5*1_working,
    where β = 0.1*abs(first_term)+1.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, seed=42, ocv_csv="data/ocv_table.csv",
                 profile="discharge-rest-charge",
                 dt_s=30.0, R_shunt=6.0, n_cells=5):
        super().__init__()
        self.n = n_cells
        self.dt = dt_s
        self.R_shunt = R_shunt
        self.seed(seed)
        self.rng = RNG(seed)
        self.soc_grid, self.ocv_grid = load_ocv_table(ocv_csv)

        # Observation: retracing of 3 frames, each: SoC(5), switches(5), SoH(5), I_avg(1) = 16 dims -> 48 total
        obs_dim = 16 * 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Action: 5-bit binary as discrete 32
        self.action_space = spaces.Discrete(2**self.n)

        # Pack constants
        self.Vmin = 2.5
        self.Vmax = 4.2
        self.soc_min = 0.10  # pack cutoffs per paper setup
        self.soc_max = 0.90

        # Load profile parameters (±2.35A at ~0.7C)
        self.I_mag = 2.35
        self.profile = profile

        # Initialize
        self.reset(seed=seed)

    def seed(self, seed=None):
        np.random.seed(seed)

    # OCV interpolation
    def OCV(self, soc):
        soc = np.clip(soc, 0.0, 1.0)
        return np.interp(soc, self.soc_grid, self.ocv_grid)

    def _profile_current(self, t):
        # Two profiles (Fig. 8):
        # discharge-rest-charge: 0..1900s: +2.35A (discharge); 1900..2000 rest; 2000..6200 charge -2.35A
        # charge-rest-discharge: 0..2300 charge -2.35A; 2300..2400 rest; 2400..6600 discharge +2.35A
        if self.profile == "discharge-rest-charge":
            if t < 1900: return +self.I_mag
            if t < 2000: return 0.0
            if t < 6200: return -self.I_mag
            return 0.0
        else:
            if t < 2300: return -self.I_mag
            if t < 2400: return 0.0
            if t < 6600: return +self.I_mag
            return 0.0

    def _pack_working(self):
        # Pack is considered working if all cells within SoC window
        return float((self.soc >= self.soc_min).all() and (self.soc <= self.soc_max).all())

    def _invalid_action(self, action_bits, I_load):
        # Example impermissible rule from paper: during discharge (I>0), you should NOT turn on the lowest-SoC cell.
        if I_load > 0:
            lowest = np.argmin(self.soc)
            if action_bits[lowest] == 1:
                return True
        return False

    def _switch_cost(self, new_bits):
        # Count bit changes
        switches = np.sum(new_bits != self.switch_on)
        return switches

    def step(self, action):
        # Decode action to 5-bit vector
        bits = np.array([(action >> i) & 1 for i in range(self.n)], dtype=np.int32)

        # Current from profile
        I_load = self._profile_current(self.t)

        # Filter invalid actions: if invalid, treat as no-op but with small penalty
        invalid = self._invalid_action(bits, I_load)
        if invalid:
            bits = self.switch_on.copy()  # no change

        # Switch cost
        n_sw = self._switch_cost(bits)

        # Apply: compute cell currents (series load + optional shunt discharge)
        # Series load current is same for all cells: I_load
        # Shunt adds extra discharge current: (V_cell / R_shunt) if switch ON
        # We'll simulate over dt using ECM with 2 RC pairs (parameters depend on SoC).
        new_soc = self.soc.copy()
        new_iR1 = self.iR1.copy()
        new_iR2 = self.iR2.copy()

        # average load for state
        self.I_avg_prev2 = self.I_avg_prev
        self.I_avg_prev = self.I_avg
        self.I_avg = I_load

        for i in range(self.n):
            r0, r1, c1, r2, c2 = ecm_params(self.soc[i])
            # terminal voltage needs states iR1,iR2; but for shunt current we use OCV approx (common simplification)
            V_oc = self.OCV(self.soc[i])
            V_term = V_oc - r0*I_load - r1*self.iR1[i] - r2*self.iR2[i]
            I_shunt = (V_term / self.R_shunt) if bits[i] == 1 else 0.0
            I_cell = I_load + I_shunt  # positive = discharge

            # SoC update: SoC[k+1] = SoC[k] - (eta*dt/Q)*I
            # Use capacity Q_i via SoH multiplier around nominal 3.4 Ah
            Q_nom = 3.4  # Ah
            Q_i = Q_nom * self.soh[i]
            eta = 1.0 if I_cell >= 0 else 0.99  # coulombic efficiency
            dSoC = -(eta * self.dt / 3600.0) * (I_cell / Q_i)
            new_soc[i] = np.clip(self.soc[i] + dSoC, 0.0, 1.0)

            # RC branch currents (discrete)
            exp1 = math.exp(-self.dt / (r1*c1))
            exp2 = math.exp(-self.dt / (r2*c2))
            new_iR1[i] = exp1*self.iR1[i] + (1 - exp1)*I_load
            new_iR2[i] = exp2*self.iR2[i] + (1 - exp2)*I_load

        self.soc = new_soc
        self.iR1 = new_iR1
        self.iR2 = new_iR2
        self.switch_on = bits
        self.t += self.dt

        # Reward
        var_soc = float(np.var(self.soc))
        first_term = - (math.log(max(var_soc, 1e-9))**2) - 4.0
        beta = 0.1*abs(first_term) + 1.0
        r = first_term - beta * n_sw + 0.5 * self._pack_working()
        if invalid:
            r -= 0.5  # small penalty for proposing invalid

        # Episode termination: when profile ends or pack not working anymore
        terminated = False
        # profile_end = 6200.0 if self.profile == "discharge-rest-charge" else 6600.0
        # out_of_bounds = not bool(self._pack_working())
        # if self.t >= profile_end or out_of_bounds:
        #     terminated = True
        profile_end = 1e12  # allow full discharge to natural cutoff for capacity evaluation
        terminated = not bool(self._pack_working())

        obs = self._get_obs()
        info = {"var_soc": var_soc, "switches": int(n_sw), "invalid_action": bool(invalid), "I_load": I_load}
        return obs, r, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = RNG(seed or 0)
        # Initial SoC ~ U(0.4, 0.6) as in paper
        self.soc = self.rng.uniform(0.4, 0.6, size=self.n)
        # SoH ~ around 1.0 with mild variation
        self.soh = 1.0 + self.rng.uniform(-0.05, 0.05, size=self.n)
        self.iR1 = np.zeros(self.n, dtype=np.float64)
        self.iR2 = np.zeros(self.n, dtype=np.float64)
        self.switch_on = np.zeros(self.n, dtype=np.int32)
        self.t = 0.0
        self.I_avg = 0.0
        self.I_avg_prev = 0.0
        self.I_avg_prev2 = 0.0
        self._frame_t = self._frame(self.soc, self.switch_on, self.soh, self.I_avg)
        self._frame_t1 = self._frame(self.soc, self.switch_on, self.soh, self.I_avg_prev)
        self._frame_t2 = self._frame(self.soc, self.switch_on, self.soh, self.I_avg_prev2)
        return self._get_obs(), {}

    def _frame(self, soc, switches, soh, Iavg):
        # Normalize to [-1,1]: soc in [0,1] -> [-1,1]; switches in {0,1}->{-1,1}; soh ~ [0.9,1.1] -> approx normalize; Iavg in [-I_mag, I_mag]
        def s01(x): return 2.0*x - 1.0
        f_soc = s01(soc)
        f_sw = switches*2 - 1
        f_soh = (soh - 1.0) / 0.1  # roughly [-0.5,0.5] -> [-5,5]; then clip
        f_soh = np.clip(f_soh, -1.0, 1.0)
        f_I = np.array([Iavg / self.I_mag], dtype=np.float64)  # in [-1,1]
        return np.concatenate([f_soc, f_sw, f_soh, f_I])

    def _get_obs(self):
        self._frame_t2 = self._frame_t1
        self._frame_t1 = self._frame_t
        self._frame_t = self._frame(self.soc, self.switch_on, self.soh, self.I_avg)
        obs = np.concatenate([self._frame_t, self._frame_t1, self._frame_t2]).astype(np.float32)
        return obs

    def render(self):
        pass
