import numpy as np

# =========================================================
# ROTOR MODEL (Bramwell-based minimal solver)
# =========================================================
class RotorModel:

    def __init__(self,
                 R=6.0,          # radius [m]
                 Nb=4,           # blades
                 RPM=360,        # rotation speed
                 rho=1.225,      # air density
                 chord=0.4,      # blade chord
                 CLalpha=5.7,    # lift slope [1/rad]
                 Cd0=0.012,      # drag coefficient
                 theta0_deg=5.0, # collective [deg]
                 V=0.0           # forward speed [m/s]
                 ):

        # --- basic parameters ---
        self.R = R
        self.Nb = Nb
        self.rho = rho
        self.chord = chord
        self.CLalpha = CLalpha
        self.Cd0 = Cd0

        self.theta0 = np.deg2rad(theta0_deg)
        self.V = V

        self.Omega = RPM * 2*np.pi / 60
        self.A = np.pi * R**2

        # discretization
        self.Nr = 40
        self.r = np.linspace(0.2*R, R, self.Nr)
        self.dr = self.r[1] - self.r[0]

    # =====================================================
    # SOLVE INFLOW (momentum theory)
    # =====================================================
    def solve_inflow(self, CT):

        mu = self.V / (self.Omega * self.R)

        lam = 0.05  # initial guess

        for _ in range(100):
            lam_new = CT / (2*np.sqrt(mu**2 + lam**2 + 1e-6))
            if abs(lam_new - lam) < 1e-5:
                break
            lam = lam_new

        return lam

    # =====================================================
    # BLADE ELEMENT CALCULATION
    # =====================================================
    def blade_element(self, lam):

        T = 0
        Q = 0

        for ri in self.r:

            psi = 0  # symmetric assumption (mean values)

            UT = self.Omega * ri
            UP = self.V*np.sin(psi) + lam*self.Omega*self.R

            Vrel = np.sqrt(UT**2 + UP**2)
            phi = np.arctan2(UP, UT)

            alpha = self.theta0 - phi
            alpha = np.clip(alpha, -0.35, 0.35)

            CL = self.CLalpha * alpha
            CD = self.Cd0 + 0.01 * CL**2

            q = 0.5 * self.rho * Vrel**2

            dL = q * self.chord * CL * self.dr
            dD = q * self.chord * CD * self.dr

            dT = dL*np.cos(phi) - dD*np.sin(phi)
            dQ = ri*(dD*np.cos(phi) + dL*np.sin(phi))

            T += dT
            Q += dQ

        T *= self.Nb
        Q *= self.Nb

        return T, Q

    # =====================================================
    # MAIN SOLVER
    # =====================================================
    def solve(self):

        # initial guess
        CT = 0.005

        for _ in range(50):

            lam = self.solve_inflow(CT)

            T, Q = self.blade_element(lam)

            CT_new = T / (self.rho * self.A * (self.Omega*self.R)**2)

            if abs(CT_new - CT) < 1e-5:
                break

            CT = CT_new

        # final values
        P = Q * self.Omega
        CP = P / (self.rho * self.A * (self.Omega*self.R)**3)

        # flapping estimate (Bramwell approx)
        mu = self.V / (self.Omega*self.R)

        beta_1s = (8/3)*mu*self.theta0
        beta_1c = -(4/3)*lam

        return {
            "Thrust (N)": T,
            "Torque (Nm)": Q,
            "Power (W)": P,
            "CT": CT,
            "CP": CP,
            "lambda": lam,
            "mu": mu,
            "beta_1s (rad)": beta_1s,
            "beta_1c (rad)": beta_1c
        }


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    rotor = RotorModel(
        R=6,
        Nb=4,
        RPM=360,
        theta0_deg=6,
        V=0
    )

    results = rotor.solve()

    print("\n=== ROTOR RESULTS ===\n")
    for k,v in results.items():
        print(f"{k:20s}: {v:.5f}")