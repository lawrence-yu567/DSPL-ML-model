import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c as light_speed

rng = np.random.default_rng(seed=100)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


N_ROWS = 10000
ARCSEC_PER_RAD = 206264.80624709636
c = light_speed.to("km/s").value


SIGMA_V_RANGE = (150.0, 300.0)
SOURCE_SIGMA_V_RANGE = (80.0, 200.0)
Q_RANGE = (0.5, 1.0)
PHI_RANGE = (0, 2*np.pi)
ZLENS_RANGE = (0.2, 0.4)
ZSOURCE_MIN_OFFSET = 0.3
ZSOURCE_MAX = 4.0


def velocity_dispersion_to_theta_e(sig_v, zl, zs):
    d_s = cosmo.angular_diameter_distance(zs).value
    d_ls = cosmo.angular_diameter_distance_z1z2(zl, zs).value
    return 4 * np.pi * (sig_v/c)**2 * (d_ls/d_s) * ARCSEC_PER_RAD

def axis_ratio_to_e1e2(q, phi):
    e = (1.0 - q) / (1.0 + q)
    e1 = e * np.cos(2 * phi)
    e2 = e * np.sin(2 * phi)
    return e1, e2


def main():
 
    sigma_v = rng.uniform(*SIGMA_V_RANGE, size=N_ROWS)
    q = rng.uniform(*Q_RANGE, size=N_ROWS)
    phi = rng.uniform(*PHI_RANGE, size=N_ROWS)
    z_lens = rng.uniform(*ZLENS_RANGE, size=N_ROWS)

    # --- source galaxy's own mass model ---
    sigma_v_source = rng.uniform(*SOURCE_SIGMA_V_RANGE, size=N_ROWS)
    q_source = rng.uniform(*Q_RANGE, size=N_ROWS)
    phi_source = rng.uniform(*PHI_RANGE, size=N_ROWS)
 
    # z_source uniform in [z_lens + offset, ZSOURCE_MAX], resampled where needed
    z_source = np.empty(N_ROWS)
    for i in range(N_ROWS):
        lo = z_lens[i] + ZSOURCE_MIN_OFFSET
        z_source[i] = rng.uniform(lo, ZSOURCE_MAX)
 
    theta_e = np.array([
        velocity_dispersion_to_theta_e(sv, zl, zs)
        for sv, zl, zs in zip(sigma_v, z_lens, z_source)
    ])
    e1, e2 = axis_ratio_to_e1e2(q, phi)


    theta_e_source = np.array([
        velocity_dispersion_to_theta_e(sv, zl, zs)
        for sv, zl, zs in zip(sigma_v_source, z_lens, z_source)
    ])
    e1_source, e2_source = axis_ratio_to_e1e2(q_source, phi_source)

    weight = np.ones(N_ROWS)

    header = ["z", "z_source", "theta_E", "e1", "e2",
              "theta_E_source", "e1_source", "e2_source", "WEIGHT"]
    header = "\t".join(header)

    out = np.column_stack([
        z_lens, z_source, theta_e, e1, e2,
        theta_e_source, e1_source, e2_source,
        weight,
    ])
    
    np.savetxt(
        "physical_lens_table.txt",
        out,
        header=header,
        comments="",
        delimiter="\t",
        fmt="%.6f",
    )
    print(f"Wrote {N_ROWS} rows to physical_lens_table.txt")
    print(f"theta_E range: {theta_e.min():.3f} - {theta_e.max():.3f} arcsec")
    print(f"e (=|e1,e2|) range: {np.hypot(e1, e2).min():.3f} - {np.hypot(e1, e2).max():.3f}")

main()