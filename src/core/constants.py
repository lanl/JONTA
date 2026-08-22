"""Physical constants and legacy RAMc constants used by JONTA."""

from __future__ import annotations

import math

PI = math.pi
C_LIGHT_M_S = 2.99792458e8
C_LIGHT_CM_S = C_LIGHT_M_S * 100.0
E_CHARGE_C = 1.602176634e-19
M_E_KG = 9.1093837139e-31
ME_C2_J = M_E_KG * C_LIGHT_M_S**2
ME_C2_EV = ME_C2_J / E_CHARGE_C
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
R_E_M = 2.8179403262e-15
R_E_CM = R_E_M * 100.0
ALPHA_FS = 7.2973525643e-3
ALFVEN_CURRENT_A = 4.0 * PI * M_E_KG * C_LIGHT_M_S / (MU0 * E_CHARGE_C)

# RAMc uses the electron charge in Gaussian-cgs units in the normalized
# current deposition routine. Keep this explicit rather than hiding the
# legacy normalization in deposition code.
E_CHARGE_ESU = 4.80320471257e-10

# Tritium beta endpoint and half-life used in Ekmark et al. (2024).
TRITIUM_BETA_ENDPOINT_EV = 18.6e3
TRITIUM_HALF_LIFE_DAYS = 4500.0
TRITIUM_HALF_LIFE_S = TRITIUM_HALF_LIFE_DAYS * 86400.0
