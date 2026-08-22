# Published reference data

`mcdevitt_2019_ppcf_fig3.csv` contains a manual/digital extraction of the
central markers and plotted vertical error-bar limits from Figure 3 of:

C. J. McDevitt, Z. Guo, and X.-Z. Tang, "Spatial transport of runaway
electrons in axisymmetric tokamak plasmas," *Plasma Physics and Controlled
Fusion* **61**, 024004 (2019).

The source PDF was rendered at 300 dpi.  Pixel coordinates were calibrated
against the plotted `r/a = 0, 0.2, 0.4, 0.6, 0.8` ticks and the logarithmic
`D/D0 = 1, 10` ticks.  The values therefore carry ordinary figure-digitization
uncertainty and are not claimed to be the authors' underlying raw data.  The
error-bar columns reproduce the visible plotted limits; the paper caption does
not assign a statistical interpretation to those bars, so JONTA deliberately
does not label them as a specific confidence interval.

The red dashed curve in the paper is *not* digitized into the CSV.  JONTA
computes it independently from Eq. (1).  For the fully ionized pure-Z case,
using `A f_t ~= 0.689 sqrt(2 epsilon)` and the paper's definition of `D0`, Eq.
(1) reduces to `D_non-rel / D0 = c/v`.  Figure 3 applies this scaling to the
10-keV population.

`mcdevitt_2019_ppcf_fig6.csv` contains the corresponding digitization of
Figure 6 from the same paper.  Figure 6 repeats the monoenergetic isotropic
transport scan for deuterium plus singly ionized argon with
`n_Ar+ = n_D/10`, using the partially screened collision coefficients.  The
PDF was rendered at 250 dpi and calibrated against the plotted `r/a` ticks and
logarithmic `D/D0 = 10, 100` ticks.  The central values and visible vertical
error-bar limits therefore have ordinary plot-digitization uncertainty.  The
paper does not assign a statistical interpretation to those bars.

For Figure 6 JONTA also evaluates the asymptotic banana-regime scaling implied
by Eqs. (7)-(8).  With the same small-inverse-aspect-ratio trapped-fraction
factor used to define Eq. (3), this reduces to a comparison proportional to the
partially screened deflection coefficient `C_B = p^2 tau_c nu_D^ps`, with the
`tau_c^a/tau_c` correction required because `D0` is normalized to the main-ion
electron density while the simulation collision time uses the total free
electron density.

`guo_2017_ppcf_bump.csv` stores the two bump momenta explicitly labeled in
Figure 9 of Z. Guo, C. J. McDevitt, and X.-Z. Tang, *Plasma Physics and
Controlled Fusion* **59**, 044003 (2017), together with the O-X disappearance
threshold stated in Sec. 5.  These values are transcribed from labels/text, not
digitized estimates.  The continuous comparison curves are computed directly
from Guo Eqs. (22)-(24), not stored as sampled reference data.

`mcdevitt_2019_ppcf_large_angle.csv` contains manual digitizations of the
large-angle/avalanche validation curves in C. J. McDevitt, Z. Guo, and
X.-Z. Tang, "Avalanche mechanism for runaway electron amplification in a
tokamak plasma," *Plasma Physics and Controlled Fusion* **61**, 054008
(2019).  The `B3a` rows are the green Monte-Carlo markers in Appendix
Figure B3(a), giving `gamma_av tau_c` versus `E/Ec` for `alpha=0.5` and
`Zeff=2`.  The `13` rows are the colored markers in Figure 13, giving the
conservative avalanche growth rate versus `gamma_min^LA-1` at
`E/Ec = 2.05, 2.25, 2.5, 3`.

These are figure extractions rather than author-supplied raw values, so they
carry ordinary plot-digitization uncertainty.  Analytical curves are not
stored as sampled data: the Appendix-B3 threshold comparison is evaluated
directly from Eq. (B15), and the conservative/source-only Coulomb logarithms
are evaluated directly from Eqs. (32) and (33).  Figure 14 is used primarily
as a model-to-model consistency target (the conservative and source-only
growth rates are nearly coincident) rather than digitizing two curves whose
separation is comparable to the graphical line width.
