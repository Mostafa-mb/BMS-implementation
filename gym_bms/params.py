import numpy as np

# Appendix table: SOC grid and ECM parameters (NCA cell)
_SOC_GRID = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])[::-1]  # ascending 0.1..0.9
R0 = np.array([0.1325,0.1070,0.1042,0.1040,0.1024,0.1023,0.1020,0.1016,0.1063])[::-1]
R1 = np.array([0.0498,0.0272,0.0272,0.0275,0.0271,0.0390,0.0315,0.0302,0.0303])[::-1]
C1 = np.array([747.54,982.50,1128.74,1161.67,1131.40,929.76,766.53,734.10,726.32])[::-1]
R2 = np.array([0.0096,0.0076,0.0077,0.0077,0.0078,0.0078,0.0105,0.0102,0.0099])[::-1]
C2 = np.array([639.61,746.24,788.39,791.77,789.73,613.48,575.44,594.41,636.78])[::-1]

def _interp(x, xgrid, y):
    # clip then linear interpolate
    x = np.clip(x, xgrid.min(), xgrid.max())
    return np.interp(x, xgrid, y)

def ecm_params(soc):
    """Return R0,R1,C1,R2,C2 for a given SoC in [0,1]."""
    soc = np.clip(soc, 0.1, 0.9)  # params defined in [0.1,0.9]
    r0 = _interp(soc, _SOC_GRID, R0)
    r1 = _interp(soc, _SOC_GRID, R1)
    c1 = _interp(soc, _SOC_GRID, C1)
    r2 = _interp(soc, _SOC_GRID, R2)
    c2 = _interp(soc, _SOC_GRID, C2)
    return r0, r1, c1, r2, c2
