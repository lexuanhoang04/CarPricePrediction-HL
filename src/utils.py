import re
import numpy as np
import pandas as pd

_OWNER_MAP = {
    "UnRegistered Car": 0,
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth": 4,
    "4 or More": 5,
}

_TRANSMISSION_MAP = {"Manual": 0, "Automatic": 1}


def parse_engine_cc(s):
    """
    Examples:
      '1198 cc' -> 1198.0
      '1498 CC' -> 1498.0
      '1,198 cc' -> 1198.0
      NaN / ''  -> NaN
    """
    if pd.isna(s):
        return np.nan
    
    s = str(s).strip().lower()
    if not s:
        return np.nan

    # remove commas: "1,198" -> "1198"
    s = s.replace(",", "")

    m = re.search(r"(\d+(?:\.\d+)?)\s*cc", s)
    if m:
        return float(m.group(1))

    # fallback: just grab first number if "cc" missing
    m2 = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m2.group(1)) if m2 else np.nan

import re
import numpy as np
import pandas as pd

# --- helpers ---
num_re = r"(\d+(?:\.\d+)?)"   # int or float

def parse_value_and_rpm(s: str):
    """
    Parse strings like:
      '87 bhp @ 6000 rpm'  -> (87.0, 6000.0)
      '109 Nm @ 4500 rpm'  -> (109.0, 4500.0)
      '74 bhp'             -> (74.0, nan)
      nan / ''             -> (nan, nan)
    """
    if pd.isna(s):
        return (np.nan, np.nan)

    s = str(s).strip()
    if not s:
        return (np.nan, np.nan)

    # 1) value = first number in the string
    m_val = re.search(num_re, s)
    val = float(m_val.group(1)) if m_val else np.nan

    # 2) rpm = number after '@' (if present)
    m_rpm = re.search(r"@\s*" + num_re, s)
    rpm = float(m_rpm.group(1)) if m_rpm else np.nan

    return (val, rpm)

