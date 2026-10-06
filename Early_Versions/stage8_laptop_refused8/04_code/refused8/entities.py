"""Entity registry: Grid-India State control areas, their regions, power-exchange / DSM bid areas and name variants.

Bid areas follow CEA, Market Monitoring Cell Annual Report 2019-20 (Table "States covered under bid area"):
  N1 J&K and Ladakh, Himachal Pradesh, Chandigarh, Haryana   N2 Uttar Pradesh, Uttarakhand, Rajasthan, Delhi   N3 Punjab
  E1 West Bengal, Sikkim, Bihar, Jharkhand                    E2 Odisha
  W1 Madhya Pradesh   W2 Maharashtra, Gujarat, Daman & Diu, Dadra & Nagar Haveli, North Goa   W3 Chhattisgarh
  S1 Andhra Pradesh, Telangana, Karnataka, Puducherry (Yanam), South Goa   S2 Tamil Nadu, Puducherry   S3 Kerala
  A1 Tripura, Manipur, Mizoram, Nagaland                      A2 Assam, Arunachal Pradesh, Meghalaya
Conventions: Goa is assigned to W2 (North Goa); DVC (a control area in Jharkhand/West Bengal) to E1; Puducherry to S2.
DNH and DD reported separately until their merger are summed into one entity "DNHDD".
Industrial / railway entities reported by Grid-India (Essar Steel/AMNSIL, BALCO, RIL Jamnagar, Railways, bulk
consumers) are recognised but excluded from the State panel.
"""
import re

# id: (display name, region, bid area, pilot-18 member, geographic state for CEA/IMD joins or None)
ENTITIES = {
    "PB": ("Punjab", "NR", "N3", True, "Punjab"),
    "HR": ("Haryana", "NR", "N1", True, "Haryana"),
    "RJ": ("Rajasthan", "NR", "N2", True, "Rajasthan"),
    "DL": ("Delhi", "NR", "N2", True, "Delhi"),
    "UP": ("Uttar Pradesh", "NR", "N2", True, "Uttar Pradesh"),
    "UK": ("Uttarakhand", "NR", "N2", True, "Uttarakhand"),
    "HP": ("Himachal Pradesh", "NR", "N1", True, "Himachal Pradesh"),
    "JK": ("Jammu & Kashmir and Ladakh", "NR", "N1", True, "Jammu and Kashmir"),
    "CH": ("Chandigarh", "NR", "N1", False, "Chandigarh"),
    "CG": ("Chhattisgarh", "WR", "W3", True, "Chhattisgarh"),
    "GJ": ("Gujarat", "WR", "W2", True, "Gujarat"),
    "MP": ("Madhya Pradesh", "WR", "W1", True, "Madhya Pradesh"),
    "MH": ("Maharashtra", "WR", "W2", True, "Maharashtra"),
    "GA": ("Goa", "WR", "W2", False, "Goa"),
    "DNHDD": ("Dadra & Nagar Haveli and Daman & Diu", "WR", "W2", False, "Dadra and Nagar Haveli and Daman and Diu"),
    "AP": ("Andhra Pradesh", "SR", "S1", False, "Andhra Pradesh"),
    "TG": ("Telangana", "SR", "S1", False, "Telangana"),
    "KA": ("Karnataka", "SR", "S1", True, "Karnataka"),
    "KL": ("Kerala", "SR", "S3", False, "Kerala"),
    "TN": ("Tamil Nadu", "SR", "S2", True, "Tamil Nadu"),
    "PY": ("Puducherry", "SR", "S2", True, "Puducherry"),
    "BR": ("Bihar", "ER", "E1", True, "Bihar"),
    "DVC": ("Damodar Valley Corporation", "ER", "E1", False, None),
    "JH": ("Jharkhand", "ER", "E1", True, "Jharkhand"),
    "OD": ("Odisha", "ER", "E2", True, "Odisha"),
    "WB": ("West Bengal", "ER", "E1", False, "West Bengal"),
    "SK": ("Sikkim", "ER", "E1", False, "Sikkim"),
    "AR": ("Arunachal Pradesh", "NER", "A2", False, "Arunachal Pradesh"),
    "AS": ("Assam", "NER", "A2", False, "Assam"),
    "MN": ("Manipur", "NER", "A1", False, "Manipur"),
    "ML": ("Meghalaya", "NER", "A2", False, "Meghalaya"),
    "MZ": ("Mizoram", "NER", "A1", False, "Mizoram"),
    "NL": ("Nagaland", "NER", "A1", False, "Nagaland"),
    "TR": ("Tripura", "NER", "A1", False, "Tripura"),
}

# normalised alias -> id ("DD" and "DNH" map to DNHDD and are summed when both appear on a date)
_ALIASES = {
    "PUNJAB": "PB", "HARYANA": "HR", "RAJASTHAN": "RJ", "DELHI": "DL", "UP": "UP", "UTTARPRADESH": "UP",
    "UTTARAKHAND": "UK", "UTTARANCHAL": "UK", "HP": "HP", "HIMACHALPRADESH": "HP",
    "JK": "JK", "JANDK": "JK", "JAMMUANDKASHMIR": "JK", "JAMMUKASHMIR": "JK", "JKUTLADAKHUT": "JK", "JKANDLADAKH": "JK",
    "JAMMUKASHMIRLADAKH": "JK", "LADAKH": "JK", "CHANDIGARH": "CH",
    "CHHATTISGARH": "CG", "CHHATISGARH": "CG", "GUJARAT": "GJ", "MP": "MP", "MADHYAPRADESH": "MP", "MAHARASHTRA": "MH",
    "GOA": "GA", "DD": "DNHDD", "DNH": "DNHDD", "DNHDDPDCL": "DNHDD", "DAMANANDDIU": "DNHDD", "DAMANDIU": "DNHDD",
    "DADRANAGARHAVELI": "DNHDD", "DADRAANDNAGARHAVELI": "DNHDD", "DADRANAGARHAVELIANDDAMANDIU": "DNHDD",
    "DADRAANDNAGARHAVELIANDDAMANANDDIU": "DNHDD", "DNHDD": "DNHDD", "DADRAANDNAGAR": "DNHDD",
    "HIMACHAL": "HP", "CHATTISGARH": "CG", "ARUNACHAL": "AR",  # CEA CO2 database spellings
    "ANDHRAPRADESH": "AP", "TELANGANA": "TG", "KARNATAKA": "KA", "KERALA": "KL", "TAMILNADU": "TN",
    "PONDY": "PY", "PUDUCHERRY": "PY", "PONDICHERRY": "PY",
    "BIHAR": "BR", "DVC": "DVC", "JHARKHAND": "JH", "ODISHA": "OD", "ORISSA": "OD", "WESTBENGAL": "WB", "SIKKIM": "SK",
    "ARUNACHALPRADESH": "AR", "ASSAM": "AS", "MANIPUR": "MN", "MEGHALAYA": "ML", "MIZORAM": "MZ", "NAGALAND": "NL",
    "TRIPURA": "TR",
}
EXCLUDED = re.compile(r"ESSAR|AMNSIL|BALCO|RIL|RAILWAY|BULK|JAMNAGAR|NVVN", re.I)


def norm(name):
    return re.sub(r"[^A-Z]", "", str(name).upper().replace("&", "AND"))


def entity_id(name):
    """Map a raw State/UT name from any source to an entity id, or None (unknown or excluded entity)."""
    if name is None or EXCLUDED.search(str(name)):
        return None
    n = norm(name)
    if n in _ALIASES:
        return _ALIASES[n]
    n2 = n.replace("AND", "")
    for k, v in _ALIASES.items():
        if k.replace("AND", "") == n2:
            return v
    return None


def is_state_label(name):
    return entity_id(name) is not None


def table():
    import pandas as pd
    return pd.DataFrame([dict(entity=k, name=v[0], region=v[1], bid_area=v[2], pilot18=v[3], geo_state=v[4])
                         for k, v in ENTITIES.items()])
