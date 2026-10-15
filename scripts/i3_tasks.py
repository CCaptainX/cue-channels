"""Experiment I3 tasks: one-pass-retrievable facts (pre-registered in experiments/I3_memorizable_tasks/)."""

from __future__ import annotations

import random

LETTERS = "ABCDE"
TASKS = ["country_capitals", "us_state_capitals", "element_symbols", "times_tables", "unit_conversions"]
SEEDS = {"country_capitals": 401, "us_state_capitals": 402, "element_symbols": 403,
         "times_tables": 404, "unit_conversions": 405}
DESC = {
    "country_capitals": "Answer the geography question by choosing the correct option.",
    "us_state_capitals": "Answer the geography question by choosing the correct option.",
    "element_symbols": "Answer the chemistry question by choosing the correct option.",
    "times_tables": "Answer the arithmetic question by choosing the correct option.",
    "unit_conversions": "Answer the unit-conversion question by choosing the correct option.",
}

# Unambiguous capitals only: countries with contested, multiple, or moving capitals and capitals containing the
# country's name are excluded (see preregistration).
COUNTRY_CAPITALS = {
    "Afghanistan": "Kabul", "Albania": "Tirana", "Algeria": "Algiers", "Angola": "Luanda",
    "Argentina": "Buenos Aires", "Armenia": "Yerevan", "Australia": "Canberra", "Austria": "Vienna",
    "Azerbaijan": "Baku", "the Bahamas": "Nassau", "Bahrain": "Manama", "Bangladesh": "Dhaka",
    "Barbados": "Bridgetown", "Belarus": "Minsk", "Belgium": "Brussels", "Belize": "Belmopan",
    "Bhutan": "Thimphu", "Bosnia and Herzegovina": "Sarajevo", "Botswana": "Gaborone", "Brazil": "Brasília",
    "Brunei": "Bandar Seri Begawan", "Bulgaria": "Sofia", "Burkina Faso": "Ouagadougou",
    "Cambodia": "Phnom Penh", "Cameroon": "Yaoundé", "Canada": "Ottawa", "Cape Verde": "Praia",
    "the Central African Republic": "Bangui", "Chad": "N'Djamena", "Chile": "Santiago", "China": "Beijing",
    "Colombia": "Bogotá", "Comoros": "Moroni", "Costa Rica": "San José", "Croatia": "Zagreb", "Cuba": "Havana",
    "Cyprus": "Nicosia", "the Czech Republic": "Prague", "Denmark": "Copenhagen",
    "the Dominican Republic": "Santo Domingo", "Ecuador": "Quito", "Egypt": "Cairo", "Eritrea": "Asmara",
    "Estonia": "Tallinn", "Ethiopia": "Addis Ababa", "Fiji": "Suva", "Finland": "Helsinki", "France": "Paris",
    "Gabon": "Libreville", "the Gambia": "Banjul", "Georgia (the country)": "Tbilisi", "Germany": "Berlin",
    "Ghana": "Accra", "Greece": "Athens", "Grenada": "St. George's", "Guinea": "Conakry", "Guyana": "Georgetown",
    "Haiti": "Port-au-Prince", "Honduras": "Tegucigalpa", "Hungary": "Budapest", "Iceland": "Reykjavík",
    "India": "New Delhi", "Iran": "Tehran", "Iraq": "Baghdad", "Ireland": "Dublin", "Italy": "Rome",
    "Jamaica": "Kingston", "Japan": "Tokyo", "Jordan": "Amman", "Kazakhstan": "Astana", "Kenya": "Nairobi",
    "Kyrgyzstan": "Bishkek", "Laos": "Vientiane", "Latvia": "Riga", "Lebanon": "Beirut", "Lesotho": "Maseru",
    "Liberia": "Monrovia", "Libya": "Tripoli", "Liechtenstein": "Vaduz", "Lithuania": "Vilnius",
    "Madagascar": "Antananarivo", "Malawi": "Lilongwe", "the Maldives": "Malé", "Mali": "Bamako",
    "Malta": "Valletta", "Mauritania": "Nouakchott", "Mauritius": "Port Louis", "Moldova": "Chișinău",
    "Mongolia": "Ulaanbaatar", "Montenegro": "Podgorica", "Morocco": "Rabat", "Mozambique": "Maputo",
    "Myanmar": "Naypyidaw", "Namibia": "Windhoek", "Nepal": "Kathmandu", "New Zealand": "Wellington",
    "Nicaragua": "Managua", "Niger": "Niamey", "Nigeria": "Abuja", "North Korea": "Pyongyang",
    "North Macedonia": "Skopje", "Norway": "Oslo", "Oman": "Muscat", "Pakistan": "Islamabad",
    "Papua New Guinea": "Port Moresby", "Paraguay": "Asunción", "Peru": "Lima", "the Philippines": "Manila",
    "Poland": "Warsaw", "Portugal": "Lisbon", "Qatar": "Doha", "Romania": "Bucharest", "Russia": "Moscow",
    "Rwanda": "Kigali", "Samoa": "Apia", "Saudi Arabia": "Riyadh", "Senegal": "Dakar", "Serbia": "Belgrade",
    "Seychelles": "Victoria", "Sierra Leone": "Freetown", "Slovakia": "Bratislava", "Slovenia": "Ljubljana",
    "the Solomon Islands": "Honiara", "Somalia": "Mogadishu", "South Korea": "Seoul", "South Sudan": "Juba",
    "Spain": "Madrid", "Sudan": "Khartoum", "Suriname": "Paramaribo", "Sweden": "Stockholm",
    "Switzerland": "Bern", "Syria": "Damascus", "Tajikistan": "Dushanbe", "Thailand": "Bangkok",
    "Timor-Leste": "Dili", "Togo": "Lomé", "Tonga": "Nukuʻalofa", "Trinidad and Tobago": "Port of Spain",
    "Tunisia": "Tunis", "Turkey": "Ankara", "Turkmenistan": "Ashgabat", "Uganda": "Kampala", "Ukraine": "Kyiv",
    "the United Arab Emirates": "Abu Dhabi", "the United Kingdom": "London", "the United States": "Washington, D.C.",
    "Uruguay": "Montevideo", "Uzbekistan": "Tashkent", "Vanuatu": "Port Vila", "Venezuela": "Caracas",
    "Vietnam": "Hanoi", "Zambia": "Lusaka", "Zimbabwe": "Harare", "the Republic of the Congo": "Brazzaville",
    "the Democratic Republic of the Congo": "Kinshasa",
}

US_STATE_CAPITALS = {
    "Alabama": "Montgomery", "Alaska": "Juneau", "Arizona": "Phoenix", "Arkansas": "Little Rock",
    "California": "Sacramento", "Colorado": "Denver", "Connecticut": "Hartford", "Delaware": "Dover",
    "Florida": "Tallahassee", "Georgia": "Atlanta", "Hawaii": "Honolulu", "Idaho": "Boise",
    "Illinois": "Springfield", "Indiana": "Indianapolis", "Iowa": "Des Moines", "Kansas": "Topeka",
    "Kentucky": "Frankfort", "Louisiana": "Baton Rouge", "Maine": "Augusta", "Maryland": "Annapolis",
    "Massachusetts": "Boston", "Michigan": "Lansing", "Minnesota": "Saint Paul", "Mississippi": "Jackson",
    "Missouri": "Jefferson City", "Montana": "Helena", "Nebraska": "Lincoln", "Nevada": "Carson City",
    "New Hampshire": "Concord", "New Jersey": "Trenton", "New Mexico": "Santa Fe", "New York": "Albany",
    "North Carolina": "Raleigh", "North Dakota": "Bismarck", "Ohio": "Columbus", "Oklahoma": "Oklahoma City",
    "Oregon": "Salem", "Pennsylvania": "Harrisburg", "Rhode Island": "Providence", "South Carolina": "Columbia",
    "South Dakota": "Pierre", "Tennessee": "Nashville", "Texas": "Austin", "Utah": "Salt Lake City",
    "Vermont": "Montpelier", "Virginia": "Richmond", "Washington": "Olympia", "West Virginia": "Charleston",
    "Wisconsin": "Madison", "Wyoming": "Cheyenne",
}

ELEMENT_SYMBOLS = {
    "hydrogen": "H", "helium": "He", "lithium": "Li", "beryllium": "Be", "boron": "B", "carbon": "C",
    "nitrogen": "N", "oxygen": "O", "fluorine": "F", "neon": "Ne", "sodium": "Na", "magnesium": "Mg",
    "aluminum": "Al", "silicon": "Si", "phosphorus": "P", "sulfur": "S", "chlorine": "Cl", "argon": "Ar",
    "potassium": "K", "calcium": "Ca", "titanium": "Ti", "chromium": "Cr", "manganese": "Mn", "iron": "Fe",
    "cobalt": "Co", "nickel": "Ni", "copper": "Cu", "zinc": "Zn", "gallium": "Ga", "germanium": "Ge",
    "arsenic": "As", "selenium": "Se", "bromine": "Br", "krypton": "Kr", "rubidium": "Rb", "strontium": "Sr",
    "silver": "Ag", "tin": "Sn", "antimony": "Sb", "iodine": "I", "xenon": "Xe", "cesium": "Cs",
    "barium": "Ba", "tungsten": "W", "platinum": "Pt", "gold": "Au", "mercury": "Hg", "lead": "Pb",
    "uranium": "U", "radium": "Ra",
}

UNIT_TEMPLATES = [("feet", "inches", 12), ("hours", "minutes", 60), ("minutes", "seconds", 60),
                  ("kilometers", "meters", 1000), ("kilograms", "grams", 1000), ("weeks", "days", 7),
                  ("days", "hours", 24), ("yards", "feet", 3), ("meters", "centimeters", 100),
                  ("gallons", "quarts", 4)]


def _facts(task: str) -> list[dict]:
    """Every fact: question text, answer string, worked step, and a distractor sampler key."""
    if task in ("country_capitals", "us_state_capitals"):
        table = COUNTRY_CAPITALS if task == "country_capitals" else US_STATE_CAPITALS
        noun = "" if task == "country_capitals" else " the U.S. state of"
        return [{"q": f"What is the capital of{noun} {k}?", "a": v, "steps": f"The capital of {k} is {v}.",
                 "pool": "values"} for k, v in table.items()]
    if task == "element_symbols":
        return [{"q": f"What is the chemical symbol for {k}?", "a": v, "steps": f"The symbol for {k} is {v}.",
                 "pool": "values"} for k, v in ELEMENT_SYMBOLS.items()]
    if task == "times_tables":
        return [{"q": f"What is {a} × {b}?", "a": str(a * b), "steps": f"{a} × {b} = {a * b}.", "pool": "times",
                 "g": f"{min(a, b)}x{max(a, b)}"} for a in range(2, 13) for b in range(2, 13)]
    if task == "unit_conversions":
        out = [{"q": f"How many {small} are in {n} {big}?", "a": str(n * f),
                "steps": f"1 {big[:-1] if big.endswith('s') else big} = {f} {small}, so {n} × {f} = {n * f}.",
                "pool": ("unit", t)} for t, (big, small, f) in enumerate(UNIT_TEMPLATES) for n in range(2, 13)]
        out += [{"q": f"What is {c} °C in °F?", "a": str(c * 9 // 5 + 32),
                 "steps": f"F = C × 9/5 + 32 = {c} × 9/5 + 32 = {c * 9 // 5 + 32}.", "pool": ("unit", "c2f")}
                for c in range(-20, 101, 5)]
        return out
    raise ValueError(task)


def _distractors(task: str, fact: dict, facts: list[dict], rng: random.Random) -> list[str]:
    if fact["pool"] == "values":
        cand = [f["a"] for f in facts if f["a"] != fact["a"]]
        return rng.sample(sorted(set(cand)), 4)
    if fact["pool"] == "times":  # other a'×b' products, drawn pair-wise (same law as the correct answer)
        out: list[str] = []
        while len(out) < 4:
            v = str(rng.randint(2, 12) * rng.randint(2, 12))
            if v != fact["a"] and v not in out:
                out.append(v)
        return out
    same = sorted({f["a"] for f in facts if f["pool"] == fact["pool"] and f["a"] != fact["a"]}, key=int)
    return rng.sample(same, 4)


def _item(task, fact, facts, rng, force=None):
    vals = [fact["a"]] + _distractors(task, fact, facts, rng)
    rng.shuffle(vals)
    if force is not None:
        vals.remove(fact["a"])
        vals.insert(LETTERS.index(force), fact["a"])
    opts = "\n".join(f"({ltr}) {v}" for ltr, v in zip(LETTERS, vals, strict=True))
    return {"input": f"{fact['q']}\nOptions:\n{opts}", "target": LETTERS[vals.index(fact["a"])],
            "letters": LETTERS, "question": fact["q"], "option_values": vals, "steps": fact["steps"]}


def exemplars(task: str) -> list[dict]:
    facts = _facts(task)
    rng = random.Random(SEEDS[task] + 1000)
    chosen = rng.sample(facts, 3)
    return [_item(task, f, facts, rng, ltr) for f, ltr in zip(chosen, "BAC", strict=True)]


def items(task: str, n: int, seed_offset: int = 0) -> list[dict]:
    """Cycle through the shuffled fact pool until n items; each repeat gets fresh distractors and option order.
    'group' = fact id, used for grouped cross-validation."""
    facts = _facts(task)
    ex_q = {e["question"] for e in exemplars(task)}
    gids = {g: i for i, g in enumerate(dict.fromkeys(f.get("g", f["q"]) for f in facts))}
    pool = [(gids[f.get("g", f["q"])], f) for f in facts if f["q"] not in ex_q]
    rng = random.Random(SEEDS[task] + seed_offset)
    order = []
    while len(order) < n:
        block = pool[:]
        rng.shuffle(block)
        order.extend(block)
    out = []
    for k, (gid, f) in enumerate(order[:n]):
        it = _item(task, f, facts, rng)
        it["idx"], it["group"] = k, gid
        out.append(it)
    return out


def prompts(task: str) -> tuple[str, str]:
    ex = exemplars(task)
    cot = [f"Q: {e['input']}\nA: Let's think step by step.\n{e['steps']}\nSo the answer is ({e['target']})."
           for e in ex]
    ao = [f"Q: {e['input']}\nA: ({e['target']})" for e in ex]
    return DESC[task] + "\n\n" + "\n\n".join(cot), DESC[task] + "\n\n" + "\n\n".join(ao)
