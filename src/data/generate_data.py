"""Synthetic, semi-realistic coastal-hazard report generator (reproducible via --seed).

Usage: python -m src.data.generate_data --seed 42 --min-reports 2100
"""
import argparse
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.common import ALL_TYPES, DATA_DIR, HAZARD_TYPES, PLACES, SEED, haversine_km

T = {  # type -> language -> templates
 "tsunami": {
  "en": ["Sea water suddenly receded and then a huge wave hit {loc}", "Tsunami waves reported at {loc}, people running to higher ground", "Strong tremor felt, then the ocean surged into {loc} streets"],
  "hi": ["{loc} में समुद्र का पानी अचानक पीछे हटा और फिर विशाल लहर आई", "{loc} में सुनामी की लहरें, लोग ऊँची जगह भाग रहे हैं", "भूकंप के झटके के बाद {loc} की सड़कों पर समुद्र घुस आया"],
  "hx": ["{loc} mein samundar ka paani achanak peeche gaya aur phir badi lehar aayi", "{loc} pe tsunami jaisi lehrein, log unchi jagah bhaag rahe hain", "bhukamp ke baad {loc} ki sadko mein samundar ghus gaya"]},
 "storm_surge": {
  "en": ["Storm surge pushing seawater over the sea wall at {loc}", "Sea level rising fast with the cyclone, surge flooding {loc} waterfront", "Tide is abnormally high at {loc}, the surge has broken the embankment"],
  "hi": ["{loc} में तूफ़ान के साथ समुद्र का पानी समुद्री दीवार के ऊपर आ रहा है", "चक्रवात के कारण {loc} में ज्वार असामान्य रूप से ऊँचा, तटबंध टूट गया", "{loc} के तट पर तूफ़ानी लहरों का पानी बस्ती में घुस रहा है"],
  "hx": ["{loc} mein toofan ke saath samundar ka paani sea wall ke upar aa raha hai", "cyclone ki wajah se {loc} mein jwar bahut unchi hai, embankment toot gaya", "{loc} ke tat par toofani surge ka paani basti mein ghus raha hai"]},
 "flooding": {
  "en": ["Streets near the coast in {loc} are waterlogged after continuous rain", "Flood water entering houses at {loc}, roads are submerged", "Heavy rain and high tide together flooded the low-lying area of {loc}"],
  "hi": ["लगातार बारिश से {loc} के तटीय इलाके में सड़कों पर पानी भर गया", "{loc} में घरों में बाढ़ का पानी घुस रहा है, सड़कें डूबी हैं", "भारी बारिश और ऊँचे ज्वार से {loc} का निचला इलाका डूब गया"],
  "hx": ["lagatar baarish se {loc} ke tatiya ilake mein sadko par paani bhar gaya", "{loc} mein gharon mein baadh ka paani ghus raha hai, roads dub gayi hain", "bhaari baarish aur high tide se {loc} ka nichla ilaka dub gaya"]},
 "high_waves": {
  "en": ["Very high waves crashing over the jetty at {loc}, fishermen warned", "Rough sea at {loc}, waves over 4 metres hitting the shore", "Giant swells at {loc} beach, boats not able to leave harbour"],
  "hi": ["{loc} में जेटी के ऊपर बहुत ऊँची लहरें टकरा रही हैं, मछुआरों को चेतावनी", "{loc} में समुद्र उग्र है, चार मीटर से ऊँची लहरें किनारे से टकरा रही हैं", "{loc} बीच पर विशाल लहरें, नावें बंदरगाह से नहीं निकल पा रहीं"],
  "hx": ["{loc} mein jetty ke upar bahut unchi lehrein takra rahi hain, machuaron ko warning", "{loc} mein samundar bahut rough hai, chaar meter se unchi waves kinare se takra rahi hain", "{loc} beach pe badi badi lehrein, boats harbour se nahi nikal pa rahi"]},
 "coastal_erosion": {
  "en": ["The shoreline at {loc} is eroding fast, houses close to the edge are collapsing", "Beach sand washed away at {loc}, road along the coast is caving in", "Sea wall foundation exposed at {loc} after the waves took away the land"],
  "hi": ["{loc} में समुद्र तट तेज़ी से कट रहा है, किनारे के घर गिर रहे हैं", "{loc} में बीच की रेत बह गई, तट के किनारे की सड़क धँस रही है", "लहरों ने ज़मीन काट दी, {loc} में सी वॉल की नींव दिखने लगी"],
  "hx": ["{loc} mein samudra tat tezi se kat raha hai, kinare ke ghar gir rahe hain", "{loc} mein beach ki ret bah gayi, coast road dhans rahi hai", "lehron ne zameen kaat di, {loc} mein sea wall ki neev dikhne lagi"]},
 "oil_spill": {
  "en": ["Black oil slick washing ashore at {loc}, strong smell and dead fish", "Oil spill from a tanker near {loc}, the beach is covered in tar", "Thick oil patches on the water at {loc} harbour, fishing nets are ruined"],
  "hi": ["{loc} के किनारे काली तेल की परत आ रही है, तेज़ बदबू और मरी मछलियाँ", "{loc} के पास टैंकर से तेल रिसाव, पूरा बीच तारकोल से ढक गया", "{loc} बंदरगाह के पानी पर मोटी तेल की परत, मछली के जाल खराब"],
  "hx": ["{loc} ke kinare kaali oil ki parat aa rahi hai, tez badbu aur mari machliyan", "{loc} ke paas tanker se oil leak, poora beach tar se dhak gaya", "{loc} harbour ke paani pe moti oil ki parat, machhli ke jaal kharab"]},
 "cyclone_damage": {
  "en": ["Cyclone winds tore roofs off houses in {loc}, trees and poles are down", "Power lines down and fishing boats smashed after the cyclone at {loc}", "Cyclone hit {loc} overnight, many huts destroyed and roads blocked by fallen trees"],
  "hi": ["{loc} में चक्रवात की हवा से घरों की छतें उड़ गईं, पेड़ और खंभे गिरे", "{loc} में चक्रवात के बाद बिजली के तार टूटे और मछली पकड़ने की नावें टूट गईं", "रात में चक्रवात {loc} से टकराया, कई झोपड़ियाँ उजड़ीं और पेड़ गिरने से सड़कें बंद"],
  "hx": ["{loc} mein cyclone ki hawa se gharon ki chhatein ud gayi, ped aur khambe gire", "{loc} mein cyclone ke baad bijli ke taar toote aur machhli pakadne ki naavein tut gayi", "raat mein cyclone {loc} se takraya, kai jhopadiyan ujdi aur ped girne se sadkein band"]},
}
SEV = {"low": {"en": ["Minor so far, no damage", "Only a little water on the road"], "hi": ["अभी मामूली, नुकसान नहीं", "सड़क पर थोड़ा पानी"], "hx": ["abhi minor hai, nuksan nahi", "sadak pe thoda paani"]},
       "moderate": {"en": ["Several houses affected", "Traffic stopped, people moving out"], "hi": ["कई घर प्रभावित", "यातायात बंद, लोग निकल रहे हैं"], "hx": ["kai ghar prabhavit hain", "traffic band, log nikal rahe hain"]},
       "high": {"en": ["Very serious, people trapped, send help urgently", "Severe damage, rescue needed now"], "hi": ["बहुत गंभीर, लोग फँसे हैं, तुरंत मदद भेजें", "भारी नुकसान, अभी बचाव की ज़रूरत"], "hx": ["bahut serious hai, log fase hain, turant madad bhejo", "bhaari nuksan, abhi rescue chahiye"]}}
PREFIX = ["URGENT:", "Please help.", "#CoastalAlert", "Update:", "FYI", ""]
SHORT = {"tsunami": "huge wave!!", "storm_surge": "sea entering town", "flooding": "streets flooded", "high_waves": "very high waves",
         "coastal_erosion": "beach is vanishing", "oil_spill": "oil on the beach", "cyclone_damage": "trees down everywhere"}
WATER = ["tsunami", "storm_surge", "flooding", "high_waves"]
AMBIG = {"en": ["Sea water is coming into {loc}, roads are underwater", "Water rising fast near the shore at {loc}, people worried", "Big waves and water on the road at {loc}", "The sea looks very angry at {loc} and water is entering low areas"],
         "hi": ["{loc} में समुद्र का पानी सड़कों पर आ रहा है, लोग परेशान", "{loc} के किनारे पानी तेज़ी से बढ़ रहा है"],
         "hx": ["{loc} mein samundar ka paani sadko pe aa raha hai, log pareshan", "{loc} ke kinare paani tezi se badh raha hai, badi lehrein bhi hain"]}
HARDNEG = ["Water pipe burst at {loc}, water all over the road, municipality please fix", "Surfing contest at {loc} today, big waves and lots of fun",
           "Heavy waves of tourists at {loc} beach this weekend", "{loc} mein pipeline phat gayi, sadak pe paani bhar gaya, nagar nigam dhyan de",
           "Oil prices went up again, petrol pump at {loc} has a long queue", "Fishing boats back at {loc} harbour after the festival break",
           "Tar road repair going on near {loc} beach, expect delays", "Cyclone cricket club won the match at {loc} yesterday",
           "Erosion of trust in the government, said the speaker at {loc}", "Storm of applause at the {loc} music festival last night"]
LOCREPL = {"en": "our area", "hi": "यहाँ", "hx": "yahan"}

METAPHOR = ["Got a flood of emails after the long holiday", "A wave of nostalgia hit me watching old Bollywood songs", "Tsunami of homework this week, send help",
            "inbox mein emails ki baadh aa gayi hai", "cricket match ke baad stadium mein fans ki lehar thi", "ऑफिस में काम की सुनामी आ गई है",
            "The new phone launch caused a storm on social media", "Flood of orders at our shop today, so happy", "Wave of layoffs hits the tech industry"]
SARCASM = ["Oh great, another 'cyclone' of traffic on {loc} road, just what I needed", "Wow the 'tsunami' at {loc} beach is just my nephew splashing, lol",
           "haha {loc} mein 'baadh' aayi hai, bas meri chai ke cup mein", "Sure, 'massive flooding' at {loc}, it was one puddle outside my office",
           "{loc} mein 'toofan' aaya, mere bhai ka gussa tha bas"]
OLDNEWS = ["Remember the 2004 tsunami? Still gives me chills, {loc} was devastated back then", "Anniversary of the 2015 Chennai floods today, remembering those lost",
           "On this day in 1999 the Odisha super cyclone hit, never forget", "2004 ki sunami yaad hai? {loc} mein bahut tabahi hui thi",
           "Throwback: the 2018 Kerala floods near {loc}, 6 years ago, stay prepared"]
UNRELATED = ["Beautiful calm waves at {loc} beach this evening, perfect sunset", "Beach cleanup drive at {loc} this Sunday, everyone join us",
             "{loc} fish market is crowded today, prices are high", "Heavy traffic jam on {loc} highway near the toll", "Kids enjoying ice cream at {loc} promenade",
             "{loc} mein aaj bahut garmi hai, ghar se mat niklo", "Lighthouse at {loc} opens for tourists next week", "{loc} ka sunset aaj bahut sundar tha"]
RUMOUR = {"en": "Forwarded message: heard that {claim} at {loc}, not sure if it is true", "hi": "सुना है कि {loc} में {claim}, पता नहीं सच है या अफवाह",
          "hx": "WhatsApp pe forward aaya hai ki {loc} mein {claim}, unconfirmed hai"}
CLAIM = {"tsunami": {"en": "a tsunami is coming", "hi": "सुनामी आने वाली है", "hx": "tsunami aane wali hai"},
         "cyclone_damage": {"en": "the cyclone destroyed everything", "hi": "चक्रवात ने सब तबाह कर दिया", "hx": "cyclone ne sab tabah kar diya"},
         "oil_spill": {"en": "an oil spill happened", "hi": "तेल रिसाव हुआ है", "hx": "oil spill hua hai"}}


def typo(text, rng, n=2):
    ws = text.split()
    for _ in range(n):
        i = rng.randrange(len(ws))
        w = ws[i]
        if len(w) < 4:
            continue
        k = rng.randrange(1, len(w) - 1)
        op = rng.choice(["swap", "drop", "dup"])
        w = (w[:k] + w[k + 1] + w[k] + w[k + 2:] if op == "swap" and k + 2 <= len(w) - 0 and k + 1 < len(w)
             else w[:k] + w[k + 1:] if op == "drop" else w[:k] + w[k] + w[k:])
        ws[i] = w
    return " ".join(ws)


def jitter(lat, lon, km, rng):
    r = abs(rng.gauss(0, km))
    th = rng.uniform(0, 2 * math.pi)
    return lat + r * math.cos(th) / 111.0, lon + r * math.sin(th) / (111.0 * math.cos(math.radians(lat)))


def gen(seed=SEED, min_reports=2100):
    rng = random.Random(seed)
    np.random.seed(seed)
    t0 = datetime(2025, 6, 1)
    names = list(PLACES)
    rows, counter = [], [0]

    def new_id():
        counter[0] += 1
        return f"R{counter[0]:05d}"

    def row(**kw):
        base = dict(report_id=new_id(), text="", language="en", latitude=np.nan, longitude=np.nan, location_text="",
                    timestamp="", incident_type="not_hazard", severity="none", source="social", media_flag=0,
                    confidence=0.8, true_incident_id="NONE", is_duplicate=0, duplicate_of="", dup_group="",
                    edge_case="normal", is_hazard=0)
        base.update(kw)
        return base

    def pick_source():
        return rng.choices(["social", "app", "sms"], [0.5, 0.3, 0.2])[0]

    def media(src):
        return int(rng.random() < {"app": 0.5, "social": 0.3, "sms": 0.0}[src])

    # ---- incidents -------------------------------------------------------
    incidents, k = [], 0
    while True:
        place = rng.choice(names)
        typ = rng.choice(HAZARD_TYPES)
        start = t0 + timedelta(hours=rng.uniform(0, 24 * 120))
        n = int(np.clip(rng.lognormvariate(2.0, 0.6), 3, 40))
        sev = rng.choices(["low", "moderate", "high"], [0.35, 0.4, 0.25])[0]
        k += 1
        incidents.append(dict(id=f"INC{k:04d}", place=place, type=typ, start=start, n=n, sev=sev, spread=rng.choice([1.0, 1.5, 3.0, 5.0])))
        if rng.random() < 0.2:  # sibling incident, same place, different type, soon after
            k += 1
            incidents.append(dict(id=f"INC{k:04d}", place=place, type=rng.choice([t for t in HAZARD_TYPES if t != typ]),
                                  start=start + timedelta(hours=rng.uniform(12, 48)), n=max(3, n // 2), sev=sev, spread=1.5))
        if k >= 120 and sum(i["n"] for i in incidents) * 1.2 >= min_reports * 0.8:
            break

    for inc in incidents:
        lat0, lon0, _ = PLACES[inc["place"]]
        made = []
        for _ in range(inc["n"]):
            lang = rng.choices(["en", "hx", "hi"], [0.5, 0.3, 0.2])[0]
            src = pick_source()
            off = rng.expovariate(1 / 0.7) if rng.random() < 0.6 else rng.expovariate(1 / 6.0)
            ts = inc["start"] + timedelta(hours=off)
            lat, lon = jitter(lat0, lon0, inc["spread"], rng)
            tag, loc_in_text, loc_text = "normal", inc["place"], inc["place"]
            u = rng.random()
            text = rng.choice(T[inc["type"]][lang])
            if u < 0.04:
                tag = "missing_location"
                text, loc_in_text, loc_text, lat, lon = text.replace("{loc}", LOCREPL[lang]), "", "", np.nan, np.nan
            elif u < 0.08:
                tag = "missing_time"
            elif u < 0.11:
                tag, lang = "short_text", "en"
                text = SHORT[inc["type"]]
            elif u < 0.17:
                tag = "typo"
            elif u < 0.19:
                tag = "conflicting_location"
                far = [p for p in names if float(haversine_km(lat0, lon0, PLACES[p][0], PLACES[p][1])) > 400]
                loc_in_text = loc_text = rng.choice(far)
            elif lang == "hx":
                tag = "code_mixed"
            if inc["type"] in WATER and tag in ("normal", "code_mixed") and rng.random() < 0.15:
                tag, text = "ambiguous", rng.choice(AMBIG[lang])
            text = text.replace("{loc}", loc_in_text)
            if tag != "short_text":
                if rng.random() < 0.8:
                    text += ". " + rng.choice(SEV[inc["sev"]][lang])
                pre = rng.choice(PREFIX)
                text = (pre + " " + text).strip()
            if tag == "typo":
                text = typo(text, rng)
            r = row(text=text, language=lang, latitude=round(lat, 5) if not math.isnan(lat) else np.nan,
                    longitude=round(lon, 5) if not math.isnan(lon) else np.nan, location_text=loc_text,
                    timestamp="" if tag == "missing_time" else ts.isoformat(timespec="seconds"),
                    incident_type=inc["type"], severity=inc["sev"], source=src, media_flag=media(src),
                    confidence=round(rng.uniform(0.6, 0.95) if tag == "normal" else rng.uniform(0.35, 0.7), 2),
                    true_incident_id=inc["id"], edge_case=tag, is_hazard=1)
            rows.append(r)
            made.append((r, ts, lat, lon))
            if rng.random() < 0.15:  # re-post / forward of this report
                o = made[-1][0]
                if o["timestamp"] and o["text"]:
                    d = dict(o)
                    d.update(report_id=new_id(), is_duplicate=1, duplicate_of=o["report_id"], dup_group=o["report_id"],
                             timestamp=(ts + timedelta(minutes=rng.uniform(1, 45))).isoformat(timespec="seconds"),
                             text=o["text"] if rng.random() < 0.6 else typo(o["text"], rng, 1),
                             source=pick_source(), edge_case=o["edge_case"])
                    if not math.isnan(lat):
                        d["latitude"], d["longitude"] = [round(x, 5) for x in jitter(lat, lon, 0.2, rng)]
                    o["dup_group"] = o["report_id"]
                    rows.append(d)

    # ---- noise: metaphors, sarcasm, old news, unrelated, rumours --------
    def noise(templates, tag, n, langs=None):
        for _ in range(n):
            place = rng.choice(names)
            t = rng.choice(templates)
            lang = "hi" if any("\u0900" <= c <= "\u097f" for c in t) else ("hx" if any(w in t for w in ["mein", "aaj", "ki ", "ka "]) else "en")
            lat, lon = jitter(*PLACES[place][:2], 3, rng)
            ts = t0 + timedelta(hours=rng.uniform(0, 24 * 125))
            rows.append(row(text=t.replace("{loc}", place), language=lang, latitude=round(lat, 5), longitude=round(lon, 5),
                            location_text=place, timestamp=ts.isoformat(timespec="seconds"), source=pick_source(),
                            media_flag=0, confidence=round(rng.uniform(0.3, 0.6), 2), edge_case=tag))
    noise(METAPHOR, "metaphor", 120)
    noise(SARCASM, "sarcasm", 80)
    noise(OLDNEWS, "old_news", 80)
    noise(UNRELATED, "unrelated", 180)
    noise(HARDNEG, "hard_negative", 90)
    for _ in range(130):
        typ = rng.choice(list(CLAIM))
        lang = rng.choices(["en", "hx", "hi"], [0.5, 0.3, 0.2])[0]
        place = rng.choice(names)
        lat, lon = jitter(*PLACES[place][:2], 3, rng)
        ts = t0 + timedelta(hours=rng.uniform(0, 24 * 125))
        rows.append(row(text=RUMOUR[lang].format(loc=place, claim=CLAIM[typ][lang]), language=lang, latitude=round(lat, 5),
                        longitude=round(lon, 5), location_text=place, timestamp=ts.isoformat(timespec="seconds"),
                        incident_type=typ, severity="low", source=pick_source(), confidence=round(rng.uniform(0.1, 0.4), 2),
                        edge_case="rumour", is_hazard=1))
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    # re-number ids after shuffle, remapping references
    mp = {o: f"R{i + 1:05d}" for i, o in enumerate(df.report_id)}
    for c in ["report_id", "duplicate_of", "dup_group"]:
        df[c] = df[c].map(lambda x: mp.get(x, ""))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-reports", type=int, default=2100)
    ap.add_argument("--out", default=str(DATA_DIR / "raw" / "reports_raw.csv"))
    a = ap.parse_args()
    df = gen(a.seed, a.min_reports)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} reports -> {a.out}")
    print(df.incident_type.value_counts().to_string())


if __name__ == "__main__":
    main()
