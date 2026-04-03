import os
import time
import csv
import zipfile
import requests
from pathlib import Path
from collections import defaultdict

FREESOUND_API_KEY = "YOUR_FREESOUND_API_KEY"   
SAMPLE_RATE       = 22050
CLIP_DURATION     = 5.0
MAX_PER_SPECIES   = 80
SLEEP_BETWEEN     = 0.5
RAW_AUDIO_DIR     = "raw_audio"
LOGS_DIR          = "download_logs"

os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,      exist_ok=True)

SPECIES_CATALOG = {

    "birds": {
        "Common Cuckoo":         "Cuculus canorus",
        "Barn Owl":              "Tyto alba",
        "Common Nightingale":    "Luscinia megarhynchos",
        "House Sparrow":         "Passer domesticus",
        "Eurasian Blackbird":    "Turdus merula",
        "Common Chaffinch":      "Fringilla coelebs",
        "Great Tit":             "Parus major",
        "European Robin":        "Erithacus rubecula",
        "Peregrine Falcon":      "Falco peregrinus",
        "Common Swift":          "Apus apus",
        "Tawny Owl":             "Strix aluco",
        "Common Kingfisher":     "Alcedo atthis",
        "Blue Jay":              "Cyanocitta cristata",
        "American Robin":        "Turdus migratorius",
        "Northern Cardinal":     "Cardinalis cardinalis",
        "Wood Thrush":           "Hylocichla mustelina",
        "Common Loon":           "Gavia immer",
        "Bald Eagle":            "Haliaeetus leucocephalus",
        "Pileated Woodpecker":   "Dryocopus pileatus",
        "Whip-poor-will":        "Antrostomus vociferus",
    },

    "frogs": {
        "Common Frog":           "Rana temporaria",
        "American Bullfrog":     "Lithobates catesbeianus",
        "Spring Peeper":         "Pseudacris crucifer",
        "Green Tree Frog":       "Hyla cinerea",
        "Gray Tree Frog":        "Hyla versicolor",
        "Pacific Tree Frog":     "Pseudacris regilla",
        "Wood Frog":             "Lithobates sylvaticus",
        "Pickerel Frog":         "Lithobates palustris",
        "European Tree Frog":    "Hyla arborea",
        "Natterjack Toad":       "Epidalea calamita",
        "Common Toad":           "Bufo bufo",
        "Fire-bellied Toad":     "Bombina bombina",
        "Midwife Toad":          "Alytes obstetricans",
        "Southern Leopard Frog": "Lithobates sphenocephalus",
        "Cope's Gray Tree Frog": "Hyla chrysoscelis",
    },

    "insects": {
        "Honeybee":                  "Apis mellifera",
        "Field Cricket":             "Gryllus campestris",
        "House Cricket":             "Acheta domesticus",
        "Common Grasshopper":        "Chorthippus brunneus",
        "Cicada":                    "Magicicada septendecim",
        "Bumblebee":                 "Bombus terrestris",
        "Mole Cricket":              "Gryllotalpa gryllotalpa",
        "Great Green Bush Cricket":  "Tettigonia viridissima",
        "Common Wasp":               "Vespula vulgaris",
        "Mosquito":                  "Culex pipiens",
        "Death's Head Hawkmoth":     "Acherontia atropos",
        "Stag Beetle":               "Lucanus cervus",
    },

    "mammals": {
        "Gray Wolf":                 "Canis lupus",
        "Red Fox":                   "Vulpes vulpes",
        "Common Pipistrelle Bat":    "Pipistrellus pipistrellus",
        "Big Brown Bat":             "Eptesicus fuscus",
        "European Hedgehog":         "Erinaceus europaeus",
        "Red Deer":                  "Cervus elaphus",
        "Wild Boar":                 "Sus scrofa",
        "Brown Bear":                "Ursus arctos",
        "African Lion":              "Panthera leo",
        "Chimpanzee":                "Pan troglodytes",
        "Common Raccoon":            "Procyon lotor",
        "Eurasian Lynx":             "Lynx lynx",
        "Common Dolphin":            "Delphinus delphis",
    },
}

def save_log(log_rows, filename):
    if not log_rows:
        return
    log_path = Path(LOGS_DIR) / filename
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)


def count_files(folder):
    p = Path(folder)
    if not p.exists():
        return 0
    return len(list(p.glob("*.*")))


XC_API = "https://xeno-canto.org/api/2/recordings"

def xc_search(scientific_name, max_results=MAX_PER_SPECIES):
    results, page = [], 1
    quality_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "": 5}

    while len(results) < max_results:
        try:
            r = requests.get(
                XC_API,
                params={"query": scientific_name, "page": page},
                timeout=15
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    ⚠️  XC error p{page}: {e}")
            break

        recs = data.get("recordings", [])
        if not recs:
            break
        results.extend(recs)
        if page >= int(data.get("numPages", 1)):
            break
        page += 1
        time.sleep(SLEEP_BETWEEN)

    results.sort(key=lambda r: quality_order.get(r.get("q", ""), 5))
    return results[:max_results]


def download_xc_species(common_name, scientific_name, category):
    folder = Path(RAW_AUDIO_DIR) / category / common_name.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)

    existing = count_files(folder)
    if existing >= MAX_PER_SPECIES:
        print(f"  ⏭  {common_name}: {existing} files — skipping")
        return existing

    recs = xc_search(scientific_name)
    print(f"  🔍 {common_name}: {len(recs)} found on Xeno-Canto")
    downloaded, log_rows = 0, []

    for i, rec in enumerate(recs):
        fname = folder / f"{i:04d}_xc{rec['id']}.mp3"
        if fname.exists():
            downloaded += 1
            continue

        url = rec.get("file", "")
        if url.startswith("//"):
            url = "https:" + url
        if not url:
            continue

        try:
            resp = requests.get(url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            downloaded += 1
            log_rows.append({
                "id": rec["id"], "common": common_name,
                "sci": scientific_name, "quality": rec.get("q",""),
                "country": rec.get("cnt",""), "file": str(fname)
            })
            print(f"    ✅ [{downloaded}] {rec.get('en','?')} Q:{rec.get('q','?')}")
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            print(f"    ❌ {rec['id']}: {e}")

    save_log(log_rows, f"xc_{common_name.replace(' ','_')}.csv")
    return downloaded


FS_SEARCH = "https://freesound.org/apiv2/search/text/"

def fs_search(query, max_results=50, min_dur=2.0):
    if not FREESOUND_API_KEY or FREESOUND_API_KEY == "YOUR_FREESOUND_API_KEY":
        return []

    headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}
    results, page = {}, 1

    while len(results) < max_results:
        try:
            r = requests.get(FS_SEARCH, headers=headers, params={
                "query": query,
                "fields": "id,name,tags,duration,previews,license",
                "filter": f"duration:[{min_dur} TO 300]",
                "sort": "rating_desc",
                "page_size": 50,
                "page": page,
            }, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"    ⚠️  FreeSound error: {e}")
            break

        for s in data.get("results", []):
            results[s["id"]] = s
        if not data.get("next"):
            break
        page += 1
        time.sleep(SLEEP_BETWEEN)

    return list(results.values())[:max_results]


def download_fs_species(common_name, scientific_name, category):
    folder = Path(RAW_AUDIO_DIR) / category / common_name.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)

    existing = count_files(folder)
    if existing >= MAX_PER_SPECIES:
        print(f"  ⏭  {common_name}: {existing} files — skipping")
        return existing

    headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}
    queries = [f"{common_name} sound", f"{common_name} call",
               f"{scientific_name}", f"{common_name} animal call"]

    all_sounds = {}
    for q in queries:
        for s in fs_search(q, max_results=40):
            all_sounds[s["id"]] = s

    sounds = list(all_sounds.values())[:MAX_PER_SPECIES]
    print(f"  🔍 {common_name}: {len(sounds)} found on FreeSound")
    downloaded, log_rows = 0, []

    for i, sound in enumerate(sounds):
        url = sound["previews"].get("preview-hq-mp3") or \
              sound["previews"].get("preview-lq-mp3", "")
        if not url:
            continue

        fname = folder / f"{i:04d}_fs{sound['id']}.mp3"
        if fname.exists():
            downloaded += 1
            continue

        try:
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            resp.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            downloaded += 1
            log_rows.append({
                "id": sound["id"], "name": sound["name"],
                "common": common_name, "sci": scientific_name,
                "dur": sound.get("duration",""), "file": str(fname)
            })
            print(f"    ✅ [{downloaded}] {sound['name'][:45]}")
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            print(f"    ❌ sound {sound['id']}: {e}")

    save_log(log_rows, f"fs_{common_name.replace(' ','_')}.csv")
    return downloaded


INAT_TAX = "https://api.inaturalist.org/v1/taxa"
INAT_OBS = "https://api.inaturalist.org/v1/observations"

def inat_taxon_id(scientific_name):
    try:
        r = requests.get(INAT_TAX, params={"q": scientific_name, "per_page": 1}, timeout=10)
        r.raise_for_status()
        res = r.json().get("results", [])
        return res[0]["id"] if res else None
    except Exception:
        return None


def download_inat_species(common_name, scientific_name, category, need=30):
    folder = Path(RAW_AUDIO_DIR) / category / common_name.replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)

    taxon_id = inat_taxon_id(scientific_name)
    if not taxon_id:
        print(f"  ❌ {common_name}: taxon not found on iNaturalist")
        return 0

    page, downloaded, log_rows = 1, 0, []
    existing_count = count_files(folder)

    while downloaded < need:
        try:
            r = requests.get(INAT_OBS, params={
                "taxon_id": taxon_id, "sounds": "true",
                "quality_grade": "research",
                "per_page": 50, "page": page,
            }, timeout=15)
            r.raise_for_status()
            obs_list = r.json().get("results", [])
        except Exception as e:
            print(f"    ⚠️  iNat error: {e}")
            break

        if not obs_list:
            break

        for obs in obs_list:
            for sound in obs.get("sounds", []):
                url = sound.get("file_url", "")
                if not url:
                    continue
                ext = ".ogg" if "ogg" in sound.get("file_content_type","") else ".mp3"
                fname = folder / f"{existing_count+downloaded:04d}_inat{obs['id']}{ext}"
                if fname.exists():
                    downloaded += 1
                    continue
                try:
                    resp = requests.get(url, timeout=30, stream=True)
                    resp.raise_for_status()
                    with open(fname, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                    downloaded += 1
                    log_rows.append({
                        "obs_id": obs["id"], "common": common_name,
                        "sci": scientific_name,
                        "location": obs.get("place_guess",""),
                        "date": obs.get("observed_on",""),
                        "file": str(fname)
                    })
                    print(f"    ✅ iNat [{downloaded}/{need}] obs#{obs['id']}")
                    time.sleep(SLEEP_BETWEEN)
                    if downloaded >= need:
                        break
                except Exception as e:
                    print(f"    ❌ iNat obs {obs['id']}: {e}")
            if downloaded >= need:
                break

        if len(obs_list) < 50:
            break
        page += 1
        time.sleep(SLEEP_BETWEEN)

    save_log(log_rows, f"inat_{common_name.replace(' ','_')}.csv")
    return downloaded

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ESC50_ANIMAL_MAP = {
    "dog":      ("dog",      "mammals"),
    "rooster":  ("rooster",  "birds"),
    "pig":      ("pig",      "mammals"),
    "cow":      ("cow",      "mammals"),
    "frog":     ("frog",     "frogs"),
    "cat":      ("cat",      "mammals"),
    "hen":      ("hen",      "birds"),
    "insects":  ("insects",  "insects"),
    "sheep":    ("sheep",    "mammals"),
    "crow":     ("crow",     "birds"),
}

def download_esc50():
    zip_path = Path("kaggle_downloads/ESC-50.zip")
    zip_path.parent.mkdir(exist_ok=True)

    if zip_path.exists():
        print("  ⏭  ESC-50 zip already exists, skipping download")
    else:
        print("  📥 Downloading ESC-50 from GitHub...")
        try:
            r = requests.get(ESC50_URL, timeout=120, stream=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            print("  ✅ ESC-50 downloaded")
        except Exception as e:
            print(f"  ❌ ESC-50 download failed: {e}")
            return

    extract_dir = Path("kaggle_downloads/ESC-50")
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("  ✅ ESC-50 extracted")

    audio_src = extract_dir / "ESC-50-master" / "audio"
    meta_src  = extract_dir / "ESC-50-master" / "meta" / "esc50.csv"

    if not meta_src.exists():
        print("  ❌ ESC-50 metadata not found")
        return

    import csv as _csv
    with open(meta_src, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            cat_label = row.get("category","").strip()
            if cat_label not in ESC50_ANIMAL_MAP:
                continue
            common_name, category = ESC50_ANIMAL_MAP[cat_label]
            dst_folder = Path(RAW_AUDIO_DIR) / category / common_name
            dst_folder.mkdir(parents=True, exist_ok=True)
            src_file = audio_src / row["filename"]
            dst_file = dst_folder / f"esc50_{row['filename']}"
            if src_file.exists() and not dst_file.exists():
                import shutil
                shutil.copy2(src_file, dst_file)

    print("  ✅ ESC-50 animal sounds copied to raw_audio/")

def print_summary():
    print("\n" + "="*65)
    print("  📊  EchoSense Dataset Summary")
    print("="*65)
    print(f"  {'Category':<12} {'Species':<32} {'Files':>6}")
    print("  " + "-"*60)

    grand_total = 0
    cat_totals  = defaultdict(int)

    for cat in ["birds", "frogs", "insects", "mammals"]:
        for name in SPECIES_CATALOG[cat]:
            folder = Path(RAW_AUDIO_DIR) / cat / name.replace(" ", "_")
            n = count_files(folder)
            cat_totals[cat] += n
            grand_total     += n
            icon = "✅" if n >= 30 else ("⚠️ " if n > 0 else "❌")
            print(f"  {icon} {cat:<10} {name:<32} {n:>6}")

    print("  " + "-"*60)
    for cat, total in cat_totals.items():
        print(f"  {cat.upper():<22} subtotal: {total:>6} files")
    print(f"\n  🎯 GRAND TOTAL: {grand_total} audio files")
    print("="*65)

def run_all():
    import time as _time
    start = _time.time()

    print("\n🔵 PHASE 1: Xeno-Canto → Birds")
    for name, sci in SPECIES_CATALOG["birds"].items():
        print(f"\n📂 {name}")
        download_xc_species(name, sci, "birds")

    print("\n🟢 PHASE 2: FreeSound → Frogs, Insects, Mammals")
    for cat in ["frogs", "insects", "mammals"]:
        print(f"\n── {cat.upper()} ──")
        for name, sci in SPECIES_CATALOG[cat].items():
            print(f"\n📂 {name}")
            download_fs_species(name, sci, cat)

    print("\n🟡 PHASE 3: ESC-50 (supplemental bulk dataset)")
    download_esc50()

    print("\n🟠 PHASE 4: iNaturalist → gap-filling species < 30 files")
    for cat in ["birds", "frogs", "insects", "mammals"]:
        for name, sci in SPECIES_CATALOG[cat].items():
            folder = Path(RAW_AUDIO_DIR) / cat / name.replace(" ", "_")
            n = count_files(folder)
            if n < 30:
                print(f"\n📂 {name} — supplementing ({n} files)")
                download_inat_species(name, sci, cat, need=30 - n)

    elapsed = _time.time() - start
    print_summary()
    print(f"\n⏱️  Total time: {elapsed/60:.1f} min")
    print("✅ Run file2_model_training.py next!\n")


if __name__ == "__main__":
    run_all()