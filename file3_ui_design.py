import streamlit as st

# Theme 
PALETTE = {
    "bg": "#0d1f0e",
    "surface": "#122614",
    "surface2": "#1a3320",
    "border": "#2d5c35",
    "accent": "#4ade80",
    "accent2": "#86efac",
    "accent3": "#fbbf24",
    "text": "#e2f5e4",
    "muted": "#6b9e74",
    "danger": "#f87171",
    "bird": "#38bdf8",
    "frog": "#4ade80",
    "insect": "#fbbf24",
    "mammal": "#f472b6",
}

CATEGORY_META = {
    "birds":   {"emoji": "🐦", "color": PALETTE["bird"],   "label": "Bird"},
    "frogs":   {"emoji": "🐸", "color": PALETTE["frog"],   "label": "Amphibian"},
    "insects": {"emoji": "🦗", "color": PALETTE["insect"], "label": "Insect"},
    "mammals": {"emoji": "🦊", "color": PALETTE["mammal"], "label": "Mammal"},
    "unknown": {"emoji": "❓", "color": PALETTE["muted"],  "label": "Unknown"},
}

SPECIES_FACTS = {
    "Barn Owl": "Silent hunters — their heart-shaped face acts as a natural sound dish.",
    "Common Cuckoo": "Famous for laying eggs in other birds' nests.",
    "Common Nightingale": "Can sing over 200 distinct song types.",
    "Blue Jay": "Can mimic hawk calls to scare other birds.",
    "Common Loon": "Their haunting call echoes across lakes.",
    "Bald Eagle": "Their call is surprisingly high-pitched.",
    "American Bullfrog": "Their call can carry over 1.5km.",
    "Spring Peeper": "Tiny frog, but incredibly loud in groups.",
    "Gray Tree Frog": "Can survive freezing in winter.",
    "Common Frog": "Triggers full pond choruses in spring.",
    "Honeybee": "Communicates through waggle dances.",
    "Field Cricket": "Males chirp by rubbing wings.",
    "Cicada": "Some emerge only every 17 years.",
    "Mosquito": "Females hum around 400Hz.",
    "Gray Wolf": "Howls can travel 10km.",
    "Common Pipistrelle Bat": "Uses rapid ultrasonic pulses.",
    "African Lion": "Roars can be heard from 8km away.",
    "Chimpanzee": "Each has a unique vocal signature.",
}
# Inject styles
def inject_css():
    st.markdown(f"""
    <style>
    body {{
        background: {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'Outfit', sans-serif;
    }}
    .card {{
        background: {PALETTE['surface']};
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid {PALETTE['border']};
        margin: 1rem 0;
    }}
    .title {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {PALETTE['accent']};
    }}
    .muted {{
        color: {PALETTE['muted']};
        font-size: 0.9rem;
    }}
    .bar {{
        height: 10px;
        border-radius: 10px;
        background: {PALETTE['surface2']};
        overflow: hidden;
    }}
    .fill {{
        height: 100%;
        background: linear-gradient(90deg, {PALETTE['accent']}, {PALETTE['accent3']});
    }}
    </style>
    """, unsafe_allow_html=True)

def hero():
    st.markdown("""
    <div class="card">
        <div class="title">🎙️ EchoSense</div>
        <p class="muted">
            Upload wildlife audio and identify species instantly —
            birds, frogs, insects & mammals.
        </p>
    </div>
    """, unsafe_allow_html=True)


def result_card(result):
    meta = CATEGORY_META.get(result.get("category", "").lower(), CATEGORY_META["unknown"])

    st.markdown(f"""
    <div class="card">
        <h2 style="color:{meta['color']}">{meta['emoji']} {result.get("species","Unknown")}</h2>
        <p class="muted">{meta['label']} · {int(result.get("confidence",0)*100)}%</p>
    </div>
    """, unsafe_allow_html=True)


def confidence_bar(value):
    pct = int(value * 100)
    st.markdown(f"""
    <div class="bar">
        <div class="fill" style="width:{pct}%"></div>
    </div>
    <p class="muted">{pct}% confidence</p>
    """, unsafe_allow_html=True)

def top3(results):
    for i, r in enumerate(results):
        confidence_bar(r["confidence"])
        st.write(f"{i+1}. {r['species']}")


def fact_box(name):
    fact = SPECIES_FACTS.get(name)
    if fact:
        st.info(f"💡 {fact}")


def stats(duration, sr, size):
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration:.1f}s")
    col2.metric("Sample Rate", f"{sr//1000} kHz")
    col3.metric("Size", f"{size:.0f} KB")


def footer():
    st.markdown(
        "<p class='muted' style='text-align:center'>EchoSense · Demo Build</p>",
        unsafe_allow_html=True
    )
