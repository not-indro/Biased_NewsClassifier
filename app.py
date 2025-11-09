import os
import requests
import streamlit as st
import torch
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datetime import datetime
from zoneinfo import ZoneInfo

# ===================== Page / Meta =====================
st.set_page_config(
    page_title="MBIC — Media Bias Classifier",
    page_icon="🗽",
    layout="wide",
)

# ===================== Session State Bootstrap =====================
def init_state():
    ss = st.session_state
    # Feed state
    ss.setdefault("cat", "Top")              # default: Top
    ss.setdefault("query", "")               # blank by default for Top
    ss.setdefault("sort_by", "publishedAt")  # used by /everything
    ss.setdefault("view_mode", "Magazine")   # Magazine=3 cols, Compact=2 cols
    ss.setdefault("news_items", [])          # accumulated articles
    ss.setdefault("news_scores", [])         # accumulated scores
    ss.setdefault("news_page", 1)            # page (1..100)
    ss.setdefault("news_has_more", True)     # if more pages exist
    ss.setdefault("prev_params", None)

    # Theme / Timezone
    try:
        tzinfo = datetime.now().astimezone().tzinfo
        guessed_tz = getattr(tzinfo, "key", str(tzinfo))
    except Exception:
        guessed_tz = None
    if not guessed_tz or "/" not in str(guessed_tz):
        guessed_tz = "Asia/Kolkata"
    ss.setdefault("timezone", guessed_tz)
    ss.setdefault("theme", "Light")

init_state()

# ===================== Sidebar (Settings) =====================
DEFAULT_MODEL_ID = "hoshangchakravarty/media-bias-classifier"
with st.sidebar:
    st.header("Settings")
    model_id = st.text_input(
        "Hugging Face Model ID",
        value=os.getenv("MBIC_MODEL_ID", DEFAULT_MODEL_ID),
        help="Override the default fine-tuned model (e.g., distilbert-base-uncased fine-tuned on MBIC).",
    )
    show_probs = st.checkbox("Show raw probabilities (JSON)", value=False)

    # Theme
    theme_choice = st.radio(
        "Theme", ["Light", "Dark"],
        index=0 if st.session_state["theme"] == "Light" else 1,
        horizontal=True
    )
    st.session_state["theme"] = theme_choice
    IS_DARK = theme_choice == "Dark"

    # Timezone selector
    COMMON_TZS = [
        "UTC", "Pacific/Honolulu", "America/Anchorage", "America/Los_Angeles", "America/Denver",
        "America/Chicago", "America/New_York", "America/Sao_Paulo", "Europe/London", "Europe/Berlin",
        "Europe/Paris", "Europe/Moscow", "Africa/Johannesburg", "Asia/Dubai", "Asia/Kolkata",
        "Asia/Bangkok", "Asia/Singapore", "Asia/Hong_Kong", "Asia/Tokyo", "Australia/Sydney",
    ]
    guessed = st.session_state.get("timezone", "UTC")
    tz_options = [guessed] + [tz for tz in COMMON_TZS if tz != guessed]
    tz_selected = st.selectbox("Timezone (for dates shown in app)", tz_options, index=0)
    st.session_state["timezone"] = tz_selected

    st.caption("Tip: Set MBIC_MODEL_ID env var to override model globally.")

# ===================== Design Tokens & Global CSS =====================
LIGHT_TOKENS = """
:root {
  --bg: #ffffff;
  --ink: #0f172a;
  --muted: #5f6b7a;
  --line: #e6eaf0;
  --card: #ffffff;
  --chip: #f5f7fb;
  --brand: #274c77;
  --accent: #e63946;
  --good-1:#e6f7ec; --good-2:#eaf7ef; --mid:#efeaff; --warn:#ffefe0; --bad:#ffe8e8;
  --shadow: 0 6px 24px rgba(16,24,40,.06);
  --shadow-hover: 0 10px 28px rgba(16,24,40,.12);
  --media-bg: #f2f4f8;
  --desc: #2d3748;
  --chip-hover: #eef3fb;
  --pill-text: #274c77;
}
"""
DARK_TOKENS = """
:root {
  --bg: #0b1020;
  --ink: #e9eef7;
  --muted: #b7c2da;
  --line: #24355d;
  --card: #0f1730;
  --chip: #111b36;
  --brand: #7db0ff;
  --accent: #ff7b7b;
  --good-1:#15301f; --good-2:#1f2b22; --mid:#2a2631; --warn:#2c1f13; --bad:#3d1f1f;
  --shadow: 0 10px 30px rgba(0,0,0,.45);
  --shadow-hover: 0 16px 40px rgba(0,0,0,.6);
  --media-bg: #0e1a38;
  --desc: #d7def0;
  --chip-hover: #16224a;
  --pill-text: #d4def5;
}
"""

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Merriweather:wght@700;900&display=swap');
{DARK_TOKENS if IS_DARK else LIGHT_TOKENS}

/* App width & base */
[data-testid="stAppViewContainer"] .main .block-container {{ max-width: 1200px; padding-top: 0.5rem; }}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--bg); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }}
h1, h2, h3, h4 {{ color: var(--ink); }}

/* Masthead */
.header-wrap {{ position: sticky; top: 0; z-index: 20; background: { 'rgba(11,16,32,.92)' if IS_DARK else 'rgba(255,255,255,.9)' }; backdrop-filter: blur(6px); border-bottom: 1px solid var(--line); margin: -1rem -1rem 1rem -1rem; padding: .6rem 1rem; }}
.brand {{ display: flex; align-items: baseline; gap: .6rem; }}
.brand .logo {{ width: 14px; height: 14px; border-radius: 2px; background: var(--accent); box-shadow: 0 0 10px var(--accent); }}
.brand .name {{ font-family: Merriweather, Georgia, serif; font-weight: 900; letter-spacing: .3px; font-size: 1.25rem; }}
.tag {{ color: var(--muted); font-weight: 600; font-size: .9rem; }}

/* Ticker */
.ticker {{ display: flex; gap: 1rem; overflow: hidden; white-space: nowrap; border-top: 1px solid var(--line); margin-top: .5rem; padding-top: .4rem; }}
.ticker .item {{ color: var(--muted); }}
.ticker .dot {{ color: var(--accent); margin: 0 .5rem; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ font-weight: 700; padding: .7rem 1rem; border-radius: 999px; background: transparent; }}
.stTabs [data-baseweb="tab"]:hover {{ background: var(--chip-hover); }}
.stTabs [aria-selected="true"] {{ background: var(--chip); color: var(--brand); border: 1px solid var(--line); }}

/* Category pills (hover subtle, readable in both themes) */
.pills {{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0 .75rem 0; }}
.pill-btn > button {{
  width: 100%; background: var(--chip) !important; color: var(--pill-text) !important;
  border: 1px solid var(--line) !important; border-radius: 999px !important;
  font-weight: 700 !important; font-size: .85rem !important; padding: .35rem .6rem !important;
}}
.pill-btn > button:hover {{ background: var(--chip-hover) !important; }}
.pill-active > button {{ outline: 2px solid var(--brand) !important; outline-offset: 1px !important; }}

/* Inputs */
.stTextInput>div>div>input, textarea, .stSelectbox div[data-baseweb="select"]{{ background: var(--card) !important; color: var(--ink) !important; border:1px solid var(--line) !important; }}

/* Cards */
.card {{ border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: var(--card); box-shadow: var(--shadow); transition: transform .12s ease, box-shadow .2s ease; display: flex; flex-direction: column; height: 100%; }}
.card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-hover); }}
.card .media {{ position: relative; height: 190px; background: var(--media-bg); display:flex; align-items:center; justify-content:center; overflow:hidden; }}
.card .media img {{ width: 100%; height: 100%; object-fit: cover; transform: scale(1.0); transition: transform .25s ease; }}
.card:hover .media img {{ transform: scale(1.02); }}
.card .media::after {{ content: ""; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(0,0,0,0) 45%, rgba(0,0,0,.28) 100%); }}
.card .content {{ padding: 14px 14px 10px 14px; display:flex; flex-direction:column; gap:.35rem; flex:1; }}
.card h4 {{ font-family: Merriweather, Georgia, serif; font-weight: 900; margin: 4px 0 6px 0; line-height: 1.25; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
.card .meta {{ color: var(--muted); font-size: .9rem; }}
.card .desc {{ color: var(--desc); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; min-height: 3.6em; }}
.card .actions {{ margin-top: auto; display:flex; flex-direction:column; gap:.35rem; }}

/* CSS Bias meter */
.meter {{ position: relative; height: 10px; border-radius: 999px; overflow: hidden; border: 1px solid var(--line); background:
  linear-gradient(90deg, var(--good-1) 0%, var(--good-1) 25%, var(--good-2) 25%, var(--good-2) 45%, var(--mid) 45%, var(--mid) 55%, var(--warn) 55%, var(--warn) 75%, var(--bad) 75%, var(--bad) 100%); }}
.meter .thumb {{ position:absolute; top:-4px; width:2px; height:18px; background: var(--accent); left: 0%; }}
.meter .labelrow {{ display:flex; justify-content:space-between; font-size:.75rem; color: var(--muted); margin-top: .25rem; }}

/* Tiles */
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; }}
.tile .label {{ color: var(--muted); font-size: .85rem; }}
.tile .value {{ color: var(--ink); font-size: 1.35rem; font-weight: 800; }}

/* Badges */
.badge {{ display:inline-block; padding:6px 10px; border-radius: 999px; font-weight:700; border:1px solid rgba(0,0,0,.06); font-size:.85rem; }}
.badge-biased {{ background: var(--bad); color:{ '#ffb3b3' if IS_DARK else '#9b1c1c' }; }}
.badge-slight-biased {{ background: var(--warn); color:{ '#ffd4a6' if IS_DARK else '#7a3c00' }; }}
.badge-uncertain {{ background: var(--mid); color:{ '#d6c9ff' if IS_DARK else '#4338ca' }; }}
.badge-slight-non {{ background: var(--good-2); color:{ '#c9f4da' if IS_DARK else '#166534' }; }}
.badge-non {{ background: var(--good-1); color:{ '#b6f1c8' if IS_DARK else '#0b6b2d' }; }}

/* Footer */
.footer {{ border-top:1px solid var(--line); margin-top: 1rem; padding-top: .75rem; color: var(--muted); }}
</style>
""",
    unsafe_allow_html=True,
)

# ===================== Time Helpers (User Perspective) =====================
def now_in_tz(tz_name: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now(ZoneInfo("UTC"))

def format_dt(dt: datetime, tz_name: str) -> str:
    try:
        local = dt.astimezone(ZoneInfo(tz_name))
        abbr = local.tzname() or tz_name
        return local.strftime(f"%b %d, %Y %H:%M {abbr}")
    except Exception:
        return dt.strftime("%b %d, %Y %H:%M")

def parse_and_localize(iso_str: str, tz_name: str) -> str:
    if not iso_str:
        return ""
    try:
        dt_utc = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return format_dt(dt_utc, tz_name)
    except Exception:
        return iso_str

# ===================== Masthead =====================
ticker_now = format_dt(now_in_tz(st.session_state["timezone"]), st.session_state["timezone"])
st.markdown(
    f"""
<div class="header-wrap">
  <div class="brand">
    <div class="logo"></div>
    <div class="name">MBIC</div>
    <div class="tag">Media Bias Classifier</div>
  </div>
  <div class="ticker" id="ticker">
    <span class="item">Today • {ticker_now}</span><span class="dot">•</span>
    <span class="item">Analyze headlines for bias in real time</span><span class="dot">•</span>
    <span class="item">Powered by DistilBERT + Streamlit</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ===================== Device / Model =====================
def _pick_device_for_pipeline():
    if torch.cuda.is_available():
        return 0
    return -1

@st.cache_resource(show_spinner=True)
def load_classifier(model_id_str: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_id_str, use_fast=True, local_files_only=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_id_str, local_files_only=False)
        model.config.id2label = getattr(model.config, "id2label", {0: "Non-biased", 1: "Biased"})
        model.config.label2id = getattr(model.config, "label2id", {"Non-biased": 0, "Biased": 1})

        device = _pick_device_for_pipeline()
        if device == -1 and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                model.to("mps")
                return pipeline("text-classification", model=model, tokenizer=tok, device=-1, top_k=None)
            except Exception:
                pass
        return pipeline("text-classification", model=model, tokenizer=tok, device=device, top_k=None)
    except Exception as e:
        st.error(f"Could not load model `{model_id_str}`.\n\n{e}")
        st.stop()

clf = load_classifier(model_id)

# ===================== Helpers =====================
def _normalize_scores(result):
    def _fix(items):
        out = []
        for s in items:
            lab = s.get("label", "")
            score = float(s.get("score", 0.0))
            if lab in ("LABEL_0", "LABEL_1"):
                lab = "Biased" if lab.endswith("1") else "Non-biased"
            out.append({"label": lab, "score": score})
        return out
    if not result: return []
    if isinstance(result, list) and result and isinstance(result[0], dict): return _fix(result)
    if isinstance(result, list) and result and isinstance(result[0], list): return [_fix(r) for r in result]
    return result

def classify_batch(texts):
    return _normalize_scores(clf(texts, top_k=None))

def graded_label(p_biased: float):
    if p_biased >= 0.75: return "Biased", "badge-biased"
    if p_biased >= 0.55: return "Slightly biased", "badge-slight-biased"
    if p_biased >  0.45: return "Mixed / uncertain", "badge-uncertain"
    if p_biased >= 0.25: return "Slightly non-biased", "badge-slight-non"
    return "Non-biased", "badge-non"

def tile(label: str, value: str):
    st.markdown(f"<div class='tile'><div class='label'>{label}</div><div class='value'>{value}</div></div>", unsafe_allow_html=True)

def css_bias_meter(p_biased: float) -> str:
    pct = max(0.0, min(1.0, float(p_biased))) * 100.0
    pct_str = f"{pct:.0f}%"
    return f"""
    <div class="meter">
      <div class="thumb" style="left: calc({pct_str} - 1px);"></div>
    </div>
    <div class="labelrow">
      <span>Non-biased</span><span>Biased</span>
    </div>
    """

# ===================== NewsAPI =====================
PILL_TO_NEWSAPI = {
    "Top": ("top-headlines", {"category": "general"}),
    "Technology": ("top-headlines", {"category": "technology"}),
    "Business": ("top-headlines", {"category": "business"}),
    "Entertainment": ("top-headlines", {"category": "entertainment"}),
    "Health": ("top-headlines", {"category": "health"}),
    "Science": ("top-headlines", {"category": "science"}),
    "Sports": ("top-headlines", {"category": "sports"}),
    # Non-top-headlines categories -> everything
    "World": ("everything", {"q": "world OR international"}),
    "Climate": ("everything", {"q": "climate OR emissions OR carbon OR global warming"}),
    "Politics": ("everything", {"q": "politics OR election OR parliament OR congress"}),
}

@st.cache_data(ttl=120)
def fetch_news_page(endpoint: str, params: dict, page: int, page_size: int = 9):
    """Fetch a specific page from NewsAPI. Returns (articles, status)."""
    key = st.secrets.get("NEWSAPI_KEY", "")
    if not key:
        return [], "offline"
    try:
        base = f"https://newsapi.org/v2/{endpoint}"
        p = dict(language="en", page=int(page), pageSize=int(page_size))
        p.update(params)
        p["apiKey"] = key
        r = requests.get(base, params=p, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            return [], data.get("code", "error")
        return data.get("articles", []), "ok"
    except requests.exceptions.RequestException:
        return [], "offline"

# ========== Auto-trigger logic (param changes reset feed) ==========
def _current_params():
    cat = st.session_state.get("cat", "Top")
    query = st.session_state.get("query", "")
    sort_by = st.session_state.get("sort_by", "publishedAt")
    view_mode = st.session_state.get("view_mode", "Magazine")
    tz = st.session_state.get("timezone", "UTC")
    theme = st.session_state.get("theme", "Light")
    return (cat, query, sort_by, view_mode, tz, theme)

def _params_changed(prev, curr):
    return prev != curr

def _reset_feed():
    st.session_state["news_items"] = []
    st.session_state["news_scores"] = []
    st.session_state["news_page"] = 1
    st.session_state["news_has_more"] = True

def _append_page(articles, scores, page_size):
    st.session_state["news_items"].extend(articles)
    st.session_state["news_scores"].extend(scores)
    st.session_state["news_page"] += 1
    # if fewer than requested, no more pages
    if len(articles) < page_size:
        st.session_state["news_has_more"] = False

def _build_endpoint_and_params(cat: str, query: str, sort_by: str):
    """
    Decide endpoint and build params based on category + optional query.
    - For 'Top' and NewsAPI categories -> /top-headlines (category mapped)
    - For World/Climate/Politics or any custom search -> /everything with composed q and sortBy
    """
    mapping = PILL_TO_NEWSAPI.get(cat, None)
    if mapping:
        endpoint, base = mapping
    else:
        endpoint, base = "everything", {"q": cat.lower()}

    params = dict(base)
    if endpoint == "everything":
        q_parts = []
        if base.get("q"):
            q_parts.append(f"({base['q']})")
        if query.strip():
            q_parts.append(f"({query.strip()})")
        params["q"] = " AND ".join(q_parts) if q_parts else "news"
        params["sortBy"] = sort_by
    else:
        # /top-headlines allows q as refinement (optional)
        if query.strip():
            params["q"] = query.strip()
    return endpoint, params

# ===================== Tabs =====================
tab_live, tab_classify, tab_about = st.tabs(["📰 Live News", "🔎 Classify", "📊 About / Results"])

# ----------------- Tab 1: Live News -----------------
with tab_live:
    st.subheader("Top Stories")

    # Category pills (clean hover + active)
    categories = ["Top", "Technology", "Business", "Entertainment", "Health", "Science", "Sports", "World", "Climate", "Politics"]
    pill_cols = st.columns(len(categories))
    for i, c in enumerate(categories):
        with pill_cols[i]:
            active_class = " pill-active" if st.session_state.get("cat") == c else ""
            st.markdown(f"<div class='pill-btn{active_class}'>", unsafe_allow_html=True)
            if st.button(c, key=f"pill_{c}", use_container_width=True):
                # Set category and clear search to avoid unintended filtering
                st.session_state["cat"] = c
                st.session_state["query"] = "" if c == "Top" else ""
                _reset_feed()
            st.markdown("</div>", unsafe_allow_html=True)

    # Controls row (reactive)
    c1, c2, c3 = st.columns([1.8, 1.1, 1.1])
    with c1:
        st.session_state["query"] = st.text_input(
            "Search within category (optional)",
            value=st.session_state.get("query", ""),
            placeholder="e.g., economy, climate, elections",
            key="query_input",
        )
    with c2:
        # Only meaningful for /everything; harmless for /top-headlines
        st.session_state["sort_by"] = st.selectbox(
            "Sort by", ["publishedAt", "relevancy", "popularity"],
            index=["publishedAt","relevancy","popularity"].index(st.session_state.get("sort_by", "publishedAt")),
            key="sort_by_select",
        )
    with c3:
        st.session_state["view_mode"] = st.selectbox(
            "View", ["Magazine", "Compact"],
            index=["Magazine","Compact"].index(st.session_state.get("view_mode", "Magazine")),
            key="view_mode_select",
        )

    # Param-change detection (auto refresh feed)
    curr = _current_params()
    if st.session_state["prev_params"] is None or _params_changed(st.session_state["prev_params"], curr):
        _reset_feed()
    st.session_state["prev_params"] = curr

    # Compute endpoint/params for current selection
    endpoint, base_params = _build_endpoint_and_params(curr[0], curr[1], curr[2])

    # Initial load (if empty)
    PAGE_SIZE = 9
    if not st.session_state["news_items"] and st.session_state["news_has_more"]:
        with st.spinner("Fetching top stories…"):
            page_articles, status = fetch_news_page(
                endpoint=endpoint,
                params=base_params,
                page=st.session_state["news_page"],
                page_size=PAGE_SIZE
            )
            if status != "ok":
                if status == "offline":
                    st.error("🛰 NewsAPI unreachable or key missing. Add NEWSAPI_KEY in `.streamlit/secrets.toml`.")
                elif status == "rateLimited":
                    st.warning("⏳ NewsAPI rate limit reached. Try again shortly.")
                else:
                    st.error(f"⚠️ NewsAPI error: {status}. Check your query and key.")
                st.session_state["news_has_more"] = False
            elif not page_articles:
                st.info("No articles found. Try another category or search.")
                st.session_state["news_has_more"] = False
            else:
                texts = [(" ".join([a.get("title") or "", a.get("description") or ""]).strip()) or (a.get("title") or "(no text)") for a in page_articles]
                page_scores = classify_batch(texts)
                _append_page(page_articles, page_scores, PAGE_SIZE)

    # Snapshot tiles from accumulated scores
    if st.session_state["news_scores"]:
        biased_cnt = 0
        avg_biased = 0.0
        for sc in st.session_state["news_scores"]:
            m = {s["label"]: s["score"] for s in sc}
            p_b = float(m.get("Biased", 0.0))
            if p_b >= 0.5:
                biased_cnt += 1
            avg_biased += p_b
        avg_biased = avg_biased / max(len(st.session_state["news_scores"]), 1)

        t1, t2, t3 = st.columns(3)
        with t1: tile("Stories Loaded", f"{len(st.session_state['news_scores'])}")
        with t2: tile("Predicted Biased (≥0.5)", f"{biased_cnt}")
        with t3: tile("Avg P(Biased)", f"{avg_biased:.2f}")
        st.write("")

    # Grid render from accumulated items
    items = st.session_state["news_items"]
    scores_all = st.session_state["news_scores"]
    tz_name = st.session_state["timezone"]
    ncols = 3 if st.session_state["view_mode"] == "Magazine" else 2
    rows = [st.columns(ncols) for _ in range((len(items)+ncols-1)//ncols)]

    idx = 0
    for row in rows:
        for col in row:
            if idx >= len(items): break
            a = items[idx]
            title = a.get("title") or "(no title)"
            desc = a.get("description") or ""
            url = a.get("url")
            src = (a.get("source") or {}).get("name", "")
            published = parse_and_localize(a.get("publishedAt", ""), tz_name)
            img = a.get("urlToImage")

            score_list = scores_all[idx] if idx < len(scores_all) else []
            s_map = {s["label"]: s["score"] for s in score_list}
            p_biased = float(s_map.get("Biased", 0.0))
            p_non = float(s_map.get("Non-biased", 0.0))
            grade, badge_class = graded_label(p_biased)

            with col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                # Media
                st.markdown("<div class='media'>", unsafe_allow_html=True)
                if img:
                    st.image(img, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='width:100%;height:190px;display:flex;align-items:center;justify-content:center;"
                        f"color:{'#cfd8e3' if IS_DARK else '#9aa3b2'};border-bottom:1px solid var(--line)'>No image</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

                # Content
                st.markdown("<div class='content'>", unsafe_allow_html=True)
                if src or published:
                    st.markdown(f"<div class='meta'>{src} • {published}</div>", unsafe_allow_html=True)
                st.markdown(f"<h4 title='{title}'>{title}</h4>", unsafe_allow_html=True)
                if desc:
                    st.markdown(f"<div class='desc'>{desc}</div>", unsafe_allow_html=True)

                # Actions / badges / meter
                st.markdown("<div class='actions'>", unsafe_allow_html=True)
                if url:
                    st.markdown(f"[Read full article]({url})")
                st.markdown(f"<span class='badge {badge_class}'>{grade}</span>", unsafe_allow_html=True)
                st.caption(f"P(Non-biased)={p_non:.2f} • P(Biased)={p_biased:.2f}")
                st.markdown(css_bias_meter(p_biased), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
            idx += 1

    # ===== Infinite-ish loading (scroll → tap once; no per-page selector) =====
    load_holder = st.empty()
    if st.session_state["news_has_more"]:
        with load_holder.container():
            more = st.button("Load more", use_container_width=True, type="secondary")
        if more:
            with st.spinner("Loading more…"):
                page_articles, status = fetch_news_page(
                    endpoint=endpoint,
                    params=base_params,
                    page=st.session_state["news_page"],
                    page_size=PAGE_SIZE
                )
                if status != "ok":
                    if status == "rateLimited":
                        st.warning("⏳ Reached rate limit. Try again in a bit.")
                    else:
                        st.error("⚠️ Couldn’t load more right now.")
                    st.session_state["news_has_more"] = False
                elif not page_articles:
                    st.session_state["news_has_more"] = False
                else:
                    texts = [(" ".join([a.get("title") or "", a.get("description") or ""]).strip()) or (a.get("title") or "(no text)") for a in page_articles]
                    page_scores = classify_batch(texts)
                    _append_page(page_articles, page_scores, PAGE_SIZE)

    with st.expander("Top sources in this session"):
        if items:
            srcs = pd.Series([(a.get("source") or {}).get("name","") for a in items]).value_counts()
            st.dataframe(srcs.to_frame("Count"))
        else:
            st.caption("No sources yet.")

# ----------------- Tab 2: Classify -----------------
with tab_classify:
    st.subheader("Classify Custom Text")
    st.markdown("<div class='pills'><span style='background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:.25rem .5rem;font-weight:700;'>Bias Meter</span> <span style='background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:.25rem .5rem;font-weight:700;'>Binary Threshold</span> <span style='background:var(--chip);border:1px solid var(--line);border-radius:999px;padding:.25rem .5rem;font-weight:700;'>Model: DistilBERT</span></div>", unsafe_allow_html=True)

    with st.form("classify_form"):
        text_input = st.text_area(
            "Paste a headline or short paragraph",
            height=160,
            placeholder="Example: Government accused of hiding key evidence in climate report",
        )
        threshold = st.slider(
            "Binary threshold", 0.0, 1.0, 0.50, 0.01,
            help="Used only for the binary label; the gauge shows the full spectrum."
        )
        submitted = st.form_submit_button("Classify", use_container_width=True)

    if submitted:
        if not text_input.strip():
            st.warning("Please paste some text to classify.")
        else:
            with st.spinner("Analyzing…"):
                res = clf(text_input.strip(), top_k=None)
                scores = sorted(_normalize_scores(res), key=lambda x: x["score"], reverse=True)
                p_b = next((s["score"] for s in scores if s["label"] == "Biased"), 0.0)
                p_non = next((s["score"] for s in scores if s["label"] == "Non-biased"), 0.0)
                grade, badge_class = graded_label(p_b)
                binary = "Biased" if p_b >= threshold else "Non-biased"

                l, r = st.columns([1, 2])
                with l:
                    tile("Binary (thresholded)", binary)
                    st.markdown(f"<span class='badge {badge_class}'>{grade}</span>", unsafe_allow_html=True)
                    st.caption(f"P(Non-biased)={p_non:.2f} • P(Biased)={p_b:.2f}")
                with r:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=float(p_b),
                            number={'valueformat': ".2f"},
                            gauge={
                                'axis': {'range': [0, 1], 'tickwidth': 0},
                                'bar': {'color': '#274c77' if st.session_state["theme"] == "Light" else '#7db0ff'},
                                'bgcolor': "rgba(0,0,0,0)",
                                'borderwidth': 0,
                                'steps': [
                                    {'range': [0.00, 0.25], 'color': ('#e6f7ec' if st.session_state["theme"] == "Light" else '#15301f')},
                                    {'range': [0.25, 0.45], 'color': ('#eaf7ef' if st.session_state["theme"] == "Light" else '#1f2b22')},
                                    {'range': [0.45, 0.55], 'color': ('#efeaff' if st.session_state["theme"] == "Light" else '#2a2631')},
                                    {'range': [0.55, 0.75], 'color': ('#ffefe0' if st.session_state["theme"] == "Light" else '#2c1f13')},
                                    {'range': [0.75, 1.00], 'color': ('#ffe8e8' if st.session_state["theme"] == "Light" else '#3d1f1f')},
                                ],
                            },
                            domain={'x': [0, 1], 'y': [0, 1]}
                        )
                    )
                    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor='rgba(0,0,0,0)', height=160)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                if show_probs:
                    st.json([{s['label']: round(s['score'], 4)} for s in scores])

# ----------------- Tab 3: About / Results -----------------
with tab_about:
    st.title("About the Model & Results")
    st.caption("MBIC • DistilBERT-based classifier for linguistic bias in headlines.")

    st.header("Overview")
    st.markdown(
        "Detect **biased language** in short texts (news headlines, social posts, editorials). "
        "Outputs a probability for *Biased* vs *Non-biased* and a graded label."
    )

    st.header("Model")
    st.markdown(
        "**Base:** `distilbert-base-uncased`  \n"
        "**Classes:** `Non-biased (0)`, `Biased (1)`  \n"
        "**Loss:** CrossEntropy • **Opt:** AdamW  \n"
        "**Frameworks:** 🤗 Transformers + PyTorch"
    )

    st.header("Dataset")
    st.markdown("MBIC (Kaggle) • ~1.5K English samples • 80/20 split.")
    overview_df = pd.DataFrame(
        [("Total Samples", "1551"), ("Train Set", "1240"), ("Test Set", "311"), ("Labels", "2 (Biased / Non-biased)")],
        columns=["Metric", "Count"],
    )
    st.table(overview_df)

    st.header("Training Config")
    st.code(
        """Model: distilbert-base-uncased
Batch size (train): 16
Batch size (eval): 32
Learning rate: 2e-5
Epochs: 4
Weight decay: 0.01
Optimizer: AdamW
Device: CUDA / CPU (auto); MPS attempted w/ safe fallback""",
        language="yaml",
    )

    st.header("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "73.6%")
    c2.metric("F1 (macro)", "0.68")
    c3.metric("F1 (weighted)", "0.72")

    metrics_df = pd.DataFrame({"Metric": ["Accuracy", "F1 (macro)", "F1 (weighted)"], "Score": [0.736, 0.68, 0.72]})
    chart = (
        alt.Chart(metrics_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort=None),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format="~p")),
            tooltip=["Metric", "Score"],
        )
        .properties(height=220)
        + alt.Chart(metrics_df).mark_text(dy=-6, fontWeight="bold").encode(
            x="Metric:N", y="Score:Q", text=alt.Text("Score:Q", format=".2f")
        )
    )
    st.altair_chart(chart, use_container_width=True)

    st.header("Examples")
    examples = [
        ("Government accused of covering up data", "Biased"),
        ("China to remove tariffs on US agriculture goods from Nov 10", "Non-biased"),
        ("New reforms spark outrage across citizens", "Biased"),
        ("Prime Minister to address nation tomorrow", "Non-biased"),
    ]
    for text, label in examples:
        color = ("#e63946" if not IS_DARK else "#ff7b7b") if label == "Biased" else ("#0b6b2d" if not IS_DARK else "#b6f1c8")
        st.markdown(
            f"<div style='padding:10px 12px;border:1px solid var(--line);border-left:6px solid {color};border-radius:10px;margin:6px 0;background:var(--card);'><b>{text}</b><br><span style='color: var(--muted)'>Predicted: {label}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='footer'>© MBIC — Media Bias Classifier</div>", unsafe_allow_html=True)
