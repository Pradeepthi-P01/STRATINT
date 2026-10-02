"""
ner.py — Phase 2: Named Entity Recognition using spaCy en_core_web_lg.

════════════════════════════════════════════════════════════════════════════════
DESIGN CHOICES (plain English)
════════════════════════════════════════════════════════════════════════════════

1. Why en_core_web_lg (and how to upgrade to en_core_web_trf)
   ──────────────────────────────────────────────────────────────
   The original plan was en_core_web_trf (RoBERTa-based, ~90 F1 on OntoNotes).
   However, en_core_web_trf is NOT installable on Python 3.13 + Windows without
   issues:
     - v3.7.x: requires blis Cython compilation which fails on Python 3.13's
               stricter GIL semantics.
     - v3.8.x: installs cleanly but downloads ~500 MB of RoBERTa weights from
               HuggingFace at runtime (first spacy.load() call), which can
               take 10+ minutes and fails on restricted networks.

   We therefore default to en_core_web_lg:
     - Self-contained 560 MB wheel, no runtime downloads.
     - Works on Python 3.13 + Windows immediately.
     - Uses floret vectors; achieves ~85.5 F1 on OntoNotes NER.
     - For a student OSINT project this is perfectly adequate.

   To upgrade to trf later (e.g. in a Python 3.11/3.12 venv or Colab):
     pip install spacy==3.7.5
     python -m spacy download en_core_web_trf
     python ner.py --model en_core_web_trf

2. Entity types: GPE, ORG, NORP only
   ────────────────────────────────────
   spaCy's OntoNotes model recognises 18 entity types.  We discard all but
   three because they are the only ones with geographic or actor relevance:
     GPE  — Countries, cities, states used as political actors.
     ORG  — Organisations, militant groups, governments, companies.
     NORP — Nationalities, political/religious groups (e.g. "Islamist",
            "Republican", "Afghan").

3. Text used for NER: title + " " + summary
   ──────────────────────────────────────────
   We concatenate title and summary into a single string.  The separator
   space keeps character offsets consistent.  spaCy offsets are stored in
   the DB so the Phase 6 dashboard can highlight entities in the source text.

4. Entity normaliser
   ──────────────────
   Raw surface forms from news are inconsistent: "U.S.", "America", "Washington",
   "the Americans" all refer to the same country.  The normaliser maps to
   ISO 3166-1 alpha-2 codes so downstream aggregation (Phase 6: tension score
   per country) works cleanly.

   Priority order:
     a. MANUAL_MAP  — hand-curated for abbreviations, capital-city metonyms,
        demonyms, and regional blocs.
     b. pycountry.countries.lookup()  — covers all ISO-standardised full names
        not in the manual map.
     c. None  — entity is kept in the DB but canonical is NULL (e.g., ORG
        entities that are not countries, NORP groups like "Republicans").

   Regional blocs (EU, NATO, ASEAN, UN, etc.) get a "BLOC:xxx" canonical so
   Phase 6 can handle them separately rather than mapping them to one country.

5. Incremental processing
   ──────────────────────
   Uses `get_unprocessed_articles(conn, "entities")` from db.py so re-running
   ner.py only processes articles that don't yet have any entity rows.
   If you want to re-run NER on all articles (e.g. after changing the model),
   run with `--rerun` which deletes existing entity rows first.

6. nlp.pipe() for batch throughput
   ─────────────────────────────────
   spaCy's `nlp.pipe()` processes a list of texts as a stream, which is more
   efficient than calling `nlp(text)` in a loop because it batches GPU/CPU
   compute.  We disable pipeline components that we don't need (tagger,
   parser, senter, attribute_ruler, lemmatizer) to cut memory and time.

7. Dedup within article
   ─────────────────────
   The same entity surface form may appear multiple times in an article.
   We dedup on (article_id, text_lower, label) — keeping only the first
   occurrence — to avoid inflating entity counts. The canonical code and
   char offsets of the first occurrence are stored.

════════════════════════════════════════════════════════════════════════════════
ASSUMPTIONS
════════════════════════════════════════════════════════════════════════════════
- en_core_web_lg is installed:
    python -m spacy download en_core_web_lg
    (or: pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl)
- pycountry is installed:
    pip install pycountry
- Articles were ingested by ingest.py and are in osint.db.
- Python 3.11+ (str | None union hints).

USAGE
──────
    python ner.py                       # Process all unprocessed articles
    python ner.py --limit 50            # Process at most 50 articles
    python ner.py --batch-size 8        # Use batch size 8 (default: 16)
    python ner.py --rerun               # Delete & recompute all entity rows
    python ner.py --generate-validation # Sample 25 articles → JSONL template
    python ner.py --generate-validation --val-size 30
    python ner.py --model en_core_web_trf  # Use transformer model (Python 3.11+)
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── UTF-8 stdout fix for Windows PowerShell ──────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tabulate import tabulate

from db import get_connection, init_db, migrate_db, get_unprocessed_articles

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ner.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("ner")

# ─────────────────────────────────────────────────────────────────────────────
# Entity target labels — spaCy labels we care about (discard the rest)
# ─────────────────────────────────────────────────────────────────────────────
TARGET_LABELS = {"GPE", "ORG", "NORP"}

# ─────────────────────────────────────────────────────────────────────────────
# Manual normalisation map: lowercase surface form → ISO alpha-2 or BLOC:xxx
#
# Design rationale:
#   - Capital city metonyms: "Washington", "Moscow", "Beijing" are routinely
#     used in wire-service prose to mean the national government.
#   - Abbreviations: "U.S.", "UK" etc. fail pycountry lookup.
#   - Demonyms: spaCy tags "American", "Russian" as NORP, not GPE, so
#     pycountry (which only handles country names) won't resolve them.
#   - Regional blocs are NOT countries; mapping them to a single ISO code
#     would be semantically wrong. "BLOC:xxx" is a sentinel prefix.
# ─────────────────────────────────────────────────────────────────────────────
MANUAL_MAP: dict[str, str] = {
    # ── Abbreviations & common informal names ─────────────────────────────
    "u.s.": "US",  "u.s.a.": "US", "usa": "US",
    "u.k.": "GB",  "uk": "GB",     "britain": "GB", "great britain": "GB",
    "uae": "AE",   "u.a.e.": "AE",
    "drc": "CD",                    # Democratic Republic of Congo
    "roc": "TW",                    # Republic of China (used for Taiwan)
    "prc": "CN",                    # People's Republic of China
    "rok": "KR",                    # Republic of Korea
    "dprk": "KP",  "north korea": "KP", "nk": "KP",
    "south korea": "KR",            "sk": "KR",
    "czechia": "CZ",
    "taiwan": "TW",
    "palestine": "PS",
    "western sahara": "EH",
    "kosovo": "XK",                 # Not yet ISO 3166-1, use interim XK
    "burma": "MM",                  # Myanmar official name, Burma common name

    # ── Capital cities used as government metonyms ────────────────────────
    # These appear in wire-service prose to mean the government of that country.
    "washington": "US",     "washington d.c.": "US", "washington dc": "US",
    "the white house": "US","white house": "US",     "pentagon": "US",
    "moscow": "RU",         "the kremlin": "RU",     "kremlin": "RU",
    "beijing": "CN",        "peking": "CN",
    "kyiv": "UA",           "kiev": "UA",
    "tehran": "IR",
    "london": "GB",         "whitehall": "GB",       "downing street": "GB",
    "paris": "FR",          "elysee": "FR",
    "new delhi": "IN",      "delhi": "IN",
    "islamabad": "PK",
    "riyadh": "SA",
    "ankara": "TR",
    "tokyo": "JP",
    "brussels": "BE",           # Belgium capital; also EU HQ metonym
    "kabul": "AF",
    "baghdad": "IQ",
    "damascus": "SY",
    "cairo": "EG",
    "tel aviv": "IL",       "jerusalem": "IL",
    "gaza": "PS",           "gaza city": "PS",  # Palestinian territory
    "ramallah": "PS",
    "pyongyang": "KP",
    "seoul": "KR",
    "bangkok": "TH",
    "nairobi": "KE",
    "pretoria": "ZA",       "cape town": "ZA",
    "taipei": "TW",
    "bern": "CH",           "geneva": "CH",
    "ottawa": "CA",
    "canberra": "AU",
    "warsaw": "PL",
    "budapest": "HU",
    "athens": "GR",
    "stockholm": "SE",
    "berlin": "DE",
    "rome": "IT",
    "madrid": "ES",
    "amsterdam": "NL",      "the hague": "NL",
    "vienna": "AT",
    "kyiv": "UA",
    "minsk": "BY",
    "tbilisi": "GE",
    "baku": "AZ",
    "yerevan": "AM",
    "tashkent": "UZ",
    "astana": "KZ",
    "dhaka": "BD",
    "kathmandu": "NP",
    "colombo": "LK",
    "rangoon": "MM",        "yangon": "MM",
    "hanoi": "VN",
    "jakarta": "ID",
    "manila": "PH",
    "kuala lumpur": "MY",
    "singapore": "SG",
    "addis ababa": "ET",
    "khartoum": "SD",
    "tripoli": "LY",
    "algiers": "DZ",
    "tunis": "TN",
    "rabat": "MA",
    "accra": "GH",
    "abuja": "NG",
    "kinshasa": "CD",
    "caracas": "VE",
    "bogota": "CO",
    "lima": "PE",
    "santiago": "CL",
    "buenos aires": "AR",
    "brasilia": "BR",       "rio de janeiro": "BR",

    # ── Demonyms (spaCy labels these as NORP, not GPE) ─────────────────────
    "american": "US",       "americans": "US",
    "russian": "RU",        "russians": "RU",
    "chinese": "CN",
    "british": "GB",
    "french": "FR",
    "german": "DE",         "germans": "DE",
    "indian": "IN",         "indians": "IN",
    "pakistani": "PK",      "pakistanis": "PK",
    "iranian": "IR",        "iranians": "IR",       "persian": "IR",
    "israeli": "IL",        "israelis": "IL",
    "ukrainian": "UA",      "ukrainians": "UA",
    "saudi": "SA",          "saudis": "SA",
    "turkish": "TR",        "turk": "TR",
    "japanese": "JP",
    "korean": "KR",
    "north korean": "KP",
    "south korean": "KR",
    "australian": "AU",     "australians": "AU",
    "canadian": "CA",       "canadians": "CA",
    "afghan": "AF",         "afghans": "AF",
    "iraqi": "IQ",          "iraqis": "IQ",
    "syrian": "SY",         "syrians": "SY",
    "yemeni": "YE",         "yemenis": "YE",
    "libyan": "LY",         "libyans": "LY",
    "egyptian": "EG",       "egyptians": "EG",
    "ethiopian": "ET",      "ethiopians": "ET",
    "nigerian": "NG",       "nigerians": "NG",
    "taiwanese": "TW",
    "vietnamese": "VN",
    "thai": "TH",
    "indonesian": "ID",     "indonesians": "ID",
    "philippine": "PH",     "filipino": "PH",
    "malaysian": "MY",
    "bangladeshi": "BD",
    "sri lankan": "LK",
    "nepalese": "NP",       "nepali": "NP",
    "burmese": "MM",        "myanmar": "MM",
    "venezuelan": "VE",     "venezuelans": "VE",
    "colombian": "CO",      "colombians": "CO",
    "brazilian": "BR",      "brazilians": "BR",
    "argentinian": "AR",    "argentine": "AR",
    "mexican": "MX",        "mexicans": "MX",
    "european": "BLOC:EU",  # demonym maps to bloc, not a single country

    # ── Regional blocs & supranational bodies ─────────────────────────────
    # These intentionally get a BLOC: prefix instead of a country ISO code.
    # Phase 6 aggregation handles BLOC: entries separately from country scores.
    "eu": "BLOC:EU",                "european union": "BLOC:EU",
    "nato": "BLOC:NATO",            "north atlantic treaty organization": "BLOC:NATO",
                                    "north atlantic treaty organisation": "BLOC:NATO",
    "asean": "BLOC:ASEAN",
    "un": "BLOC:UN",                "united nations": "BLOC:UN",
    "g7": "BLOC:G7",                "g-7": "BLOC:G7",
    "g20": "BLOC:G20",              "g-20": "BLOC:G20",
    "au": "BLOC:AU",                "african union": "BLOC:AU",
    "gcc": "BLOC:GCC",              "gulf cooperation council": "BLOC:GCC",
    "sco": "BLOC:SCO",              "shanghai cooperation organisation": "BLOC:SCO",
                                    "shanghai cooperation organization": "BLOC:SCO",
    "brics": "BLOC:BRICS",
    "arab league": "BLOC:ARABLEAGUE",
    "oecd": "BLOC:OECD",
    "imf": "BLOC:IMF",              "international monetary fund": "BLOC:IMF",
    "world bank": "BLOC:WORLDBANK",
    "wto": "BLOC:WTO",              "world trade organization": "BLOC:WTO",
    "who": "BLOC:WHO",              "world health organization": "BLOC:WHO",
    "iaea": "BLOC:IAEA",            "international atomic energy agency": "BLOC:IAEA",
    "opcw": "BLOC:OPCW",
    "interpol": "BLOC:INTERPOL",
}

# Path for the validation JSONL template
VALIDATION_DIR = Path(__file__).parent / "validation"
VALIDATION_PATH = VALIDATION_DIR / "ner_validation.jsonl"
METRICS_PATH = Path(__file__).parent / "metrics.json"


# ─────────────────────────────────────────────────────────────────────────────
# Entity normaliser
# ─────────────────────────────────────────────────────────────────────────────

def normalize_entity(text: str, label: str) -> str | None:
    """
    Map a raw entity surface form to an ISO 3166-1 alpha-2 code or BLOC:xxx.

    Returns None if the entity cannot be meaningfully mapped to a country or
    bloc (e.g., an ORG like "Goldman Sachs", or a NORP like "Republicans").

    Priority:
      1. MANUAL_MAP   (exact lowercase match — handles abbreviations, metonyms,
                        demonyms, and blocs not in ISO 3166-1)
      2. pycountry    (ISO-standardised full country names via library lookup)
      3. None         (unmappable — stored in DB with canonical=NULL)

    Why pycountry as fallback rather than primary?
      pycountry does string matching against official ISO names only. It will
      find "Ukraine" (official name) but not "U.S." (abbreviation), "American"
      (demonym), or "Washington" (metonym). The manual map handles everything
      that pycountry misses.
    """
    key = text.strip().lower()

    # Step 1: manual map (fastest, most precise)
    if key in MANUAL_MAP:
        return MANUAL_MAP[key]

    # Step 2: pycountry lookup — only attempt for GPE/NORP, not plain ORG.
    # ORG entities like "the United Nations" are already in MANUAL_MAP;
    # falling through to pycountry for ORGs would cause false positives
    # (e.g., "General Motors" accidentally matching "Gambia").
    if label in ("GPE", "NORP"):
        try:
            import pycountry
            country = pycountry.countries.lookup(text)
            return country.alpha_2
        except LookupError:
            pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# spaCy model loading

# Using en_core_web_lg instead of en_core_web_trf.
# Reason: en_core_web_trf fails on Python 3.13 + Windows due to
# blis Cython compilation error, and HuggingFace runtime download
# hangs on this network. en_core_web_lg (85.5% F1 vs 90.1%) is
# sufficient for wire-service news text.
# To switch: change model name below + pip install en_core_web_trf
# (requires Python 3.11/3.12 + unrestricted HuggingFace access)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = "en_core_web_lg"):
    """
    Load the spaCy pipeline, disabling components we don't need for NER.

    For en_core_web_lg:
      We keep:   tok2vec (shared encoder), ner
      Disable:   tagger, parser, senter, attribute_ruler, lemmatizer

    For en_core_web_trf (Python 3.11/3.12 only):
      We keep:   transformer (backbone), ner
      Disable:   tagger, parser, senter, attribute_ruler, lemmatizer

    Disabling unused components reduces memory by ~15-20% and speeds up
    processing by ~30% (no dependency parsing tree construction).

    We safely check which components exist before disabling to support
    both the lg and trf model families.
    """
    # ── Compatibility patch for spacy_transformers with transformers >= 4.49/5.x ──
    try:
        import transformers.tokenization_utils_base
        import transformers.tokenization_utils
        if not hasattr(transformers.tokenization_utils, "BatchEncoding"):
            transformers.tokenization_utils.BatchEncoding = transformers.tokenization_utils_base.BatchEncoding
    except Exception:
        pass

    import spacy

    log.info("Loading spaCy model: %s", model_name)
    try:
        nlp = spacy.load(model_name)
    except OSError:
        log.error(
            "Model '%s' not found. Install with:\n"
            "  python -m spacy download %s",
            model_name, model_name,
        )
        sys.exit(1)

    # Disable components we don't need — safe: we only use NER output.
    _can_disable = {"tagger", "parser", "senter", "attribute_ruler", "lemmatizer"}
    to_disable = [name for name in nlp.pipe_names if name in _can_disable]
    if to_disable:
        nlp.disable_pipes(to_disable)
        log.info("Disabled pipeline components: %s", to_disable)

    log.info("Active pipeline: %s", nlp.pipe_names)
    return nlp


# ─────────────────────────────────────────────────────────────────────────────
# Text construction
# ─────────────────────────────────────────────────────────────────────────────

def article_to_text(article) -> str:
    """Combine title and summary into a single string for NER processing.

    The space separator ensures a word on the title's last character and
    the summary's first character don't get merged into one token.
    Character offsets in the entities table are relative to this combined text.
    """
    parts = []
    if article["title"]:
        parts.append(article["title"].strip())
    if article["summary"]:
        parts.append(article["summary"].strip())
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Entity extraction from a spaCy Doc
# ─────────────────────────────────────────────────────────────────────────────

def extract_entities_from_doc(doc, article_id: str) -> list[dict]:
    """
    Extract GPE/ORG/NORP entities from a spaCy Doc object.

    Returns a list of dicts ready for DB insertion. Dedup is applied on
    (article_id, text_lower, label) — only the first occurrence of each
    unique (surface form, label) pair per article is kept.

    Why deduplicate?
      An article about Russia may mention "Russia" 15 times. Storing 15
      identical rows inflates entity counts and distorts Phase 6 aggregation
      (the tension score shouldn't be 15x higher just because a journalist
      was repetitive).
    """
    seen: set[tuple[str, str]] = set()  # (text_lower, label)
    results = []

    for ent in doc.ents:
        if ent.label_ not in TARGET_LABELS:
            continue

        text_raw = ent.text.strip()
        if not text_raw:
            continue

        dedup_key = (text_raw.lower(), ent.label_)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        canonical = normalize_entity(text_raw, ent.label_)

        results.append({
            "article_id": article_id,
            "text":       text_raw,
            "label":      ent.label_,
            "canonical":  canonical,
            "start_char": ent.start_char,
            "end_char":   ent.end_char,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DB insertion
# ─────────────────────────────────────────────────────────────────────────────

def insert_entities(conn, entities: list[dict]) -> int:
    """Bulk-insert entities in a single transaction. Returns row count."""
    if not entities:
        return 0
    with conn:
        conn.executemany(
            """
            INSERT INTO entities (article_id, text, label, canonical, start_char, end_char)
            VALUES (:article_id, :text, :label, :canonical, :start_char, :end_char)
            """,
            entities,
        )
    return len(entities)


# ─────────────────────────────────────────────────────────────────────────────
# Main NER pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_ner(
    batch_size: int = 16,
    limit: int | None = None,
    rerun: bool = False,
    model_name: str = "en_core_web_lg",
) -> dict:
    """
    Process all unprocessed articles, extract NER entities, store in DB.

    Parameters
    ----------
    batch_size : int
        Number of texts to pass to nlp.pipe() at once.
    limit : int or None
        Cap the number of articles to process (testing).
    rerun : bool
        If True, delete all existing entity rows before processing.
    model_name : str
        spaCy model to load.

    Returns
    -------
    stats : dict
        Summary counts for metrics logging.
    """
    init_db()
    migrate_db()
    conn = get_connection()

    if rerun:
        log.warning("--rerun specified: deleting all existing entity rows.")
        with conn:
            conn.execute("DELETE FROM entities")

    articles = get_unprocessed_articles(conn, "entities", limit=limit)
    total = len(articles)
    log.info("Articles to process: %d", total)

    if total == 0:
        log.info("Nothing to do -- all articles have entity rows.")
        conn.close()
        return {"articles_processed": 0, "entities_stored": 0}

    nlp = load_model(model_name)

    # Build (article_id, text) pairs for nlp.pipe()
    pairs = [(art["id"], article_to_text(art)) for art in articles]
    ids   = [p[0] for p in pairs]
    texts = [p[1] for p in pairs]

    label_counts: dict[str, int] = {"GPE": 0, "ORG": 0, "NORP": 0}
    total_entities = 0
    mapped_count   = 0  # entities with a non-NULL canonical

    log.info("Running NER with batch_size=%d ...", batch_size)
    for i, (article_id, doc) in enumerate(
        zip(ids, nlp.pipe(texts, batch_size=batch_size)), start=1
    ):
        entities = extract_entities_from_doc(doc, article_id)
        insert_entities(conn, entities)

        for e in entities:
            label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
            if e["canonical"] is not None:
                mapped_count += 1
        total_entities += len(entities)

        if i % 50 == 0 or i == total:
            log.info("  Processed %d / %d articles ...", i, total)

    conn.close()

    stats = {
        "articles_processed": total,
        "entities_stored":    total_entities,
        "by_label":           label_counts,
        "canonical_mapped":   mapped_count,
        "unmapped":           total_entities - mapped_count,
    }

    # Print summary table
    print("\n" + "=" * 60)
    print("NER SUMMARY")
    print("=" * 60)
    rows = [
        ["Articles processed", total],
        ["Total entities stored", total_entities],
        ["GPE entities", label_counts.get("GPE", 0)],
        ["ORG entities", label_counts.get("ORG", 0)],
        ["NORP entities", label_counts.get("NORP", 0)],
        ["Canonical mapped", mapped_count],
        ["Unmapped (canonical=NULL)", total_entities - mapped_count],
    ]
    print(tabulate(rows, tablefmt="rounded_outline"))
    print("=" * 60 + "\n")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Validation template generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_validation_template(
    n: int = 25,
    model_name: str = "en_core_web_lg",
    output_path: Path = VALIDATION_PATH,
) -> None:
    """
    Sample n articles from the DB, run NER, and write a JSONL template.

    Each line is a JSON object:
    {
      "article_id": "<sha256>",
      "source": "BBC World",
      "text": "<title> <summary>",
      "entities": [
        {
          "text": "Ukraine",
          "label": "GPE",
          "canonical": "UA",
          "start_char": 8,
          "end_char": 15
        }, ...
      ]
    }

    The file is your ground-truth validation set.  Review each article and:
      - Remove false positive entities (ones the model predicted incorrectly)
      - Add false negative entities (ones the model missed)
      - Correct wrong labels (e.g., model said ORG but it should be GPE)
      - Correct wrong canonical codes (e.g., "Washington" → "US" is correct,
        but "Washington" → "GB" would be wrong)

    After editing, run:
      python evaluate.py --phase ner --input validation/ner_validation.jsonl
    """
    init_db()
    migrate_db()
    conn = get_connection()

    # Sample articles spread across sources for diversity
    rows = conn.execute(
        """
        SELECT * FROM articles
        WHERE title IS NOT NULL AND summary IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,),
    ).fetchall()
    conn.close()

    if not rows:
        log.error("No articles in DB. Run ingest.py first.")
        return

    nlp = load_model(model_name)
    texts = [article_to_text(r) for r in rows]

    VALIDATION_DIR.mkdir(exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for art, doc in zip(rows, nlp.pipe(texts, batch_size=8)):
            entities = extract_entities_from_doc(doc, art["id"])
            record = {
                "article_id": art["id"],
                "source":     art["source"],
                "text":       article_to_text(art),
                "entities":   [
                    {
                        "text":       e["text"],
                        "label":      e["label"],
                        "canonical":  e["canonical"],
                        "start_char": e["start_char"],
                        "end_char":   e["end_char"],
                    }
                    for e in entities
                ],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info(
        "Validation template written: %s  (%d articles)",
        output_path, written,
    )
    print(f"\n[OK] Validation template saved to: {output_path}")
    print(f"     {written} articles, each with model-predicted entity labels.")
    print("     Review the file, correct any mistakes, then run:")
    print("       python evaluate.py --phase ner\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: spaCy NER pipeline for OSINT articles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ner.py                          # Process all unprocessed articles
  python ner.py --limit 50              # Process first 50 only
  python ner.py --rerun                 # Reprocess all (deletes existing rows)
  python ner.py --generate-validation   # Create validation/ner_validation.jsonl
  python ner.py --generate-validation --val-size 30
""",
    )
    parser.add_argument(
        "--generate-validation",
        action="store_true",
        help="Generate a pre-filled JSONL validation template and exit.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=25,
        metavar="N",
        help="Number of articles to include in the validation template (default: 25).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N articles (for testing).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        dest="batch_size",
        help="nlp.pipe() batch size (default: 16).",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Delete existing entity rows and reprocess all articles.",
    )
    parser.add_argument(
        "--model",
        default="en_core_web_lg",
        help="spaCy model name (default: en_core_web_lg; use en_core_web_trf on Python 3.11/3.12).",
    )
    args = parser.parse_args()

    if args.generate_validation:
        generate_validation_template(
            n=args.val_size,
            model_name=args.model,
        )
    else:
        run_ner(
            batch_size=args.batch_size,
            limit=args.limit,
            rerun=args.rerun,
            model_name=args.model,
        )
