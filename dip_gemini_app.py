"""DIP × Gemini – Analyse-Workbench (Single-File-App)
==================================================

Modernes, effizientes, state-of-the-art UI auf Basis von Streamlit.
Bezieht Daten aus der Bundestag DIP API und fasst ausgewählten Text mit Gemini 2.5 zusammen.

Funktionen auf einen Blick
--------------------------
- Browsen der DIP-Endpunkte: Vorgang, Vorgangsposition, Drucksache, Drucksache-Text,
  Plenarprotokoll, Plenarprotokoll-Text, Aktivität, Person
- Filter, Suche, Paginierung (Cursor), Vorschau
- Auswahl von Dokumenten/Text und Übergabe mit Button an den „Gemini“-Reiter
- Gemini-Reiter mit Aufgabenwahl: Zusammenfassen, Stichpunkte, Kernaussagen, Übersetzen
- Konfigurations-Reiter: API-Keys (DIP & Gemini), System-Prompt, Modellwahl, Speicherort
- Persistente Einstellungen unter ~/.config/dip-gemini/config.json
- Ergebnisse als Markdown speichern

Getestet für Kubuntu 24.04 (Python 3.10+)

Installation
------------
python -m venv .venv && source .venv/bin/activate
pip install -U streamlit requests pydantic google-genai

Start
-----
streamlit run dip_gemini_app.py

Hinweise
--------
- Für die DIP API benötigen Sie einen persönlichen API-Key:
  https://dip.bundestag.de/über-dip/hilfe/api
- Für Gemini 2.5 Pro benötigen Sie einen Gemini API-Key aus Google AI Studio.
- Keys werden **nur lokal** in Ihrer Config-Datei gespeichert.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# Gemini SDK
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover – SDK noch nicht installiert
    genai = None
    genai_types = None

# ----------------------
# Konstanten & Pfade
# ----------------------
APP_TITLE = "DIP × Gemini – Analyse-Workbench"
DIP_BASE_URL = "https://search.dip.bundestag.de/api/v1"
CONFIG_DIR = Path.home() / ".config" / "dip-gemini"
CONFIG_PATH = CONFIG_DIR / "config.json"
DEFAULT_SAVE_DIR = Path.home() / "DIP-Gemini-Analysen"

# ----------------------
# Modelle (Config)
# ----------------------
class AppConfig(BaseModel):
    dip_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    gemini_model: str = Field(default="gemini-2.5-pro")
    system_prompt: str = Field(
        default=(
            "Du bist ein präziser, sachlicher Assistent für deutscher Parlamentsdokumente. "
            "Fasse klar, neutral und komprimiert zusammen, nenne Kernaussagen und nützliche Stichpunkte. "
            "Wenn Zitate nötig sind, gib kurze, relevante Ausschnitte mit Absatz-/Seitenhinweis."
        )
    )
    save_dir: str = Field(default=str(DEFAULT_SAVE_DIR))
    default_language: str = Field(default="Deutsch")

    @staticmethod
    def load() -> "AppConfig":
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return AppConfig(**raw)
        except Exception:
            pass
        # Fallback: Umgebungsvariablen
        return AppConfig(
            dip_api_key=os.environ.get("DIP_API_KEY", ""),
            gemini_api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", ""),
        )

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)


# ----------------------
# DIP API Client
# ----------------------
class DipClient:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"ApiKey {self.api_key}"})

    def _get(self, path: str, params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        url = f"{DIP_BASE_URL}{path}"
        try:
            r = self.session.get(url, params=params, timeout=30)
            if r.status_code == 401:
                return None, "401 – Ungültiger oder fehlender DIP API-Key."
            if r.status_code >= 400:
                return None, f"HTTP {r.status_code}: {r.text[:500]}"
            return r.json(), None
        except requests.RequestException as e:
            return None, f"Netzwerkfehler: {e}"

    def search(self, endpoint: str, filters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, str, Optional[str]]:
        """Gibt (documents, numFound, cursor, error) zurück."""
        params: Dict[str, Any] = {"format": "json"}
        # Filter korrekt einsetzen – Listenwerte als Liste belassen
        for k, v in filters.items():
            if v in (None, ""):
                continue
            params[k] = v
        data, err = self._get(endpoint, params)
        if err:
            return [], 0, "", err
        docs = data.get("documents", []) if isinstance(data, dict) else []
        num = data.get("numFound", 0)
        cur = data.get("cursor", "")
        return docs, num, cur, None

    def get_by_id(self, endpoint: str, _id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return self._get(f"{endpoint}/{_id}", params={"format": "json"})


# ----------------------
# Gemini Client Wrapper
# ----------------------
class GeminiClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key.strip()
        self.model = model
        if genai is None:
            raise RuntimeError(
                "Das Paket 'google-genai' ist nicht installiert. Führen Sie 'pip install google-genai' aus."
            )
        self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()

    def generate(self, prompt: str, system_instruction: str, mime: str = "text/markdown") -> str:
        try:
            cfg = (
                genai_types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    response_mime_type=mime,
                )
                if genai_types is not None
                else None
            )
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=cfg,
            )
            text = getattr(resp, "text", None)
            if not text:
                # Robustheit bei SDK-Versionen
                text = getattr(resp, "output_text", None) or str(resp)
            return text
        except Exception as e:
            raise RuntimeError(f"Gemini-Fehler: {e}")


# ----------------------
# Hilfsfunktionen
# ----------------------
TASK_OPTIONS = [
    "Zusammenfassen",
    "Stichpunkte",
    "Kernaussagen",
    "Übersetzen",
]

LANG_OPTIONS = [
    "Deutsch",
    "Englisch",
    "Französisch",
    "Spanisch",
    "Italienisch",
]

ENDPOINTS = {
    "Vorgänge (Liste)": "/vorgang",
    "Vorgangspositionen (Liste)": "/vorgangsposition",
    "Drucksachen (Liste)": "/drucksache",
    "Drucksache – Volltext (Liste)": "/drucksache-text",
    "Plenarprotokolle (Liste)": "/plenarprotokoll",
    "Plenarprotokoll – Volltext (Liste)": "/plenarprotokoll-text",
    "Aktivitäten (Liste)": "/aktivitaet",
    "Personen (Liste)": "/person",
}

# Felder für Tabellenansicht je Endpunkt (leichtgewichtig, nur die wichtigsten)
TABLE_FIELDS = {
    "/vorgang": ["id", "wahlperiode", "vorgangstyp", "titel", "datum", "aktualisiert"],
    "/vorgangsposition": ["id", "vorgangsposition", "vorgangstyp", "titel", "datum"],
    "/drucksache": ["id", "dokumentnummer", "drucksachetyp", "titel", "datum"],
    "/drucksache-text": ["id", "dokumentnummer", "drucksachetyp", "titel", "datum"],
    "/plenarprotokoll": ["id", "dokumentnummer", "titel", "datum"],
    "/plenarprotokoll-text": ["id", "dokumentnummer", "titel", "datum"],
    "/aktivitaet": ["id", "aktivitaetsart", "titel", "datum", "wahlperiode"],
    "/person": ["id", "titel", "vorname", "nachname", "datum"],
}


def short(s: Any, n: int = 140) -> str:
    t = str(s) if s is not None else ""
    return t if len(t) <= n else t[: n - 1] + "…"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------
# Streamlit – Seitenaufbau
# ----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Custom CSS für moderneres Look & Feel
st.markdown(
    """
    <style>
    .app-header {font-size: 1.6rem; font-weight: 700; margin: .3rem 0 1rem 0}
    .muted {opacity: .7}
    .codebox {background: var(--secondary-background-color); padding: .5rem .75rem; border-radius: .5rem;}
    .small {font-size: .9rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='app-header'>📜 {APP_TITLE}</div>", unsafe_allow_html=True)

# Lade/Initialisiere Konfiguration
if "config" not in st.session_state:
    st.session_state.config = AppConfig.load()
config: AppConfig = st.session_state.config

# Clients (lazy)
_dip_client: Optional[DipClient] = None

def dip_client() -> DipClient:
    global _dip_client
    if _dip_client is None:
        _dip_client = DipClient(config.dip_api_key)
    return _dip_client

# Session-Zwischenspeicher
if "dip_results" not in st.session_state:
    st.session_state.dip_results = []  # type: List[Dict[str, Any]]
if "dip_num_found" not in st.session_state:
    st.session_state.dip_num_found = 0
if "dip_cursor" not in st.session_state:
    st.session_state.dip_cursor = ""
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = set()  # type: set
if "prepared_text" not in st.session_state:
    st.session_state.prepared_text = ""
if "gemini_output" not in st.session_state:
    st.session_state.gemini_output = ""
if "last_endpoint" not in st.session_state:
    st.session_state.last_endpoint = None

# Tabs
search_tab, selection_tab, gemini_tab, settings_tab = st.tabs([
    "🔎 Suche",
    "✅ Auswahl & Vorschau",
    "✨ Gemini",
    "⚙️ Einstellungen",
])

# ----------------------
# Einstellungen
# ----------------------
with settings_tab:
    st.subheader("Allgemeine Einstellungen")
    c1, c2 = st.columns(2)
    with c1:
        config.dip_api_key = st.text_input(
            "DIP API-Key", value=config.dip_api_key, type="password", help="Wird im Authorization-Header als 'ApiKey <key>' gesendet."
        )
        config.gemini_api_key = st.text_input(
            "Gemini API-Key", value=config.gemini_api_key, type="password", help="Wird lokal gespeichert und nur serverseitig verwendet."
        )
        config.gemini_model = st.selectbox(
            "Gemini Modell", options=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"], index=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"].index(config.gemini_model) if config.gemini_model in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"] else 0,
        )
    with c2:
        config.default_language = st.selectbox("Bevorzugte Ausgabesprache", options=LANG_OPTIONS, index=max(0, LANG_OPTIONS.index(config.default_language) if config.default_language in LANG_OPTIONS else 0))
        config.save_dir = st.text_input("Speicherordner für Ergebnisse", value=config.save_dir)
    config.system_prompt = st.text_area("System-Prompt für Gemini", value=config.system_prompt, height=140)
    st.caption("Tipp: Halten Sie den System-Prompt knapp, präzise und auf Ihren Anwendungsfall zugeschnitten.")

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("💾 Einstellungen speichern", use_container_width=True):
            try:
                # Reset DIP Client bei Key-Änderung
                _dip_client = DipClient(config.dip_api_key)
                ensure_dir(Path(config.save_dir))
                config.save()
                st.success(f"Gespeichert unter {CONFIG_PATH}")
            except ValidationError as e:
                st.error(f"Validierungsfehler: {e}")
    with colB:
        if st.button("📂 Speicherordner öffnen", use_container_width=True):
            st.write(Path(config.save_dir).as_posix())

# ----------------------
# Suche (DIP)
# ----------------------
with search_tab:
    st.subheader("DIP durchsuchen")

    left, right = st.columns([1.2, 1])
    with left:
        selected_endpoint_title = st.selectbox("Endpunkt", list(ENDPOINTS.keys()))
        endpoint = ENDPOINTS[selected_endpoint_title]

        with st.expander("Filter", expanded=True):
            # Gemeinsame Filter (siehe OpenAPI)
            col1, col2, col3 = st.columns(3)
            with col1:
                f_datum_start = st.date_input("f.datum.start", value=None, format="YYYY-MM-DD")
                f_akt_start = st.text_input("f.aktualisiert.start (YYYY-MM-DDThh:mm:ss)")
                f_wahlperiode = st.text_input("f.wahlperiode (kommasepariert)")
            with col2:
                f_datum_end = st.date_input("f.datum.end", value=None, format="YYYY-MM-DD")
                f_akt_end = st.text_input("f.aktualisiert.end (YYYY-MM-DDThh:mm:ss)")
                f_titel = st.text_input("f.titel (ODER – mehrere durch \n trennen)")
            with col3:
                f_dokumentnummer = st.text_input("f.dokumentnummer (ODER – mehrere durch \n)")
                f_vorgangstyp = st.text_input("f.vorgangstyp (ODER – mehrere durch \n)")
                f_deskriptor = st.text_input("f.deskriptor (UND – mehrere durch \n)")

            # Endpunkt-spezifische Filter (leichtgewichtig)
            if endpoint in ("/drucksache", "/drucksache-text"):
                colx, coly = st.columns(2)
                with colx:
                    f_drucksachetyp = st.text_input("f.drucksachetyp (z.B. Antrag, Gesetzentwurf)")
                    f_urheber = st.text_input("f.urheber (UND – mehrere durch \n)")
                with coly:
                    f_ressort = st.text_input("f.ressort_fdf (UND – mehrere durch \n)")
                    f_initiative = st.text_input("f.initiative (UND – mehrere durch \n)")
            else:
                f_drucksachetyp = f_urheber = f_ressort = f_initiative = ""

            if endpoint == "/aktivitaet":
                colp, colq = st.columns(2)
                with colp:
                    f_person = st.text_input("f.person (ODER – mehrere durch \n)")
                with colq:
                    f_person_id = st.text_input("f.person_id (ODER – mehrere durch \n)")
            else:
                f_person = f_person_id = ""

            # Paginierung
            per_page = st.slider("Max. pro Anfrage", 10, 100, 50, help="Die API liefert maximal 100 Elemente pro Anfrage.")

        # Such- und Mehr-Laden-Buttons
        cols = st.columns([1, 1, 2])
        with cols[0]:
            start_search = st.button("🔎 Suchen", type="primary", use_container_width=True)
        with cols[1]:
            load_more = st.button("➕ Mehr laden", use_container_width=True)

    with right:
        st.info(
            "Tipp: Für Volltexte nutzen Sie **Drucksache – Volltext** oder **Plenarprotokoll – Volltext**.\n\n"
            "Markieren Sie in der Tabelle unten die gewünschten Einträge und wechseln Sie anschließend zum Reiter **Auswahl & Vorschau**."
        )

    def parse_multiline(s: str) -> Optional[List[str]]:
        s = (s or "").strip()
        if not s:
            return None
        return [line.strip() for line in s.splitlines() if line.strip()]

    def parse_csv_ints(s: str) -> Optional[List[int]]:
        s = (s or "").strip()
        if not s:
            return None
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                pass
        return out or None

    # Filters zusammenstellen
    filters: Dict[str, Any] = {"cursor": None, "limit": per_page}
    if f_datum_start:
        filters["f.datum.start"] = f_datum_start.isoformat()
    if f_datum_end:
        filters["f.datum.end"] = f_datum_end.isoformat()
    if f_akt_start:
        filters["f.aktualisiert.start"] = f_akt_start
    if f_akt_end:
        filters["f.aktualisiert.end"] = f_akt_end
    if f_wahlperiode:
        wp = parse_csv_ints(f_wahlperiode)
        if wp:
            filters["f.wahlperiode"] = wp
    if f_titel:
        filters["f.titel"] = parse_multiline(f_titel)
    if f_dokumentnummer:
        filters["f.dokumentnummer"] = parse_multiline(f_dokumentnummer)
    if f_vorgangstyp:
        filters["f.vorgangstyp"] = parse_multiline(f_vorgangstyp)
    if f_deskriptor:
        filters["f.deskriptor"] = parse_multiline(f_deskriptor)
    if f_drucksachetyp:
        filters["f.drucksachetyp"] = f_drucksachetyp
    if f_urheber:
        filters["f.urheber"] = parse_multiline(f_urheber)
    if f_ressort:
        filters["f.ressort_fdf"] = parse_multiline(f_ressort)
    if f_initiative:
        filters["f.initiative"] = parse_multiline(f_initiative)
    if f_person:
        filters["f.person"] = parse_multiline(f_person)
    if f_person_id:
        try:
            filters["f.person_id"] = [int(x) for x in parse_multiline(f_person_id) or []]
        except Exception:
            pass

    # Suche ausführen
    if start_search:
        st.session_state.selected_ids = set()
        st.session_state.last_endpoint = endpoint
        st.session_state.dip_results, st.session_state.dip_num_found, st.session_state.dip_cursor, err = dip_client().search(
            endpoint, {**filters}
        )
        if err:
            st.error(err)
        else:
            st.success(f"Gefunden: {st.session_state.dip_num_found} Einträge (erste {len(st.session_state.dip_results)})")

    if load_more:
        cur = st.session_state.dip_cursor
        if not cur:
            st.warning("Kein weiterer Cursor vorhanden.")
        else:
            more, _, next_cursor, err = dip_client().search(endpoint, {**filters, "cursor": cur})
            if err:
                st.error(err)
            else:
                st.session_state.dip_results.extend(more)
                st.session_state.dip_cursor = next_cursor
                st.info(f"Weitere {len(more)} geladen. Cursor: {short(next_cursor, 60)}")

    # Tabelle rendern
    docs = st.session_state.dip_results
    if docs:
        st.markdown("### Treffer")
        fields = TABLE_FIELDS.get(endpoint, list(docs[0].keys()))

        # Build Table
        import pandas as pd

        rows = []
        for d in docs[: per_page]:  # Sichtbare Auswahl begrenzen
            row = {k: short(d.get(k, ""), 240) for k in fields}
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("**Auswahl**")
        # Einfaches Auswahl-UI (IDs ankreuzen)
        id_col1, id_col2 = st.columns(2)
        with id_col1:
            ids_str = st.text_input(
                "IDs zum Auswählen (kommasepariert)",
                value=",".join(sorted({str(x.get("id")) for x in docs[: per_page]})),
            )
            if st.button("✓ Diese IDs auswählen/aktualisieren"):
                selected = {int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()}
                st.session_state.selected_ids = selected
        with id_col2:
            st.write("Aktuell ausgewählt:", ", ".join(str(x) for x in sorted(st.session_state.selected_ids)))

        # Vorschau einzelner Einträge
        st.markdown("### Vorschau Einzelansicht")
        preview_default = int(next(iter(st.session_state.selected_ids), docs[0].get("id", 0)))
        preview_id = st.number_input("ID für Vorschau", min_value=0, step=1, value=preview_default)
        if preview_id:
            preview_data = None
            err: Optional[str] = None
            if endpoint.endswith("-text"):
                preview_data, err = dip_client().get_by_id(endpoint, preview_id)
            else:
                try:
                    preview_data, err = dip_client().get_by_id(endpoint, preview_id)
                except Exception as e:  # pragma: no cover - defensive
                    err = str(e)
            if err:
                st.error(err)
            elif preview_data:
                st.json(preview_data)

# ----------------------
# Auswahl & Vorschau – Text zusammenstellen
# ----------------------
with selection_tab:
    st.subheader("Auswahl & Textvorbereitung")
    st.caption("Wählen Sie unten, wie der Text zusammengestellt werden soll. Für Volltexte eignen sich die *-Text Endpunkte.")

    if not st.session_state.selected_ids:
        st.warning("Keine IDs ausgewählt. Wählen Sie in der Suche Einträge aus.")
    else:
        mode = st.radio("Quellenpriorität", ["Volltext bevorzugen (drucksache-text/plenarprotokoll-text)", "Nur Metadaten/Titel"], horizontal=False)
        max_chars = st.slider("Maximale Zeichen für Zusammenstellung", 2_000, 300_000, 60_000, step=1_000)
        joiner = st.text_input("Trenner zwischen Dokumenten", value="\n\n---\n\n")

        if st.button("🧩 Text aus Auswahl zusammenstellen", type="primary"):
            compiled_parts: List[str] = []
            current_endpoint = st.session_state.last_endpoint

            for _id in sorted(st.session_state.selected_ids):
                text_blob = None
                if mode.startswith("Volltext"):
                    preferred_endpoints: List[str] = []
                    if current_endpoint and current_endpoint.endswith("-text"):
                        preferred_endpoints.append(current_endpoint)
                    preferred_endpoints.extend(["/drucksache-text", "/plenarprotokoll-text"])
                    for ep in preferred_endpoints:
                        data, err = dip_client().get_by_id(ep, _id)
                        if data and not err and "text" in data:
                            title = data.get("titel") or data.get("fundstelle", {}).get("dokumentnummer") or f"Dokument {_id}"
                            head = f"## {title} (ID {data.get('id')})\n\n"
                            text_blob = head + data.get("text", "").strip()
                            break
                if not text_blob:
                    for ep in ("/drucksache", "/plenarprotokoll", "/vorgang", "/vorgangsposition"):
                        data, err = dip_client().get_by_id(ep, _id)
                        if data and not err:
                            title = data.get("titel") or data.get("dokumentnummer") or f"ID {_id}"
                            abstract = data.get("abstract") or ""
                            text_blob = f"## {title} (ID {_id})\n\n{abstract}"
                            break
                if text_blob:
                    compiled_parts.append(text_blob)

            compiled = joiner.join(compiled_parts)
            if len(compiled) > max_chars:
                compiled = compiled[:max_chars] + "\n\n… (abgeschnitten)"
            st.session_state.prepared_text = compiled
            st.success(f"Text vorbereitet: {len(compiled):,} Zeichen")

        st.markdown("### Vorschau des vorbereiteten Texts")
        st.text_area("Prepared Text", value=st.session_state.prepared_text, height=320)
        if st.button("➡️ In den Gemini-Reiter übernehmen", type="secondary"):
            st.success("Text ist im Gemini-Reiter verfügbar.")

# ----------------------
# Gemini – Aufgaben & Ausführung
# ----------------------
with gemini_tab:
    st.subheader("Gemini – Analyse & Zusammenfassung")

    if not config.gemini_api_key:
        st.error("Bitte hinterlegen Sie Ihren Gemini API-Key unter *Einstellungen*.")
    else:
        task = st.selectbox("Aufgabe", TASK_OPTIONS, index=0)
        lang = st.selectbox("Zielsprache", LANG_OPTIONS, index=LANG_OPTIONS.index(config.default_language) if config.default_language in LANG_OPTIONS else 0)
        length = st.select_slider("Länge", options=["sehr kurz", "kurz", "mittel", "ausführlich"], value="kurz")
        include_key_points = st.checkbox("Kernaussagen als Liste beilegen", value=True)
        include_citations = st.checkbox("Absatz-/Seitenhinweise (falls vorhanden)", value=True)

        src_text = st.text_area("Eingabetext (von Auswahl-Reiter)", value=st.session_state.prepared_text, height=260)

        def build_prompt(task: str, lang: str, length: str, include_points: bool, include_refs: bool, system_prompt: str, text: str) -> str:
            instructions = []
            if task == "Zusammenfassen":
                instructions.append("Erstelle eine präzise Zusammenfassung.")
            elif task == "Stichpunkte":
                instructions.append("Erstelle kompakte Stichpunkte, jeweils ein prägnanter Punkt.")
            elif task == "Kernaussagen":
                instructions.append("Extrahiere die 5–12 wichtigsten Kernaussagen.")
            elif task == "Übersetzen":
                instructions.append("Übersetze den Text.")
            instructions.append(f"Zielsprache: {lang}.")
            instructions.append(f"Länge: {length}.")
            if include_points and task != "Stichpunkte":
                instructions.append("Füge zusätzlich eine Liste mit Bulletpoints hinzu.")
            if include_refs:
                instructions.append("Wenn eindeutig, gib knappe Absatz-/Seitenhinweise aus (z. B. Dokumentnummer + Abschnitt).")
            instructions.append("Antworte in Markdown.")

            prompt = (
                f"System: {system_prompt}\n\n"
                f"Aufgabe: {' '.join(instructions)}\n\n"
                f"Text:\n" + text
            )
            return prompt

        colx, coly = st.columns([1, 1])
        with colx:
            if st.button("🚀 An Gemini senden", type="primary", use_container_width=True):
                if not src_text.strip():
                    st.warning("Kein Eingabetext – bitte Text vorbereiten oder einfügen.")
                else:
                    try:
                        gcli = GeminiClient(config.gemini_api_key, config.gemini_model)
                        prompt = build_prompt(task, lang, length, include_key_points, include_citations, config.system_prompt, src_text)
                        with st.spinner("Gemini arbeitet…"):
                            out = gcli.generate(prompt, config.system_prompt)
                        st.session_state.gemini_output = out
                        st.success("Fertig – siehe Ergebnis unten.")
                    except Exception as e:
                        st.error(str(e))
        with coly:
            if st.button("📝 Ergebnis als Markdown speichern", use_container_width=True):
                if not st.session_state.gemini_output:
                    st.warning("Kein Ergebnis vorhanden.")
                else:
                    ensure_dir(Path(config.save_dir))
                    path = Path(config.save_dir) / "gemini_auswertung.md"
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.gemini_output)
                    st.success(f"Gespeichert: {path}")

        st.markdown("### Ergebnis")
        st.markdown(st.session_state.gemini_output or "*(noch nichts)*")

# Fuß
st.markdown(
    "<div class='muted small'>Quellen: Deutscher Bundestag – DIP API · Gemini 2.5 (Google GenAI SDK). Dieses Tool speichert Ihre Einstellungen lokal.</div>",
    unsafe_allow_html=True,
)
