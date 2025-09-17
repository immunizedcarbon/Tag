# DIP × Gemini – Analyse-Workbench

Dieses Repository enthält eine Streamlit-Anwendung zum Durchsuchen der Bundestags-Datenbank DIP
und zur Übergabe der Ergebnisse an Googles Gemini-Modelle für Zusammenfassungen und Analysen.

## Features
- Komfortable Suche in den wichtigsten DIP-Endpunkten inkl. Filter und Cursor-Paginierung
- Vorauswahl relevanter Dokumente und Zusammenstellung eines Textkontexts
- Übergabe der Texte an Gemini 2.5 (Pro/Flash/Flash Lite) für verschiedene Aufgaben wie Zusammenfassung,
  Stichpunkte, Kernaussagen oder Übersetzung
- Persistente Konfiguration (API-Keys, Prompt, Ausgabesprache, Speicherort) unter `~/.config/dip-gemini/config.json`
- Export der Ergebnisse als Markdown-Datei

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U streamlit requests pydantic google-genai pandas
```

## Nutzung
```bash
streamlit run dip_gemini_app.py
```

Hinterlegen Sie im Reiter **Einstellungen** Ihre DIP- und Gemini-API-Keys. Die Daten werden ausschließlich
lokal gespeichert.

## Hinweise
- Persönliche DIP-API-Keys erhalten Sie unter <https://dip.bundestag.de/über-dip/hilfe/api>
- Den Gemini-API-Key erhalten Sie im Google AI Studio.
- Die Anwendung wurde unter Python 3.10+ getestet.
