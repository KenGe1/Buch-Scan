# Buch-Scan

Ein kleines Python-Skript, das Buchfotos automatisch in saubere Seiten umwandelt.

## Was es macht
- erkennt Doppelseiten
- schneidet Seiten zu
- korrigiert Perspektive
- verbessert Licht und Lesbarkeit
- speichert Ergebnisse als JPG

## Nutzung
1. In `process.py` die Pfade `INPUT_DIR` und `OUTPUT_DIR` anpassen.
2. Benötigte Pakete installieren:
   ```bash
   pip install opencv-python numpy
   ```
3. Script starten:
   ```bash
   python process.py
   ```

Die fertigen Seiten werden im `OUTPUT_DIR` gespeichert.
