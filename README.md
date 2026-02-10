# Buch-Scan

Ein kleines Python-Skript, das Buchfotos automatisch in saubere Seiten umwandelt.

## Was es macht
- erkennt Doppelseiten
- schneidet Seiten zu
- korrigiert Perspektive
- verbessert Licht und Lesbarkeit
- speichert Ergebnisse als JPG oder als eine PDF

## Nutzung
1. In `process.py` die Pfade `INPUT_DIR` und `OUTPUT_DIR` anpassen.
2. Optional den Toggle `OUTPUT_AS_PDF` setzen:
   - `False`: Ausgabe als einzelne JPG-Dateien (Standard)
   - `True`: Alle Seiten in der richtigen Reihenfolge in `PDF_FILENAME` als eine PDF
3. Benötigte Pakete installieren:
   ```bash
   pip install opencv-python numpy pillow
   ```
4. Script starten:
   ```bash
   python process.py
   ```

Die fertigen Seiten werden im `OUTPUT_DIR` gespeichert.
