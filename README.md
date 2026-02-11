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
   pip install opencv-python numpy pillow ultralytics
   ```
4. Script starten:
   ```bash
   python process.py
   ```

Die fertigen Seiten werden im `OUTPUT_DIR` gespeichert.

## YOLO für stabilere Seitenerkennung (optional)
Wenn die klassische Kontur-Erkennung bei deinen Fotos instabil ist, kannst du YOLO aktivieren:

1. In `process.py` diese Optionen setzen:
   - `ENABLE_YOLO_PAGE_DETECTION = True`
   - `YOLO_MODEL_PATH` auf dein Modell (z. B. `"yolov8n.pt"` oder ein trainiertes Buchseiten-Modell)
   - Optional `YOLO_TARGET_CLASSES` (Standard: `["book"]`)
   - Bei Fehl-Erkennungen zusätzlich diese Filter feinjustieren:
     - `YOLO_MIN_AREA_RATIO` (größere Werte filtern kleine/illustrationsartige Boxen)
     - `YOLO_MIN_MASK_COVERAGE` (größere Werte erzwingen bessere Übereinstimmung mit der Seitenmaske)
     - `YOLO_MIN_SIDE_RATIO` (filtern schmale Streifen)
     - `YOLO_MIN_RELATIVE_TO_CONTOUR` (YOLO-Box darf nicht zu klein ggü. OpenCV-Region sein)
     - `YOLO_MASTER_MODE` (Standard: `True`, YOLO ist die führende Quelle)
     - `YOLO_MASTER_MAX_CONTOUR_EXPAND` (optional: kleine Sicherheits-Erweiterung in Richtung OpenCV)
2. Danach läuft die Pipeline so:
   - YOLO erkennt zuerst Buch-/Seitenbereich
   - Die Erkennung wird zusätzlich mit der vorhandenen Seitenmaske validiert (gegen schmale Fremdbereiche/Illustrationen)
   - Standardmäßig ist YOLO der Master und OpenCV dient nur noch als sanfte Sicherheits-Erweiterung
   - Cropping, Perspective, Dewarping nutzen die final gewählte robuste Region
   - Falls YOLO nicht verfügbar ist oder nichts findet, wird automatisch auf die bisherige OpenCV-Logik zurückgefallen

Für beste Ergebnisse solltest du ein eigenes Modell auf Buchseiten trainieren (Label z. B. `page` oder `book_page`) und dann `YOLO_TARGET_CLASSES` entsprechend anpassen.


## Performance-Hinweis
Wenn die Verarbeitung zu langsam ist, kannst du in `process.py` `ENABLE_GLOBAL_ALIGNMENT = False` setzen.
Das deaktiviert die globale Rotations-Ausrichtung des kompletten Bildes und spart Rechenzeit, besonders bei großen Bildern.
