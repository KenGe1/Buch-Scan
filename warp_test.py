import cv2
import numpy as np

# ==========================
# Bild laden
# ==========================
image = cv2.imread(r"C:\Users\Kevin\OneDrive\Desktop\Buch test input\Buch test Output\page_0015.png")

if image is None:
    raise ValueError("Bild konnte nicht geladen werden!")

orig = image.copy()

# ==========================
# Graustufen
# ==========================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nur leicht glätten
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ==========================
# Helle Fläche (Papier) extrahieren
# ==========================
_, mask = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)

# Kleine Löcher schließen
kernel = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# ==========================
# Konturen finden
# ==========================
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not contours:
    raise ValueError("Keine Konturen gefunden!")

# Größte helle Fläche = Seite
doc = max(contours, key=cv2.contourArea)

# Rotated Rectangle (stabil)
rect = cv2.minAreaRect(doc)
box = cv2.boxPoints(rect)
box = box.astype("float32")

# Punkte sortieren
s = box.sum(axis=1)
tl = box[np.argmin(s)]
br = box[np.argmax(s)]

diff = np.diff(box, axis=1)
tr = box[np.argmin(diff)]
bl = box[np.argmax(diff)]

rect = np.array([tl, tr, br, bl], dtype="float32")

# ==========================
# Zielgröße berechnen
# ==========================
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

# ==========================
# Perspektive korrigieren
# ==========================
M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

cv2.imwrite(r"C:\Users\Kevin\OneDrive\Desktop\Buch test input\Buch test Output\scan_ergebnis.jpg", warped)

print("Fertig gespeichert.")
