import re
from pathlib import Path

import pydicom
from pydicom.dataelem import DataElement

# Eingabeordner: hier liegen die Originaldateien
input_folder = Path(r"C:\Users\juliu\Desktop\Julius\Uni\Master MP\Proekt\Dicom\MM\26.08(2)")

# Ausgabeordner: hier werden die neuen Dateien gespeichert
output_folder = input_folder / "Dicom_neu"
output_folder.mkdir(exist_ok=True)

# Funktion, um Winkel für Dateinamen sauber zu formatieren
def format_angle_for_filename(value):
    """
    Wandelt z. B. '90.0' in '90' um.
    Aus '90.5' bleibt '90.5'.
    """
    value_float = float(value)
    if value_float.is_integer():
        return str(int(value_float))
    else:
        return str(value_float).replace(".", "p")


# Nur .dcm-Dateien im Eingabeordner durchgehen
for file_path in input_folder.glob("*.dcm"):
    try:
        # DICOM laden
        ds = pydicom.dcmread(str(file_path), force=True)

        # Prüfen, ob der relevante Tag existiert
        text = str(ds.get((0x0008, 0x103E), ""))
        if not text:
            print(f"{file_path.name}: Kein Tag (0008,103E) gefunden -> übersprungen")
            continue

        # G, K, T auslesen
        g_match = re.search(r"G(-?\d+(?:\.\d+)?)", text)
        k_match = re.search(r"K(-?\d+(?:\.\d+)?)", text)
        t_match = re.search(r"T(-?\d+(?:\.\d+)?)", text)

        if not (g_match and k_match and t_match):
            print(f"{file_path.name}: G, K oder T nicht gefunden -> übersprungen")
            continue

        g_wert = g_match.group(1)
        k_wert = k_match.group(1)
        t_wert = t_match.group(1)

        # G nur setzen, wenn der Tag noch nicht existiert
        if (0x300A, 0x011E) not in ds:
            ds[(0x300A, 0x011E)] = DataElement((0x300A, 0x011E), "DS", g_wert)

        # K und T setzen/überschreiben
        ds[(0x300A, 0x0120)] = DataElement((0x300A, 0x0120), "DS", k_wert)
        ds[(0x300A, 0x0122)] = DataElement((0x300A, 0x0122), "DS", t_wert)

        # Winkel für Dateinamen formatieren
        g_name = format_angle_for_filename(g_wert)
        k_name = format_angle_for_filename(k_wert)
        t_name = format_angle_for_filename(t_wert)

        # Neuer Dateiname entsprechend Gantry-, Kollimator- und Tischwinkel
        new_name = f"G{g_name}_K{k_name}_T{t_name}_neu{file_path.suffix}"

        output_path = output_folder / new_name

        # Falls Dateiname schon existiert, laufende Nummer anhängen
        counter = 1
        while output_path.exists():
            new_name = f"G{g_name}_K{k_name}_T{t_name}_neu_{counter}{file_path.suffix}"
            output_path = output_folder / new_name
            counter += 1

        # Speichern
        ds.save_as(str(output_path))

        print(f"{file_path.name}: gespeichert als {output_path.name}")

    except Exception as e:
        print(f"{file_path.name}: Fehler -> {e}")