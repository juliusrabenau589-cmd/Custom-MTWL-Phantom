from pathlib import Path

from pylinac.winston_lutz import (
    WinstonLutzMultiTargetMultiField,
    BBConfig
)


# ---------------------------------
# Phantomdefinition
# ---------------------------------

my_special_phantom_bbs = [

    BBConfig(
        name="Iso",
        offset_left_mm=0,
        offset_up_mm=0,
        offset_in_mm=0,
        bb_size_mm=5,
        rad_size_mm=14
    ),

    BBConfig(
        name="1",
        offset_left_mm=20,
        offset_up_mm=0,
        offset_in_mm=40,
        bb_size_mm=5,
        rad_size_mm=14
    ),

    BBConfig(
        name="2",
        offset_left_mm=0,
        offset_up_mm=-30,
        offset_in_mm=60,
        bb_size_mm=5,
        rad_size_mm=14
    ),

    BBConfig(
        name="3",
        offset_left_mm=-20,
        offset_up_mm=0,
        offset_in_mm=-30,
        bb_size_mm=5,
        rad_size_mm=14
    ),

    BBConfig(
        name="4",
        offset_left_mm=0,
        offset_up_mm=30,
        offset_in_mm=-50,
        bb_size_mm=5,
        rad_size_mm=14
    ),
]


# ---------------------------------
# DICOM-Ordner
# ---------------------------------

my_directory = Path(
    r"C:\Users\juliu\Desktop\Julius\Uni\Master MP\Proekt\Dicom\MM\22.08\Dicom_neu"
)


# ---------------------------------
# pylinac laden
# ---------------------------------

wl = WinstonLutzMultiTargetMultiField(my_directory)


# ---------------------------------
# Analyse
# ---------------------------------

wl.analyze(
    bb_arrangement=my_special_phantom_bbs,
    bb_proximity_mm=30
)


# ---------------------------------
# Ergebnisse
# ---------------------------------

results_text = wl.results()

print(results_text)


# ---------------------------------
# PDF im DICOM-Ordner speichern
# ---------------------------------

pdf_path = my_directory / "WinstonLutz_MultiTarget.pdf"

wl.publish_pdf(str(pdf_path))


# ---------------------------------
# TXT-Datei erstellen
# ---------------------------------

txt_path = my_directory / "WinstonLutz_Auswertung.txt"


with open(txt_path, "w", encoding="utf-8") as file:

    file.write("Winston-Lutz Multi-Target Auswertung\n")
    file.write("===================================\n\n")

    # ---------------------------------
    # Targetkoordinaten
    # ---------------------------------

    file.write("Ermittelte Targetmittelpunkte\n")
    file.write("-----------------------------\n\n")

    file.write(
        f"{'Target':<10}"
        f"{'X [mm]':>12}"
        f"{'Y [mm]':>12}"
        f"{'Z [mm]':>12}\n"
    )

    file.write("-" * 46 + "\n")

    for bb in wl.bbs:

        position = bb.measured_bb_position

        name = bb.bb_config.name

        file.write(
            f"{name:<10}"
            f"{position.x:>12.3f}"
            f"{position.y:>12.3f}"
            f"{position.z:>12.3f}\n"
        )


    # ---------------------------------
    # Testergebnisse
    # ---------------------------------

    file.write("\n\n")
    file.write("Testergebnisse\n")
    file.write("--------------\n\n")

    file.write(results_text)


# ---------------------------------
# Speicherorte ausgeben
# ---------------------------------

print("\nDateien wurden erstellt:")

print(f"\nPDF:\n{pdf_path}")

print(f"\nTXT:\n{txt_path}")