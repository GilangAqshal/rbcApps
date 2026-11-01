"""
Evaluasi pipeline pada dataset BCCD: bandingkan jumlah RBC/WBC hasil program
dengan jumlah anotasi (file XML Pascal VOC).

Pemakaian:
    python evaluate.py --root path/ke/BCCD --n 15
    (folder --root berisi JPEGImages/ dan Annotations/)

Hasil: tabel di terminal + file evaluasi.csv
"""
import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import rbc


def count_annotations(xml_path):
    c = {"RBC": 0, "WBC": 0, "Platelets": 0}
    for o in ET.parse(xml_path).getroot().findall("object"):
        name = o.find("name").text
        c[name] = c.get(name, 0) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=15, help="jumlah citra yang dievaluasi")
    ap.add_argument("--out", default="evaluasi.csv")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "JPEGImages", "*.jpg")))[: a.n]
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        gt = count_annotations(os.path.join(a.root, "Annotations", name + ".xml"))
        res = rbc.process(cv2.imread(f))
        diff = res["count"] - gt["RBC"]
        acc = max(0.0, 100 * (1 - abs(diff) / gt["RBC"])) if gt["RBC"] else 0.0
        rows.append([name, gt["RBC"], res["count"], diff, round(acc, 1), gt["WBC"], res["wbc_count"]])

    hdr = ["citra", "RBC_anotasi", "RBC_program", "selisih", "akurasi_%", "WBC_anotasi", "WBC_program"]
    print("{:<18}{:>12}{:>12}{:>9}{:>11}{:>12}{:>12}".format(*hdr))
    for r in rows:
        print("{:<18}{:>12}{:>12}{:>9}{:>11}{:>12}{:>12}".format(*r))
    mae = np.mean([abs(r[3]) for r in rows])
    print(f"\nMAE jumlah RBC : {mae:.2f} sel")
    print(f"Rata-rata akurasi : {np.mean([r[4] for r in rows]):.1f}%")
    print(f"WBC cocok anotasi : {sum(r[5] == r[6] for r in rows)}/{len(rows)} citra")

    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    print("Disimpan ke", a.out)


if __name__ == "__main__":
    main()
