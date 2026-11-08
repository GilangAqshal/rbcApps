import base64
import csv
import io
import uuid
from datetime import datetime
from flask import Flask, render_template, request, send_file, abort
import cv2
import numpy as np
import rbc

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
# Muat ulang template otomatis saat file HTML diubah (tanpa restart server)
app.config["TEMPLATES_AUTO_RELOAD"] = True
CSV_EXPORTS = {}

def to_data_uri(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Gagal mengubah hasil citra.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

def _int(name, default, lo, hi):
    try: v = int(request.form.get(name, default))
    except (TypeError, ValueError): v = default
    return max(lo, min(hi, v))

def _float(name, default, lo, hi):
    try: v = float(request.form.get(name, default))
    except (TypeError, ValueError): v = default
    return max(lo, min(hi, v))

def get_params():
    return {
        "channel": request.form.get("channel", "green"),
        "blur_ksize": _int("blur_ksize", 5, 1, 21),
        "use_clahe": request.form.get("use_clahe") == "on",
        "morph_ksize": _int("morph_ksize", 5, 1, 21),
        "use_watershed": request.form.get("use_watershed") == "on",
        "remove_wbc": request.form.get("remove_wbc") == "on",
        "wbc_sat": _int("wbc_sat", 55, 10, 200),
        "dist_ratio": _float("dist_ratio", 0.45, 0.1, 0.9),
        "min_area": _int("min_area", 300, 10, 5000),
    }

@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"result": None, "error": None, "form": dict(rbc.DEFAULTS)}
    if request.method == "POST":
        files = [f for f in request.files.getlist("images") if f and f.filename]
        if not files:
            ctx["error"] = "Pilih minimal satu gambar JPG, PNG, atau JPEG."
            return render_template("index.html", **ctx)
        if len(files) > 30:
            ctx["error"] = "Maksimal 30 gambar dalam satu kali proses."
            return render_template("index.html", **ctx)
        params = get_params()
        ctx["form"] = params
        results, csv_rows = [], []
        for file in files:
            raw = file.read()
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                res = rbc.process(img, params)
                for stage in res["stages"]:
                    stage["uri"] = to_data_uri(stage["image"])
                    del stage["image"]
                item = {
                    "filename": file.filename,
                    "count": res["count"],
                    "otsu_threshold": round(res["otsu_threshold"], 2),
                    "median_area": round(res["median_area"], 2),
                    "wbc_count": res["wbc_count"],
                    "ignored_small": res["ignored_small"],
                    "ignored_big": res["ignored_big"],
                    "stages": res["stages"],
                }
                results.append(item)
                csv_rows.append({k: item[k] for k in (
                    "filename", "count", "otsu_threshold", "median_area",
                    "wbc_count", "ignored_small", "ignored_big")})
            except Exception as exc:
                app.logger.exception("Gagal memproses %s", file.filename)
        if not results:
            ctx["error"] = "Tidak ada gambar yang berhasil diproses. Pastikan file berupa JPG atau PNG."
        else:
            export_id = uuid.uuid4().hex
            CSV_EXPORTS[export_id] = csv_rows
            ctx["result"] = results
            ctx["export_url"] = f"/download-csv/{export_id}"
            ctx["processed_at"] = datetime.now().strftime("%d-%m-%Y %H:%M")
    return render_template("index.html", **ctx)

@app.route("/download-csv/<export_id>")
def download_csv(export_id):
    rows = CSV_EXPORTS.get(export_id)
    if rows is None:
        abort(404)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    mem = io.BytesIO(("\ufeff" + output.getvalue()).encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv; charset=utf-8",
                     as_attachment=True, download_name="hasil_deteksi_rbc.csv")

@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", result=None, error="Ukuran total upload terlalu besar (maksimal 80 MB).",
                           form=dict(rbc.DEFAULTS)), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
