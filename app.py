import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

import rbc

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def to_data_uri(img):
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _int(name, default, lo, hi):
    try:
        v = int(request.form.get(name, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _float(name, default, lo, hi):
    try:
        v = float(request.form.get(name, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"defaults": rbc.DEFAULTS, "result": None, "error": None, "form": rbc.DEFAULTS}
    if request.method == "POST":
        f = request.files.get("image")
        if not f or not f.filename:
            ctx["error"] = "Pilih file citra terlebih dahulu."
            return render_template("index.html", **ctx)

        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            ctx["error"] = "File tidak dapat dibaca sebagai citra (gunakan JPG/PNG)."
            return render_template("index.html", **ctx)

        params = {
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
        res = rbc.process(img, params)
        for s in res["stages"]:
            s["uri"] = to_data_uri(s["image"])
            del s["image"]

        truth = request.form.get("truth", "").strip()
        if truth.isdigit():
            t = int(truth)
            diff = res["count"] - t
            res["truth"] = t
            res["diff"] = diff
            res["accuracy"] = max(0.0, 100.0 * (1 - abs(diff) / t)) if t else None

        ctx["result"] = res
        ctx["filename"] = f.filename
        ctx["form"] = params
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
