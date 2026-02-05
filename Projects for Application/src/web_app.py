import os
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from recommend_core import load_products, load_rules, build_routine

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)


CSV_PATH = os.path.join("Data", "skincare_products.csv")
RULES_PATH = "ingredients_rules.json"

@lru_cache(maxsize=1)
def get_products():
    return load_products(CSV_PATH)

@lru_cache(maxsize=1)
def get_rules():
    return load_rules(RULES_PATH)

def parse_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/result")
def result():
    skin_type = request.form.get("skin_type", "").strip().lower()
    concerns = parse_list(request.form.getlist("concerns"))
    avoid = parse_list(request.form.getlist("avoid"))

    df = get_products()
    rules = get_rules()

    data = build_routine(df, rules, concerns, avoid, skin_type)
    return render_template("results.html", skin_type=skin_type, concerns=concerns, avoid=avoid, data=data)

@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    payload = request.get_json(silent=True) or {}
    skin_type = (payload.get("skin_type") or request.args.get("skin_type") or "").strip().lower()
    concerns = parse_list(payload.get("concerns") or request.args.get("concerns"))
    avoid = parse_list(payload.get("avoid") or request.args.get("avoid"))

    df = get_products()
    rules = get_rules()
    data = build_routine(df, rules, concerns, avoid, skin_type)

    return jsonify({
        "skin_type": skin_type,
        "concerns": concerns,
        "avoid": avoid,
        "data": data
    })

if __name__ == "__main__":
    app.run(debug=True)
