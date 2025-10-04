from flask import Flask, render_template, request, jsonify
from model import get_response
from ExoTerra import ExoTerraModel
import joblib
import pandas as pd
import json

app = Flask(__name__, static_url_path='/static')

# ---------------------- Pages ----------------------
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/explore')
def explore():
    return render_template("explore.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/manually')
def manually():
    return render_template("manually.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/data')
def data():
    return render_template("data.html")

# ---------------------- CSV Upload & AI Model ----------------------
@app.route('/upload_form', methods=['POST'])
def upload_form():
    # Safely get form fields
    name = request.form.get('name', 'N/A')
    category = request.form.get('category', 'N/A')
    description = request.form.get('description', 'N/A')

    print(f"Name: {name}, Category: {category}, Description: {description}")

    # Get uploaded file
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    if not uploaded_file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        # Convert FileStorage to DataFrame
        df = pd.read_csv(uploaded_file)

        # Pass DataFrame to ExoTerraModel
        model = ExoTerraModel(df, target_name="disposition_num")
        model.pipeline = joblib.load("files/best_exo_randomsearch_model_disposition_num.joblib")
        predictions = model.predict(df)
        predictions_json = json.loads(predictions)

        print("Predictions:", predictions_json)

        return jsonify({
            "message": "Form submitted!",
            "name": name,
            "category": category,
            "description": description,
            "predictions": predictions_json
        })

    except Exception as e:
        print("Error processing file:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------- Manual Form ----------------------
@app.route('/manually_form', methods=['POST'])
def manually_form():
    try:
        # Use get() with defaults to avoid KeyError
        data = {
            'insolation_flux': request.form.get('insolation_flux'),
            'transit_depth': request.form.get('transit_depth'),
            'stellar_rad': request.form.get('stellar_rad'),
            'planet_radius': request.form.get('planet_radius'),
            'orbital_period': request.form.get('orbital_period'),
            'stellar_logg': request.form.get('stellar_logg'),
            'mission': request.form.get('mission'),
            'equilibrium_temp': request.form.get('equilibrium_temp'),
            'stellar_teff': request.form.get('stellar_teff'),
            'transit_duration': request.form.get('transit_duration'),
        }

        print("Manual Form Data:", data)
        # Optional: process this data using your model if needed
        return jsonify({"message": "Manual form submitted!", "data": data})

    except Exception as e:
        print("Error in manually_form:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------- AI Response ----------------------
@app.route('/get_response', methods=['POST'])
def get_ai_response():
    try:
        prompt = request.json.get('prompt')
        response_text = get_response(prompt)
        print("AI Response:", response_text)
        return jsonify({"response": response_text})
    except Exception as e:
        print("Error in get_response:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------- Run App ----------------------
if __name__ == '__main__':
    app.run(debug=True)
