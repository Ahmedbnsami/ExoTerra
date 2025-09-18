from flask import Flask, render_template, request

app = Flask(__name__)

SPORTS = [
    "Football",
    "Cricket",
    "Hockey",
    "Tennis",
    "Badminton",
    "Swimming",
    "Skating",
    "Surfing",    
]

REGISTRANTS = {}

@app.route("/")
def index():
    return render_template("index.html", sports=SPORTS)

@app.route("/register", methods=["POST"])
def register():
    if not request.form.get("name"):
        return render_template("failure.html", message="Name is required.")
    if not request.form.get("sports"):
        return render_template("failure.html", message="Sport is required.")
    if request.form.get("sports") not in SPORTS:
        return render_template("failure.html", message="Invalid sport selected.")
    
    REGISTRANTS[request.form.get("name")] = request.form.get("sports")
    return render_template("success.html", registrants=REGISTRANTS)

@app.route("/registrants")
def registrants():
    return render_template("registrants.html", registrants=REGISTRANTS)