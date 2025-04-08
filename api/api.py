from flask import Flask, request, jsonify, render_template
import joblib
import os


app = Flask(__name__, template_folder="../templates")  

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-form")
def form():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [[
        data["interest_rate"],
        data["mortgage_rate"],
        data["inflation_cpi"],
        data["unemployment"],
        data["housing_starts"]
    ]]
    prediction = model.predict(features)
    return jsonify({"predicted_home_price_index": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
