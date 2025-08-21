from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from flask import Flask, render_template, request, send_from_directory, jsonify


# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = "F:\cancer_project\\trained_multi_cancer_model.h5"
UPLOAD_FOLDER = "F:\cancer_project\\uploads"
STATIC_FOLDER = "F:\cancer_project\\static\\images"
VISUALIZATION_FOLDER = "F:\cancer_project\\static\\Cancer_Visualizations"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['VISUALIZATION_FOLDER'] = VISUALIZATION_FOLDER

# Load trained model
model = load_model(MODEL_PATH)

# Image processing details
IMAGE_SIZE = (224, 224)
CLASS_LABELS = [
    "Leukemia begin", "Leukemia early", "Leukemia pre", "Leukemia pro",
    "Brain glimo", "Brain menin", "Brain tumor",
    "Breast benign", "Breast malignant",
    "Cervical Dyskeratotic", "Cervix Koilocytotic", "Cervix Metaplastic",
    "Cervix Parabasal"
]

# Suggested diet plans based on cancer type and diabetes status
DIET_PLANS = {
    "Leukemia Begin": {
        "yes": "Low-sugar diet with leafy greens, turmeric, and green tea.",
        "no": "High-antioxidant diet with turmeric, green tea, and fish."
    },
    "Leukemia Early": {
        "yes": "Balanced diet with whole grains, berries, and lean protein.",
        "no": "Mediterranean diet with olive oil, fish, and nuts."
    },
    "Leukemia Pre": {
        "yes": "High-protein, low-carb diet with chia seeds and avocados.",
        "no": "Plant-based diet with whole grains, legumes, and vegetables."
    },
    "Leukemia Pro": {
        "yes": "Keto-based diet with nuts, seeds, and fish.",
        "no": "High-fiber diet with beans, berries, and green vegetables."
    },
    "Brain Tumor": {
        "yes": "Ketogenic diet with avocados, nuts, and coconut oil.",
        "no": "Omega-3 rich diet with salmon, flaxseeds, and walnuts."
    },
    "Brain Glioma": {
        "yes": "Anti-inflammatory diet with turmeric, ginger, and berries.",
        "no": "Mediterranean diet with olive oil, nuts, and leafy greens."
    },
    "Brain Meningioma": {
        "yes": "Plant-based diet with lentils, almonds, and whole grains.",
        "no": "Low-sodium diet with lean meats and fresh vegetables."
    },
    "Breast Malignant": {
        "yes": "Low-carb, high-protein diet with cruciferous vegetables.",
        "no": "Plant-based diet with whole grains and legumes."
    },
    "Breast Benign": {
        "yes": "Folate-rich diet with citrus fruits, leafy greens, and lentils.",
        "no": "Balanced diet with fresh fruits, nuts, and yogurt."
    },
    "Cervical Dyskeratotic": {
        "yes": "Vitamin A and C-rich diet with carrots, oranges, and bell peppers.",
        "no": "Probiotic diet with yogurt, kefir, and fermented foods."
    },
    "Cervix Koilocytotic": {
        "yes": "Immunity-boosting diet with garlic, spinach, and turmeric.",
        "no": "High-fiber diet with beans, whole grains, and leafy greens."
    },
    "Cervix Metaplastic": {
        "yes": "Low-sugar, high-fiber diet with nuts, seeds, and vegetables.",
        "no": "Antioxidant-rich diet with pomegranates, blueberries, and dark chocolate."
    },
    "Cervix Parabasal": {
        "yes": "Iron and folate-rich diet with spinach, lentils, and fish.",
        "no": "Vitamin-rich diet with fruits, nuts, and whole grains."
    }
}

# Cancer Treatment Advisory Dictionary
treatment_advisory = {
    "Leukemia Begin": {
        "Chemotherapy": "Mild chemotherapy (Methotrexate, Vincristine) for 4-6 months.",
        "Radiation Therapy": "Usually not required.",
        "Bone Marrow Transplant": "Not necessary at this stage.",
        "Other Treatments": "Immunotherapy (Blincyto) may be considered."
    },
    "Leukemia Early": {
        "Chemotherapy": "Combination of Anthracyclines (Daunorubicin) and Cytarabine for 6-12 months.",
        "Radiation Therapy": "Targeted radiation if the leukemia has spread to the central nervous system (CNS).",
        "Bone Marrow Transplant": "Considered if high risk of progression."
    },
    "Leukemia Pre": {
        "Chemotherapy": "Intensive chemotherapy with drugs like Fludarabine, Cyclophosphamide, and Rituximab.",
        "Radiation Therapy": "Cranial radiation if leukemia is in the brain.",
        "Bone Marrow Transplant": "Recommended for younger patients."
    },
    "Leukemia Pro": {
        "Chemotherapy": "High-dose chemotherapy with targeted therapy (Tyrosine kinase inhibitors like Imatinib).",
        "Radiation Therapy": "Used in limited cases for pain relief.",
        "Bone Marrow Transplant": "Urgently needed for survival."
    },
    "Brain Glioma": {
        "Radiation Therapy": "30-40 sessions over 6-7 weeks.",
        "Chemotherapy": "Temozolomide (TMZ) combined with radiation.",
        "Surgery": "If operable, surgical removal is preferred.",
        "Targeted Therapy": "Bevacizumab for aggressive gliomas."
    },
    "Brain Meningioma": {
        "Radiation Therapy": "Stereotactic radiosurgery if surgery isn’t possible.",
        "Surgery": "First-line treatment for accessible tumors.",
        "Chemotherapy": "Rarely needed unless the tumor is aggressive."
    },
    "Brain Tumor": {
        "Radiation Therapy": "Intensity-Modulated Radiation Therapy (IMRT).",
        "Surgery": "Often performed if the tumor is accessible.",
        "Chemotherapy": "Temozolomide or Carmustine (BCNU)."
    },
    "Breast Benign": {
        "Surgery": "Lumpectomy if needed.",
        "Radiation Therapy": "Not required.",
        "Chemotherapy": "Not required.",
        "Other Treatments": "Healthy diet, regular screenings."
    },
    "Breast Malignant": {
        "Chemotherapy": "Neoadjuvant (before surgery) and adjuvant (after surgery).",
        "Radiation Therapy": "5-6 weeks post-surgery.",
        "Surgery": "Mastectomy or lumpectomy.",
        "Hormonal Therapy": "If estrogen receptor-positive (Tamoxifen)."
    },
    "Cervical Dyskeratotic": {
        "Radiation Therapy": "External beam radiation (EBRT) for 5-6 weeks.",
        "Chemotherapy": "Cisplatin-based therapy.",
        "Surgery": "Cone biopsy for early-stage cases."
    },
    "Cervix Koilocytotic": {
        "Surgery": "LEEP procedure or cryotherapy for mild cases.",
        "Radiation Therapy": "Not needed in early stages.",
        "Chemotherapy": "Used if the condition worsens."
    },
    "Cervix Metaplastic": {
        "Radiation Therapy": "Used for advanced stages.",
        "Chemotherapy": "Paclitaxel or Cisplatin.",
        "Surgery": "Hysterectomy if aggressive."
    },
    "Cervix Parabasal": {
        "Radiation Therapy": "High-dose brachytherapy.",
        "Chemotherapy": "5-Fluorouracil (5-FU).",
        "Surgery": "Required for localized tumors."
    }
}

# Image Processing Function
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Prediction Function
def predict_cancer_type(img_path, threshold=0.5):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_probability = np.max(predictions)

    if predicted_probability < 0.97000000:
        return "No cancer detected", predicted_probability, None

    cancer_type = CLASS_LABELS[predicted_class[0]]
    return cancer_type, predicted_probability, None

# Routes
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('diagnosis.html', error="No file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('diagnosis.html', error="No file selected.")

        if file:
            file_path = os.path.join(app.config['STATIC_FOLDER'], file.filename)
            file.save(file_path)
            cancer_type, probability, _ = predict_cancer_type(file_path)

            return render_template('diagnosis.html', prediction=cancer_type, confidence=probability, image_path=file.filename)

    return render_template('diagnosis.html')

@app.route('/treatment', methods=['GET', 'POST'])
def treatment():
    selected_cancer = None
    treatment_data = {}

    if request.method == 'POST':
        selected_cancer = request.form.get('cancer_type')
        treatment_data = treatment_advisory.get(selected_cancer, {})

    return render_template('treatment.html', treatment_advisory=treatment_advisory, selected_cancer=selected_cancer, treatment_data=treatment_data)



@app.route('/diet', methods=['GET', 'POST'])
def diet():
    diet_plan = None  # Default value

    if request.method == 'POST':
        cancer_type = request.form.get('cancer_type')
        diabetic = request.form.get('diabetic')  # "yes" or "no"

        # Retrieve diet plan if it exists, otherwise set a default message
        if cancer_type in DIET_PLANS:
            diet_plan = DIET_PLANS[cancer_type].get(diabetic, "No specific diet recommendation available.")
        else:
            diet_plan = "No diet plan found for the selected cancer type."

    return render_template('diet.html', diet_plan=diet_plan)
@app.route('/visualizations')
def visualization_page():
    # Fetch all available visualizations
    images = os.listdir(app.config['VISUALIZATION_FOLDER'])
    return render_template("visualizations.html", images=images)

@app.route('/visualizations/image/<filename>')
def visualization_images(filename):
    return send_from_directory(app.config['VISUALIZATION_FOLDER'], filename)


@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')

@app.route('/insurance')
def insurance():
    return render_template('insurance.html')

if __name__ == "__main__":
    app.run(debug=True)
