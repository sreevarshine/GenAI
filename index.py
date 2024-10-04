from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware



shape_model = load_model('shape_model.h5')
spot_model = load_model('spot_model.h5')
stem_model = load_model('stem_model.h5')
webbing_model = load_model('webbing_model.h5')
diseases = load_model('diseases.h5')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((224, 224))  
        image_array = np.array(image) / 255.0  
        if image_array.shape[-1] == 3:
            image_array = np.expand_dims(image_array, axis=0)  
        else:
            return JSONResponse(content={"error": "Input image must have 3 channels (RGB)."}, status_code=400)

        shape_prediction = shape_model.predict(image_array)
        shape_class = np.argmax(shape_prediction, axis=1)[0]
        
        spot_prediction = spot_model.predict(image_array)
        stem_prediction = stem_model.predict(image_array)
        webbing_prediction = webbing_model.predict(image_array)
        disease_prediction = diseases.predict(image_array)
        result = {}
        result['spot'] = "Yellow spot: ripe" if spot_prediction < 0.5 else "White or pale yellow spot: Not ripe"
        result['stem'] = "Green stem: Not ripe" if stem_prediction < 0.5 else "Brown stem: Ripe"
        result['webbing'] = "Webbing Present: Ripe" if webbing_prediction < 0.5 else "Webbing Not Present: Not ripe"
        shape_messages = [
            "The watermelon is Round: Ripe and Sweet",
            "The watermelon is Elongated: Watery",
            "The watermelon is Irregular: Not Ripe"
        ]
        result['shape'] = shape_messages[shape_class] if shape_class < len(shape_messages) else "Unknown shape"
        
        # Disease messages (skipping specific diseases)
        disease_names = [
            "Anthracnoseon",
            "Bacterial fruit blotch",
            "Blossom End Rot",
            "Gummy Stem Blight",
            "Cross Stitch",
            "Greasy Spot",
            "Target Cluster",
            "Phytophthora Fruit Rot",
            "No diseases"
        ]

        # Checking for diseases, skipping "Cross Stitch" and "Greasy Spot"
        diseases_present = []
        for i, prediction in enumerate(disease_prediction[0]):  
            if prediction > 0.5: 
                if disease_names[i] not in ["Cross Stitch", "Greasy Spot"]:
                    diseases_present.append(disease_names[i])
        if diseases_present:
            result['disease'] = "Diseases Present: " + ", ".join(diseases_present)
        else:
            result['disease'] = "No significant diseases detected."

        # Overall recommendation based on predictions
        if len(diseases_present) == 0:
            overall_recommendation = "This watermelon is likely good to buy."
        else:
            overall_recommendation = "Consider avoiding this watermelon due to detected diseases."

        # Add overall recommendation to the result
        result['recommendation'] = overall_recommendation
        print(result)
        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        return