# predictor/views.py

from django.shortcuts import render, redirect
from django.conf import settings
from django.http import HttpResponse
import os
import pandas as pd
import joblib
import numpy as np
from io import StringIO, BytesIO
import base64
import matplotlib.pyplot as plt

# Import LIME
import lime
import lime.lime_tabular

# Import all form classes AND all CHOICES constants from forms.py
from .forms import (
    SlideOneForm, SlideTwoForm, SlideThreeForm, SlideFourForm, BatchPredictionForm,
    GENDER_CHOICES, AGE_GROUP_CHOICES, LEVEL_OF_EDUCATION_CHOICES,
    CURRENT_LIVING_SITUATION_CHOICES, PARENT_MARITAL_STATUS_CHOICES,
    INITIAL_SMOKING_AGE_CHOICES, SMOKING_INITIATION_REASON_CHOICES,
    CURRENT_SMOKING_REASON_CHOICES, INITIAL_ALCOHOL_AGE_CHOICES,
    ALCOHOL_INITIATION_REASONS_CHOICES, CURRENT_ALCOHOL_REASONS_CHOICES,
    YES_NO_CHOICES, CONSUMPTION_COUNT_CHOICES, INITIAL_CONSUMPTION_REASON_CHOICES,
    CURRENT_DRUG_USE_REASONS_CHOICES
)

# --- Initialize Global Variables ---
model = None
scaler = None
label_encoder = None
lime_explainer = None  # New variable for LIME
TRAINING_DATA_IMPUTATION_VALUES = {}  # Placeholder

# --- Load ML Assets at Server Startup ---
print("--- ATTEMPTING TO LOAD ML ASSETS ---")
MODEL_FILENAME = 'tuned_adaboost_model.pkl'
SCALER_FILENAME = 'scaler_model.pkl'
LABEL_ENCODER_FILENAME = 'label_encoder_model.pkl'
MODEL_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model', MODEL_FILENAME)
SCALER_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model', SCALER_FILENAME)
LABEL_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model', LABEL_ENCODER_FILENAME)

try: model = joblib.load(MODEL_PATH)
except Exception as e: print(f"ERROR loading model: {e}")
try: scaler = joblib.load(SCALER_PATH)
except Exception as e: print(f"ERROR loading scaler: {e}")
try: label_encoder = joblib.load(LABEL_ENCODER_PATH)
except Exception as e: print(f"ERROR loading label encoder: {e}")

# Validate loaded assets
ml_assets_loaded = all([model, scaler, label_encoder])
if ml_assets_loaded:
    print("SUCCESS: All core ML assets loaded.")
    print(f"Model type: {type(model).__name__}")
    print(f"Scaler type: {type(scaler).__name__}")
    print(f"Label encoder classes: {label_encoder.classes_ if hasattr(label_encoder, 'classes_') else 'N/A'}")
else:
    print("WARNING: Some ML assets failed to load. Predictions will not work.")



ALL_CHOICES_MAP = {
    'Gender': GENDER_CHOICES,
    'Age_Group': AGE_GROUP_CHOICES,
    'Level_of_Education': LEVEL_OF_EDUCATION_CHOICES,
    'Current_Living_Situation': CURRENT_LIVING_SITUATION_CHOICES,
    'parent_marital_status': PARENT_MARITAL_STATUS_CHOICES,
    'Initial_Smoking_age': INITIAL_SMOKING_AGE_CHOICES,
    'Smoking_Initiation_Reason': SMOKING_INITIATION_REASON_CHOICES,
    'Current_Smoking_Reason': CURRENT_SMOKING_REASON_CHOICES,
    'Initial_Alcohol_Age': INITIAL_ALCOHOL_AGE_CHOICES,
    'Alcohol_Initiation_Reasons': ALCOHOL_INITIATION_REASONS_CHOICES,
    'Current_Alcohol_Reasons': CURRENT_ALCOHOL_REASONS_CHOICES,
    'Head_of_sedative': YES_NO_CHOICES,
    'Head_of_marijuana': YES_NO_CHOICES,
    'Head_of_thai': YES_NO_CHOICES,
    'Head_of_Ecstasy': YES_NO_CHOICES,
    'Head_of_LSD': YES_NO_CHOICES,
    'Head_of_Crack': YES_NO_CHOICES,
    'Head_of_Cocaine': YES_NO_CHOICES,
    'Head_of_Heroine': YES_NO_CHOICES,
    'Head_of_D25': YES_NO_CHOICES,
    'Head_of_Chicha': YES_NO_CHOICES,
    'Head_of_Soukouda': YES_NO_CHOICES,
    'Product_Consumption_Count_Marijuana': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Sedative': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_D25': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Cocaine': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Thai': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Solvents': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Tramol': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Hookah': CONSUMPTION_COUNT_CHOICES,
    'Product_Consumption_Count_Soudoudai': CONSUMPTION_COUNT_CHOICES,
    'Initial_Consumption_Reason': INITIAL_CONSUMPTION_REASON_CHOICES,
    'Current_Drug_Use_Reasons': CURRENT_DRUG_USE_REASONS_CHOICES,
    'Legal_Issues_Substance': YES_NO_CHOICES,
    'Friend_Conflict_Substance': YES_NO_CHOICES,
    'Financial_or_Value_Loss_Substance': YES_NO_CHOICES,
    'Parental_Conflict_Substance': YES_NO_CHOICES,
    'Emotional_Issues_Substance': YES_NO_CHOICES,
    'Teacher_Relationship_Substance': YES_NO_CHOICES,
    'Work_or_School_Performance_Substance': YES_NO_CHOICES,
    'Theft_Due_Substance': YES_NO_CHOICES,
    'Hospitalization_Due_Substance': YES_NO_CHOICES,
    'Sexual_Violence_Substance': YES_NO_CHOICES,
    'Unsafe_Sex_Substance': YES_NO_CHOICES
}




# # --- MODEL_EXPECTED_FEATURES (Source of Truth from Error Logs) ---
# These are the messy names the scaler/model file expects.
SCALER_AND_MODEL_EXPECTED_FEATURES = [
    'Gender',
    'Age_Group',
    'Level_of_Education',
    'Current_Living_Situation',
    'parent_marital_status',
    'Initial_Smoking_age',
    'Smoking_Initiation_Reason',
    'Current_smoking_Reason',  # lowercase 's' to match CSV
    'Initial_Alcohol_Age',
    'Alcohol_Initiation_Reasons',
    'Current_Alcohol_Reasons',
    'Head_of_sedative',
    'Head_of_marijuana',
    'Head_of_thai',
    'Head_of_Ecstasy',
    'Head_of_LSD',
    'Head_of_Crack',
    'Head_of_Cocaine',
    'Head_of_Heroine',
    'Head_of_D25',
    'Head_of_Chicha',
    'Head_of_Soukouda',
    'Product_Consumption_Count_Marijuana',
    'Product_Consumption_Count_Sedative',
    'Product_Consumption_Count_D25',
    'Product_Consumption_Count_Cocaine',
    'Product_Consumption_Count_Thai',
    'Product_Consumption_Count_Solvents',
    'Product_Consumption_Count_Tramol',
    'Product_Consumption_Count_Hookah',
    'Product_Consumption_Count_Soudoudai',
    'Initial_Consumption_Reason',
    'Current_Drug_Use_Reasons',
    'Legal_Issues_Substance',
    'Friend_Conflict_Substance',
    'Financial_or_value_Loss_Substance',  # lowercase 'v' to match CSV
    'Parental_Conflict_Substance',
    'Emotional_Issues_Substance',
    'Teacher_Relationship_Substance',
    'Work_or_School_Performance_substance',  # lowercase 's' to match CSV
    'Theft_Due_Substance',
    'Hospitalization_Due_Substance',
    'Sexaul_Violence_Substance',  # misspelled to match CSV
    'Unsafe_Sex_Substance'
]








# # predictor/views.py

# # --- FEATURE NAME DEFINITIONS AND MAPPINGS (DEFINITIVE VERSION) ---

# # THESE ARE THE MESSY NAMES YOUR SCALER/MODEL WERE TRAINED ON (GROUND TRUTH)
# SCALER_AND_MODEL_EXPECTED_FEATURES = [
#     'Gender', 'Age_Group', 'Level of Education', 'Current_Living_Situation',
#     'parent_marital_status', 'Initial_Smoking_age', 'Smoking_Initiation_Reason',
#     'Current_smooking_Reason', # Messy name
#     'Initial_Alcohol_Age', 'Alcohol_Initiation_Reasons', 'Current_Alcohol_Reasons',
#     'Head_of_sedative', 'Head_of_marijuana', 'Head_of_thai', 'Head_of_Ecstasy',
#     'Head_of_LSD', 'Head_of_Crack', 'Head_of_Cocaine', 'Head_of_Heroine',
#     'Head_of_D25', 'Head_of_Chicha', 'Head_of_Soukouda',
#     'Product_Consumption_Count_Marijuana', 'Product_Consumption_Count_Sedative',
#     'Product_Consumption_Count_D25', 'Product_Consumption_Count_Cocaine',
#     'Product_Consumption_Count_Thai', 'Product_Consumption_Count_Solvents',
#     'Product_Consumption_Count_Tramol', 'Product_Consumption_Count_Hookah',
#     'Product_Consumption_Count_Soudoudai', 'Initial_Consumption_Reason',
#     'Current_Drug_Use_Reasons', 'Legal_Issues_Substance', 'Friend_Conflict_Substance',
#     'Financial_or_valuve_Loss_Substance', # Messy name
#     'Parental_Conflict_Substance', 'Emotional_Issues_Substance',
#     'Teacher_Relationship_Substance',
#     'Work_or_School_Performance_substance', # Messy name
#     'Theft_Due_Substance', 'Hospitalization_Due_Substance',
#     'Sexaul_violence_Substance', # Messy name
#     'Unsafe_Sex_Substance'
# ]

# # THIS IS THE TRANSLATION DICTIONARY
# # Key: Messy name the model expects. Value: Clean name your form provides.
# # (This assumes your forms.py has the clean, corrected names)
# MODEL_TO_FORM_KEY_MAP = {
#     'Current_smooking_Reason': 'Current_Smoking_Reason',
#     'Financial_or_valuve_Loss_Substance': 'Financial_or_Value_Loss_Substance',
#     'Sexaul_violence_Substance': 'Sexual_Violence_Substance',
#     'Work_or_School_Performance_substance': 'Work_or_School_Performance_Substance',
#     'Level of Education': 'Level_of_Education',
#     **{f: f for f in SCALER_AND_MODEL_EXPECTED_FEATURES if f not in [
#         'Current_smooking_Reason', 'Financial_or_valuve_Loss_Substance',
#         'Sexaul_violence_Substance', 'Work_or_School_Performance_substance',
#         'Level of Education'
#     ]}
# }




# --- MODEL_TO_FORM_KEY_MAP (The Translation Layer) ---
# Key: Messy name the model expects (from CSV). Value: Clean name your form provides.
MODEL_TO_FORM_KEY_MAP = {
    'Gender': 'Gender',
    'Age_Group': 'Age_Group',
    'Level_of_Education': 'Level_of_Education',
    'Current_Living_Situation': 'Current_Living_Situation',
    'parent_marital_status': 'parent_marital_status',
    'Initial_Smoking_age': 'Initial_Smoking_age',
    'Smoking_Initiation_Reason': 'Smoking_Initiation_Reason',
    'Current_smoking_Reason': 'Current_Smoking_Reason',  # CSV has lowercase 's'
    'Initial_Alcohol_Age': 'Initial_Alcohol_Age',
    'Alcohol_Initiation_Reasons': 'Alcohol_Initiation_Reasons',
    'Current_Alcohol_Reasons': 'Current_Alcohol_Reasons',
    'Head_of_sedative': 'Head_of_sedative',
    'Head_of_marijuana': 'Head_of_marijuana',
    'Head_of_thai': 'Head_of_thai',
    'Head_of_Ecstasy': 'Head_of_Ecstasy',
    'Head_of_LSD': 'Head_of_LSD',
    'Head_of_Crack': 'Head_of_Crack',
    'Head_of_Cocaine': 'Head_of_Cocaine',
    'Head_of_Heroine': 'Head_of_Heroine',
    'Head_of_D25': 'Head_of_D25',
    'Head_of_Chicha': 'Head_of_Chicha',
    'Head_of_Soukouda': 'Head_of_Soukouda',
    'Product_Consumption_Count_Marijuana': 'Product_Consumption_Count_Marijuana',
    'Product_Consumption_Count_Sedative': 'Product_Consumption_Count_Sedative',
    'Product_Consumption_Count_D25': 'Product_Consumption_Count_D25',
    'Product_Consumption_Count_Cocaine': 'Product_Consumption_Count_Cocaine',
    'Product_Consumption_Count_Thai': 'Product_Consumption_Count_Thai',
    'Product_Consumption_Count_Solvents': 'Product_Consumption_Count_Solvents',
    'Product_Consumption_Count_Tramol': 'Product_Consumption_Count_Tramol',
    'Product_Consumption_Count_Hookah': 'Product_Consumption_Count_Hookah',
    'Product_Consumption_Count_Soudoudai': 'Product_Consumption_Count_Soudoudai',
    'Initial_Consumption_Reason': 'Initial_Consumption_Reason',
    'Current_Drug_Use_Reasons': 'Current_Drug_Use_Reasons',
    'Legal_Issues_Substance': 'Legal_Issues_Substance',
    'Friend_Conflict_Substance': 'Friend_Conflict_Substance',
    'Financial_or_value_Loss_Substance': 'Financial_or_Value_Loss_Substance',  # CSV has lowercase 'v'
    'Parental_Conflict_Substance': 'Parental_Conflict_Substance',
    'Emotional_Issues_Substance': 'Emotional_Issues_Substance',
    'Teacher_Relationship_Substance': 'Teacher_Relationship_Substance',
    'Work_or_School_Performance_substance': 'Work_or_School_Performance_Substance',  # CSV has lowercase 's'
    'Theft_Due_Substance': 'Theft_Due_Substance',
    'Hospitalization_Due_Substance': 'Hospitalization_Due_Substance',
    'Sexaul_Violence_Substance': 'Sexual_Violence_Substance',  # CSV has misspelling
    'Unsafe_Sex_Substance': 'Unsafe_Sex_Substance'
}






# --- LIME Explainer Initialization ---
def get_training_data_sample_for_lime():
    print("LIME: Attempting to load REAL training data sample for explainer.")
    try:
        # 1. It defines the path to the file you just added
        sample_path = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model', 'training_data_sample.csv')

        # 2. It loads the CSV file into a Pandas DataFrame
        training_df_sample = pd.read_csv(sample_path)

        # 3. It selects the correct columns in the correct order
        training_df_sample_ordered = training_df_sample[SCALER_AND_MODEL_EXPECTED_FEATURES]

        # 4. It converts the DataFrame to a NumPy array for LIME
        numpy_sample = training_df_sample_ordered.to_numpy()

        print(f"LIME: Successfully loaded real training sample with shape {numpy_sample.shape}")
        return numpy_sample

    except Exception as e:
        print(f"LIME ERROR: Could not load 'training_data_sample.csv'. LIME will not work. Error: {e}")
        return None

# THIS IS WHERE THE FUNCTION IS CALLED:
X_train_sample_for_lime_numpy = get_training_data_sample_for_lime()

# This code then uses the loaded data to initialize the explainer
if X_train_sample_for_lime_numpy is not None and model is not None and label_encoder is not None:
    try:
        print("LIME: Initializing LimeTabularExplainer with proper parameters...")

        # Identify categorical features (all features are categorical in your dataset)
        categorical_features = list(range(len(SCALER_AND_MODEL_EXPECTED_FEATURES)))

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_sample_for_lime_numpy,
            feature_names=SCALER_AND_MODEL_EXPECTED_FEATURES,  # ✅ Added: Feature names for readable explanations
            categorical_features=categorical_features,  # ✅ Added: All features are categorical
            mode='classification',  # ✅ Added: Explicit mode
            verbose=True,  # ✅ Added: Enable logging
            random_state=42  # ✅ Added: Reproducible results
        )
        print("LIME: LimeTabularExplainer initialized successfully with all parameters.")

    except Exception as e:
        print(f"LIME ERROR: Could not initialize LimeTabularExplainer: {e}")
        lime_explainer = None
else:
    print("LIME: Missing required assets (training data, model, or label encoder). LIME will not be available.")
    lime_explainer = None




# --- Define the sequence of forms for our "slides" ---
FORMS = [
    ("slide_one", SlideOneForm, "Demographics & Smoking History"),
    ("slide_two", SlideTwoForm, "Alcohol History & Substance Awareness"),
    ("slide_three", SlideThreeForm, "Substance Consumption & Reasons"),
    ("slide_four", SlideFourForm, "Impact Factors"),
]

# --- Helper Function ---
def preprocess_input(form_data_dict, model_feature_list, name_map):
    processed_data = {}
    for model_feature_name in model_feature_list:
        form_field_key = name_map.get(model_feature_name)
        value = form_data_dict.get(form_field_key) if form_field_key else None
        if value is not None and value != '':
            try: processed_data[model_feature_name] = int(value)
            except (ValueError, TypeError): processed_data[model_feature_name] = np.nan
        else:
            processed_data[model_feature_name] = np.nan
    return pd.DataFrame([processed_data], columns=model_feature_list)

# --- Views ---
def home(request):
    return render(request, 'predictor/home.html')





# def single_prediction_view(request):
#     if request.method == 'POST':
#         current_step_index = int(request.POST.get('current_step', 0))

#         if 'previous_step' in request.POST:
#             if current_step_index > 0:
#                 form_to_clear_name, _, _ = FORMS[current_step_index - 1]
#                 if form_to_clear_name in request.session:
#                     del request.session[form_to_clear_name]
#             return redirect('predictor:single_prediction')

#         if current_step_index < len(FORMS):
#             form_name, form_class, _ = FORMS[current_step_index]
#             form = form_class(request.POST)

#             if form.is_valid():
#                 request.session[form_name] = form.cleaned_data
#                 request.session.modified = True
                
#                 if 'get_prediction' in request.POST:
#                     all_form_data, prediction_text, error_message, decoded_user_inputs, lime_explanation_list = {}, None, None, {}, None

#                     for f_name, _, _ in FORMS:
#                         all_form_data.update(request.session.get(f_name, {}))
                    
#                     if len(all_form_data) < len(SCALER_AND_MODEL_EXPECTED_FEATURES):
#                          error_message = "Not all steps were completed. Please start over."
#                     else:
#                         try:
#                             # Decode user inputs for display first
#                             for form_field_name, encoded_value in all_form_data.items():
#                                 choices_list = ALL_CHOICES_MAP.get(form_field_name)
#                                 decoded_value = str(encoded_value)
#                                 if choices_list:
#                                     decoded_value = next((text for val, text in choices_list if str(val) == str(encoded_value)), decoded_value)
#                                 pretty_feature_name = form_field_name.replace("_", " ").title()
#                                 decoded_user_inputs[pretty_feature_name] = decoded_value
                            
#                             # Prediction Pipeline
#                             raw_input_df = preprocess_input(all_form_data, SCALER_AND_MODEL_EXPECTED_FEATURES, MODEL_TO_FORM_KEY_MAP)
#                             instance_for_lime_explanation = raw_input_df.iloc[0].to_numpy().copy()
                            
#                             if raw_input_df.isnull().values.any():
#                                 print("WARNING: NaNs found. Applying generic fillna(0).")
#                                 raw_input_df.fillna(0, inplace=True)
                            
#                             model_input_df = raw_input_df.copy()
#                             if scaler:
#                                 scaled_array = scaler.transform(raw_input_df)
#                                 model_input_df = pd.DataFrame(scaled_array, columns=SCALER_AND_MODEL_EXPECTED_FEATURES)
                            
#                             prediction_raw_val = None
#                             if model and label_encoder:
#                                 prediction_encoded = model.predict(model_input_df)[0]
#                                 prediction_raw_val = prediction_encoded
#                                 prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
#                                 prediction_text = str(prediction_decoded).replace("_", " ").title()

#                                 # LIME Explanation Generation
#                                 if lime_explainer is not None and prediction_raw_val is not None:
#                                     try:
#                                         def lime_predict_fn_wrapper(numpy_data):
#                                             df = pd.DataFrame(numpy_data, columns=SCALER_AND_MODEL_EXPECTED_FEATURES)
#                                             df.fillna(0, inplace=True)
#                                             return model.predict_proba(scaler.transform(df) if scaler else df)

#                                         predicted_class_idx = np.where(label_encoder.classes_ == prediction_raw_val)[0][0]
                                        
#                                         explanation_obj = lime_explainer.explain_instance(
#                                             data_row=instance_for_lime_explanation, 
#                                             predict_fn=lime_predict_fn_wrapper,
#                                             num_features=10, labels=(predicted_class_idx,)
#                                         )
#                                         lime_explanation_list = explanation_obj.as_list(label=predicted_class_idx)
#                                         print(f"LIME Explanation Generated.")
#                                     except Exception as e_lime_gen:
#                                         print(f"LIME ERROR during explanation generation: {e_lime_gen}")
#                             else:
#                                 error_message = "Model or Label Encoder not loaded on the server."
#                         except Exception as e:
#                             error_message = f"An error occurred during prediction: {e}"
                    
#                     for f_name, _, _ in FORMS:
#                         if f_name in request.session: del request.session[f_name]

#                     return render(request, 'predictor/single_prediction.html', {
#                         'prediction_result_text': prediction_text,
#                         'decoded_user_inputs': decoded_user_inputs,
#                         'error_message': error_message,
#                         'lime_explanation_list': lime_explanation_list,
#                     })

#                 return redirect('predictor:single_prediction')
#             else: # Form is not valid
#                 _, _, current_title = FORMS[current_step_index]
#                 return render(request, 'predictor/single_prediction.html', {
#                     'form_list': FORMS, 'current_step': current_step_index,
#                     'total_steps': len(FORMS), 'current_form': form,
#                     'current_title': current_title
#                 })
#         return redirect('predictor:single_prediction')
#     else: # GET request
#         current_step = 0
#         for i, (form_name, _, _) in enumerate(FORMS):
#             if form_name not in request.session:
#                 current_step = i
#                 break
#         else:
#             current_step = len(FORMS)
        
#         current_form, current_title = (None, "Confirm & Predict")
#         if current_step < len(FORMS):
#             _, current_form_class, current_title = FORMS[current_step]
#             current_form = current_form_class()

#         return render(request, 'predictor/single_prediction.html', {
#             'form_list': FORMS, 'current_step': current_step,
#             'total_steps': len(FORMS), 'current_form': current_form,
#             'current_title': current_title
#         })












# predictor/views.py

def single_prediction_view(request):
    if request.method == 'POST':
        current_step_index = int(request.POST.get('current_step', 0))

        if 'previous_step' in request.POST:
            if current_step_index > 0:
                form_to_clear, _, _ = FORMS[current_step_index - 1]
                if form_to_clear in request.session: del request.session[form_to_clear]
            return redirect('predictor:single_prediction')

        if current_step_index < len(FORMS):
            form_name, form_class, _ = FORMS[current_step_index]
            form = form_class(request.POST)

            if form.is_valid():
                request.session[form_name] = form.cleaned_data
                request.session.modified = True
                
                if 'get_prediction' in request.POST:
                    all_form_data = {k: v for f, _, _ in FORMS for k, v in request.session.get(f, {}).items()}
                    
                    prediction_text, error_message, decoded_inputs, lime_list, lime_error_message = None, None, {}, None, None
                    
                    try:
                        if len(all_form_data) < len(SCALER_AND_MODEL_EXPECTED_FEATURES):
                            raise ValueError("Not all steps were completed. Please start over.")
                        
                        for field, val in all_form_data.items():
                            choices = ALL_CHOICES_MAP.get(field, [])
                            decoded = next((txt for v, txt in choices if str(v) == str(val)), str(val))
                            decoded_inputs[field.replace("_", " ").title()] = decoded
                        
                        # The pipeline starts here
                        raw_df = preprocess_input(all_form_data, SCALER_AND_MODEL_EXPECTED_FEATURES, MODEL_TO_FORM_KEY_MAP)
                        raw_df.fillna(0, inplace=True)

                        # ✅ FIXED: Keep as float for LIME perturbations
                        lime_instance = raw_df.iloc[0].to_numpy().copy().astype(float)

                        scaled_array = scaler.transform(raw_df) if scaler else raw_df.to_numpy()
                        model_df = pd.DataFrame(scaled_array, columns=SCALER_AND_MODEL_EXPECTED_FEATURES)

                        prediction_raw_val = None
                        if model and label_encoder:
                            pred_encoded = model.predict(model_df)[0]
                            prediction_raw_val = pred_encoded
                            pred_decoded = label_encoder.inverse_transform([pred_encoded])[0]
                            prediction_text = str(pred_decoded).replace("_", " ").title()

                            # LIME Explanation Generation
                            if lime_explainer and prediction_raw_val is not None:
                                print("\n--- LIME: Attempting to generate explanation ---")
                                print(f"LIME: prediction_raw_val = {prediction_raw_val}")
                                print(f"LIME: label_encoder.classes_ = {label_encoder.classes_}")
                                try:
                                    def predict_fn(numpy_data):
                                        # ✅ FIXED: Keep data as float, don't convert to int
                                        df = pd.DataFrame(numpy_data, columns=SCALER_AND_MODEL_EXPECTED_FEATURES).fillna(0)
                                        print(f"LIME predict_fn: input shape = {numpy_data.shape}")

                                        # Apply scaling if scaler exists
                                        if scaler:
                                            scaled_array = scaler.transform(df)
                                            # ✅ FIXED: Handle out-of-range values from LIME perturbations
                                            scaled_sanitized = np.nan_to_num(scaled_array, nan=0.0, posinf=1.0, neginf=0.0)
                                            print(f"LIME predict_fn: scaled shape = {scaled_sanitized.shape}")
                                            probs = model.predict_proba(scaled_sanitized)
                                            print(f"LIME predict_fn: output probs shape = {probs.shape}, sample = {probs[0] if len(probs) > 0 else 'empty'}")
                                            return probs
                                        else:
                                            probs = model.predict_proba(df)
                                            print(f"LIME predict_fn: output probs shape = {probs.shape}")
                                            return probs

                                    pred_idx = pred_encoded  # pred_encoded is already the class index (0-9)
                                    print(f"LIME: pred_idx = {pred_idx} (same as pred_encoded)")
                                    print(f"LIME: lime_instance shape = {lime_instance.shape}, type = {type(lime_instance)}, sample values = {lime_instance[:5] if len(lime_instance) > 5 else lime_instance}")

                                    exp = lime_explainer.explain_instance(
                                        lime_instance,
                                        predict_fn,
                                        num_features=10,
                                        labels=(pred_idx,)
                                    )
                                    lime_list = exp.as_list(label=pred_idx)
                                    print(f"LIME: explanation list length = {len(lime_list)}, content = {lime_list}")
                                    print("--- LIME Explanation generated successfully. ---")
                                except Exception as e_lime:
                                    lime_error_message = f"LIME internal error: {e_lime}"
                                    print(f"LIME ERROR: {e_lime}")
                                    import traceback
                                    print(f"LIME ERROR traceback:\n{traceback.format_exc()}")
                        else:
                            error_message = "Model or Label Encoder not loaded."
                    except Exception as e:
                        error_message = f"An unexpected error occurred: {e}"
                        print(f"FATAL PREDICTION ERROR: {e}")

                    # Clear session data after processing
                    for f_name, _, _ in FORMS:
                        if f_name in request.session: del request.session[f_name]

                    # Render the final results page
                    return render(request, 'predictor/single_prediction.html', {
                        'prediction_result_text': prediction_text, 
                        'decoded_user_inputs': decoded_inputs,
                        'error_message': error_message, 
                        'lime_explanation_list': lime_list,
                        'lime_error_message': lime_error_message,
                        'lime_available': lime_explainer is not None,
                    })
                
                return redirect('predictor:single_prediction')
            else: # Form is not valid
                _, _, current_title = FORMS[current_step_index]
                return render(request, 'predictor/single_prediction.html', {
                    'form_list': FORMS, 'current_step': current_step_index,
                    'total_steps': len(FORMS), 'current_form': form,
                    'current_title': current_title
                })
        
        return redirect('predictor:single_prediction')

    else: # GET request logic
        current_step = 0
        for i, (f_name, _, _) in enumerate(FORMS):
            if f_name not in request.session:
                current_step = i
                break
        else:
            current_step = len(FORMS)
        
        current_form, current_title = (None, "Confirm & Predict")
        if current_step < len(FORMS):
            _, form_class, current_title = FORMS[current_step]
            current_form = form_class()

        return render(request, 'predictor/single_prediction.html', {
            'form_list': FORMS, 'current_step': current_step,
            'total_steps': len(FORMS), 'current_form': current_form,
            'current_title': current_title
        })










def batch_prediction_view(request):
    # This is where the complete batch prediction view with visualizations goes.
    # Please use the final session-based version from our previous interactions.
    form = BatchPredictionForm()
    context = {'form': form, 'error_message': "Batch view is ready for implementation."}
    return render(request, 'predictor/batch_prediction.html', context)



def batch_prediction_view(request):
    """
    Handles batch prediction via Excel file upload, preview, visualization, and download.
    Uses Django sessions to maintain state between processing and downloading.
    """
    form = BatchPredictionForm()
    error_message = None
    results_df_html_preview = None
    show_download_button = False
    context_data = {}
    age_group_pie_chart = None
    dependency_status_bar_plot = None
    gender_age_dependency_plot = None

    # --- ACTION 1: Handle a DOWNLOAD request ---
    if request.method == 'POST' and 'download_excel' in request.POST:
        print("Batch: Download request received.")
        batch_results_df_json = request.session.get('batch_results_df_json')
        if batch_results_df_json:
            try:
                json_string_io = StringIO(batch_results_df_json)
                results_df_from_session = pd.read_json(json_string_io, orient='split')
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    results_df_from_session.to_excel(writer, index=False, sheet_name='Predictions')
                output.seek(0)
                
                response = HttpResponse(
                    output,
                    content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                response['Content-Disposition'] = 'attachment; filename="batch_predictions_output.xlsx"'
                
                # Clean up session data after successful download preparation
                if 'batch_results_df_json' in request.session: del request.session['batch_results_df_json']
                if 'batch_plot_data' in request.session: del request.session['batch_plot_data']
                
                print("Batch: Preparing download response from session data.")
                return response
            except Exception as e_download:
                error_message = f"Error generating download file: {e_download}"
                print(f"Batch Download Error: {e_download}")
        else:
            error_message = "No processed data found in session to download. Please process a file first."

    # --- ACTION 2: Handle a FILE PROCESSING request ---
    elif request.method == 'POST' and 'process_file' in request.POST:
        print("Batch: File processing request received.")
        form = BatchPredictionForm(request.POST, request.FILES)
        if form.is_valid():
            excel_file = request.FILES['excel_file']
            if not excel_file.name.endswith(('.xls', '.xlsx')):
                error_message = "Invalid file type. Please upload an .xls or .xlsx file."
            else:
                try:
                    # Clear any old results from session before processing a new file
                    if 'batch_results_df_json' in request.session: del request.session['batch_results_df_json']
                    if 'batch_plot_data' in request.session: del request.session['batch_plot_data']

                    raw_batch_df = pd.read_excel(excel_file, sheet_name=0)
                    
                    # TODO: Implement robust column name mapping here if needed.
                    # Example: raw_batch_df.rename(columns=your_mapping_dict, inplace=True)

                    predictions_list = []
                    original_rows = raw_batch_df.copy()
                    
                    for index, row in raw_batch_df.iterrows():
                        try:
                            row_data_dict = row.to_dict()
                            processed_row_df = preprocess_input(row_data_dict, SCALER_AND_MODEL_EXPECTED_FEATURES, MODEL_TO_FORM_KEY_MAP)
                            
                            # Placeholder imputation - replace with robust strategy for production
                            processed_row_df.fillna(0, inplace=True)
                            
                            scaled_row_df = processed_row_df.copy()
                            if scaler:
                                scaled_array = scaler.transform(processed_row_df)
                                scaled_row_df = pd.DataFrame(scaled_array, columns=SCALER_AND_MODEL_EXPECTED_FEATURES)
                            
                            if model and label_encoder:
                                prediction_encoded = model.predict(scaled_row_df)[0]
                                prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
                                predictions_list.append(str(prediction_decoded)) # Keep original format for grouping
                            else:
                                predictions_list.append("Model/Encoder N/A")
                        except Exception as e_row:
                            print(f"Error processing batch row {index+1}: {e_row}")
                            predictions_list.append("Error Processing Row")
                    
                    results_df = original_rows.copy()
                    # Use a new column for display text to preserve original prediction for grouping
                    results_df['Predicted_Dependency_Status'] = predictions_list
                    results_df['Predicted_Dependency_Status_Display'] = [p.replace("_", " ").title() for p in predictions_list]

                    # --- Generate and Store Visualizations ---
                    plot_data = {}
                    try:
                        AGE_GROUP_MAP = {1: '0-15', 2: '16-20', 3: '21-25', 4: '>25'}
                        GENDER_MAP = {1: 'Male', 2: 'Female'}

                        # Plot 1: Age Group Pie Chart
                        if 'Age_Group' in results_df.columns:
                            df_for_plot = results_df.copy()
                            df_for_plot['Age_Group_Label'] = df_for_plot['Age_Group'].map(AGE_GROUP_MAP)
                            age_counts = df_for_plot['Age_Group_Label'].value_counts()
                            plt.figure(figsize=(6, 5)); plt.style.use('seaborn-v0_8-pastel')
                            plt.pie(age_counts, labels=age_counts.index, autopct='%1.1f%%', startangle=140)
                            plt.title('Age Group Distribution'); plt.ylabel('')
                            buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                            plot_data['age_pie'] = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()

                        # Plot 2: Dependency Status Bar Plot
                        if 'Predicted_Dependency_Status_Display' in results_df.columns:
                            dependency_counts = results_df['Predicted_Dependency_Status_Display'].value_counts()
                            plt.figure(figsize=(8, 6)); plt.style.use('seaborn-v0_8-whitegrid')
                            dependency_counts.sort_values().plot(kind='barh', color='teal')
                            plt.title('Distribution of Predicted Status'); plt.xlabel('Number of Individuals'); plt.tight_layout()
                            buf = BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight'); buf.seek(0)
                            plot_data['dependency_bar'] = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()

                        # Plot 3: Grouped Bar Chart for Gender, Age, and Status
                        if all(c in results_df.columns for c in ['Gender', 'Age_Group', 'Predicted_Dependency_Status_Display']):
                            df_for_plot['Gender_Label'] = df_for_plot['Gender'].map(GENDER_MAP)
                            crosstab_df = pd.crosstab(
                                index=[df_for_plot['Age_Group_Label'], df_for_plot['Gender_Label']],
                                columns=df_for_plot['Predicted_Dependency_Status_Display']
                            )
                            crosstab_df.plot(kind='bar', figsize=(14, 8), width=0.8, stacked=False)
                            plt.title('Predicted Status by Age Group and Gender', fontsize=16)
                            plt.xlabel('Age Group & Gender', fontsize=12); plt.ylabel('Count of Individuals', fontsize=12)
                            plt.xticks(rotation=45, ha='right'); plt.legend(title='Dependency Status'); plt.tight_layout()
                            buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
                            plot_data['gender_age_dependency_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8'); plt.close()
                    
                    except Exception as e_plot:
                        print(f"Error generating plots: {e_plot}")

                    # Store results and plots in session
                    request.session['batch_results_df_json'] = results_df.to_json(orient='split')
                    request.session['batch_plot_data'] = plot_data
                    
                    # Prepare context for immediate render after processing
                    results_df_html_preview = results_df[['Predicted_Dependency_Status_Display'] + [col for col in results_df.columns if col != 'Predicted_Dependency_Status_Display']].head(20).to_html(classes='table table-striped table-hover table-sm', index=False)
                    show_download_button = True
                    age_group_pie_chart = plot_data.get('age_pie')
                    dependency_status_bar_plot = plot_data.get('dependency_bar')
                    gender_age_dependency_plot = plot_data.get('gender_age_dependency_plot')

                except Exception as e:
                    error_message = f"An error occurred during batch processing: {e}"
        else:
            error_message = "Form is not valid. Please select a file."

    # --- ACTION 3: Handle GET request (e.g., page refresh after processing) ---
    elif request.method == 'GET' and request.session.get('batch_results_df_json'):
        print("Batch: GET request with session data found. Preparing preview.")
        try:
            json_string_io = StringIO(request.session['batch_results_df_json'])
            df_for_preview = pd.read_json(json_string_io, orient='split')
            results_df_html_preview = df_for_preview[['Predicted_Dependency_Status_Display'] + [col for col in df_for_preview.columns if col != 'Predicted_Dependency_Status_Display']].head(20).to_html(classes='table table-striped table-hover table-sm', index=False)
            show_download_button = True
            
            plot_data = request.session.get('batch_plot_data', {})
            age_group_pie_chart = plot_data.get('age_pie')
            dependency_status_bar_plot = plot_data.get('dependency_bar')
            gender_age_dependency_plot = plot_data.get('gender_age_dependency_plot')
        except Exception as e_session:
            error_message = f"Could not load previous results from session: {e_session}"
            if 'batch_results_df_json' in request.session: del request.session['batch_results_df_json']
            if 'batch_plot_data' in request.session: del request.session['batch_plot_data']
            
    # --- Final Context Preparation and Render ---
    context_data.update({
        'form': form,
        'results_df_html': results_df_html_preview,
        'show_download_button': show_download_button,
        'error_message': error_message,
        'age_group_pie_chart': age_group_pie_chart,
        'dependency_status_bar_plot': dependency_status_bar_plot,
        'gender_age_dependency_plot': gender_age_dependency_plot,
    })
    return render(request, 'predictor/batch_prediction.html', context_data)