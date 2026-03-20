# predictor/forms.py
from django import forms

# --- CHOICES CONSTANTS ---
GENDER_CHOICES = [(1, 'Male'),(2, 'Female'),]
AGE_GROUP_CHOICES = [(1, '0 - 15 years old'),(2, '16 - 20 years old'),(3, '21 - 25 years old'),(4, 'Over 25 years'),]
LEVEL_OF_EDUCATION_CHOICES = [(1, 'Primary level'),(2, 'Form 1'),(3, 'Form 2'),(4, 'Form 3'),(5, 'Form 4'),(6, 'Form 5'),(7, 'Lowersith'),(8, 'Uppersith'),(9, 'University'),]
CURRENT_LIVING_SITUATION_CHOICES = [(1, 'With both Parents'),(2, 'With Mother'),(3, 'With Father'),(4, 'With a Guardian'),(5, 'Alone'),]
PARENT_MARITAL_STATUS_CHOICES = [(1, 'Married'),(2, 'Divorced'),(3, 'Separated'),(4, 'Widow(er)'),(5, 'Single'),]
INITIAL_SMOKING_AGE_CHOICES = [(1, 'Before 10 years'),(2, 'Between 10 and 15 years old'),(3, 'Between 15 and 20 years old'),(4, 'After 20 years old'),(5, 'I have never smoked'),]
SMOKING_INITIATION_REASON_CHOICES = [(1, 'Because I had problems'),(2, 'Because I was very happy'),(3, 'Because my friends smoked too and I imitated them'),(4, 'Because I thought it was very good'),(5, 'Out of curiosity'),(6, 'I have never smoked'),(7, 'Others'),]
CURRENT_SMOKING_REASON_CHOICES = [(1, 'Because I have problems and it relieves me'),(2, 'When I am very happy'),(3, 'Because my friends smoke too and I follow them'),(4, 'Because I like it'),(5, 'Because I can’t give up anymore'),(6, 'By habit'),(7, 'I have never smoked'),(8, 'Others'),]
INITIAL_ALCOHOL_AGE_CHOICES = [(1, 'Before 10 years'),(2, 'Between 10 and 15 years old'),(3, 'Between 15 and 20 years old'),(4, 'After 20 years old'),(5, 'I have never drank alcohol'),]
ALCOHOL_INITIATION_REASONS_CHOICES = [(1, 'Because I have problems'),(2, 'Because my friends drink and I followed them'),(3, 'Because I was very happy'),(4, 'Because I thought it was very good'),(5, 'Out of curiosity'),(6, 'I never drank alcohol'),(7, 'Others'),]
CURRENT_ALCOHOL_REASONS_CHOICES = [(1, 'Because I have problems and this relieves me'),(2, 'Because my friends also consume it and I follow them'),(3, 'Because I can’t give up anymore'),(4, 'When I am happy'),(5, 'Because I like it'),(6, 'By habit'),(7, 'I have never taken alcohol'),(8, 'Others'),]
YES_NO_CHOICES = [(1, 'Yes'),(2, 'No'),]
CONSUMPTION_COUNT_CHOICES = [(1, 'Never'),(2, 'One time'),(3, 'Many times'),(4, 'Regularly'),]
INITIAL_CONSUMPTION_REASON_CHOICES = [(1, 'Because I have problems'),(2, 'Because my friends consume it and I followed them'),(3, 'Because I was very happy'),(4, 'Because I thought it was very good'),(5, 'Out of curiosity'),(6, 'I have never consumed'),(7, 'Others'),]
CURRENT_DRUG_USE_REASONS_CHOICES = [(1, 'Because I have problems and this relieves me'),(2, 'Because my friends also consume it and I follow them'),(3, 'Because I can’t give up anymore'),(4, 'When I am happy'),(5, 'Because I like it'),(6, 'By habit'),(7, 'I have never consumed'),(8, 'Others'),]


# --- FORMS FOR MULTI-STEP SINGLE PREDICTION ---

class SlideOneForm(forms.Form):
    # Slide 1: Demographics with Smoking History
    Gender = forms.ChoiceField(choices=GENDER_CHOICES, label="Gender", widget=forms.RadioSelect, required=True)
    Age_Group = forms.ChoiceField(choices=AGE_GROUP_CHOICES, label="Age Group", required=True)
    Level_of_Education = forms.ChoiceField(choices=LEVEL_OF_EDUCATION_CHOICES, label="Level of Education", required=True)
    Current_Living_Situation = forms.ChoiceField(choices=CURRENT_LIVING_SITUATION_CHOICES, label="Current Living Situation", required=True)
    parent_marital_status = forms.ChoiceField(choices=PARENT_MARITAL_STATUS_CHOICES, label="Parent Marital Status", required=True)
    Initial_Smoking_age = forms.ChoiceField(choices=INITIAL_SMOKING_AGE_CHOICES, label="Age of First Smoking", required=True)
    Smoking_Initiation_Reason = forms.ChoiceField(choices=SMOKING_INITIATION_REASON_CHOICES, label="Reason for First Smoking", required=True)
    Current_Smoking_Reason = forms.ChoiceField(choices=CURRENT_SMOKING_REASON_CHOICES, label="Reason for Smoking Now", required=True)

class SlideTwoForm(forms.Form):
    # Slide 2: Alcohol History with Awareness of Substance
    Initial_Alcohol_Age = forms.ChoiceField(choices=INITIAL_ALCOHOL_AGE_CHOICES, label="Age of First Alcohol Use", required=True)
    Alcohol_Initiation_Reasons = forms.ChoiceField(choices=ALCOHOL_INITIATION_REASONS_CHOICES, label="Reason for First Alcohol Use", required=True)
    Current_Alcohol_Reasons = forms.ChoiceField(choices=CURRENT_ALCOHOL_REASONS_CHOICES, label="Reason for Drinking Alcohol Now", required=True)
    Head_of_sedative = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Tranquilizers or Sedatives?", widget=forms.RadioSelect, required=True)
    Head_of_marijuana = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Marijuana?", widget=forms.RadioSelect, required=True)
    Head_of_thai = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of THAI?", widget=forms.RadioSelect, required=True)
    Head_of_Ecstasy = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Ecstasy?", widget=forms.RadioSelect, required=True)
    Head_of_LSD = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of LSD?", widget=forms.RadioSelect, required=True)
    Head_of_Crack = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Crack?", widget=forms.RadioSelect, required=True)
    Head_of_Cocaine = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Cocaine?", widget=forms.RadioSelect, required=True)
    Head_of_Heroine = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Heroine?", widget=forms.RadioSelect, required=True)
    Head_of_D25 = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of D25?", widget=forms.RadioSelect, required=True)
    Head_of_Chicha = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Chicha?", widget=forms.RadioSelect, required=True)
    Head_of_Soukouda = forms.ChoiceField(choices=YES_NO_CHOICES, label="Heard of Soukoudai?", widget=forms.RadioSelect, required=True)

class SlideThreeForm(forms.Form):
    # Slide 3: Substance Consumption Frequency with Reasons for Consumption
    Product_Consumption_Count_Marijuana = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Marijuana Consumption Frequency", required=True)
    Product_Consumption_Count_Sedative = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Sedative Consumption Frequency", required=True)
    Product_Consumption_Count_D25 = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="D25 Consumption Frequency", required=True)
    Product_Consumption_Count_Cocaine = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Cocaine Consumption Frequency", required=True)
    Product_Consumption_Count_Thai = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="THAI Consumption Frequency", required=True)
    Product_Consumption_Count_Solvents = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Solvents Consumption Frequency", required=True)
    Product_Consumption_Count_Tramol = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Tramol Consumption Frequency", required=True)
    Product_Consumption_Count_Hookah = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Hookah Consumption Frequency", required=True)
    Product_Consumption_Count_Soudoudai = forms.ChoiceField(choices=CONSUMPTION_COUNT_CHOICES, label="Soukoudai Consumption Frequency", required=True)
    Initial_Consumption_Reason = forms.ChoiceField(choices=INITIAL_CONSUMPTION_REASON_CHOICES, label="Reason for First Drug Use", required=True)
    Current_Drug_Use_Reasons = forms.ChoiceField(choices=CURRENT_DRUG_USE_REASONS_CHOICES, label="Reason for Current Drug Use", required=True)

class SlideFourForm(forms.Form):
    # Slide 4: Impact Factors
    Legal_Issues_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Problems with the law due to substance use?", widget=forms.RadioSelect, required=True)
    Friend_Conflict_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Conflicts with friends due to substance use?", widget=forms.RadioSelect, required=True)
    Financial_or_Value_Loss_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Financial loss due to substance use?", widget=forms.RadioSelect, required=True)
    Parental_Conflict_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Conflicts with parents due to substance use?", widget=forms.RadioSelect, required=True)
    Emotional_Issues_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Emotional problems due to substance use?", widget=forms.RadioSelect, required=True)
    Teacher_Relationship_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Problems with teachers due to substance use?", widget=forms.RadioSelect, required=True)
    Work_or_School_Performance_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Poor performance at work/school due to substance use?", widget=forms.RadioSelect, required=True)
    Theft_Due_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Involved in theft due to substance use?", widget=forms.RadioSelect, required=True)
    Hospitalization_Due_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Hospitalized due to substance use?", widget=forms.RadioSelect, required=True)
    Sexual_Violence_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Experienced sexual violence due to substance use?", widget=forms.RadioSelect, required=True)
    Unsafe_Sex_Substance = forms.ChoiceField(choices=YES_NO_CHOICES, label="Had unprotected sex due to substance use?", widget=forms.RadioSelect, required=True)

# --- BATCH PREDICTION FORM ---
class BatchPredictionForm(forms.Form):
    excel_file = forms.FileField(
        label='Upload Excel File (.xlsx or .xls)',
        widget=forms.ClearableFileInput(attrs={'class': 'form-control'})
    )