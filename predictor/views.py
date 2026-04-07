from django.shortcuts import render

from predictor.models import Student


def home(request):
    return render(request, 'index.html')


def predict(request):
    if request.method == "POST":
        algorithm = request.POST.get("algorithm")
        if algorithm == "decision_tree":
            return result_dt(request)
        if algorithm == "random_forest":
            return result_rfc(request)
        return render(
            request,
            "predict.html",
            {
                "error": "Please select a prediction algorithm.",
                "selected_algorithm": algorithm,
            },
        )

    return render(request, 'predict.html')


def students(request):
    return render(request, 'students.html', {"students": Student.objects.all()})


def companies(request):
    return render(request, 'companies.html')

def algorithms(request):
    return render(request, 'algorithms.html')



# """
import os
import joblib
from django.conf import settings
import pandas as pd

model_path_dt = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'placement_predict_dt_model.pkl')
encoder_path_dt = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'label_encoders_dt.pkl')

model_path_rfc = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'placement_predict_rfc_model.pkl')
encoder_path_rfc = os.path.join(settings.BASE_DIR, 'predictor', 'models', 'label_encoders_rfc.pkl')



model_dt = joblib.load(model_path_dt)
label_encoders_dt = joblib.load(encoder_path_dt)

model_rfc = joblib.load(model_path_rfc)
label_encoders_rfc = joblib.load(encoder_path_rfc)



def result_dt(request):
    if request.method == "POST":
        data = request.POST
        features = [
            int(data['IQ Score']),
            float(data['Previous Semester Result']),
            float(data['CGPA']),
            int(data['Academic Performance']),
            label_encoders_dt['Internship_Experience'].transform([data['Internship Experience']])[0],
            int(data['Extra-Curricular Score']),
            int(data['Communication Skills']),
            int(data['Projects Completed']),
        ]

        # Feature names must match training
        feature_names = [
            "IQ", "Prev_Sem_Result", "CGPA", "Academic_Performance", "Internship_Experience",
            "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed"
        ]

        df_input = pd.DataFrame([features], columns= feature_names)
        yes_no = model_dt.predict(df_input)[0]
        # Choose emoji based on score
        if yes_no:
            emoji = "😌"
            msg = "Congratulations! as per our prediction you will be placed."
        else:
            emoji = "😟"
            msg = "Sorry as per our prediction you won't be placed."

        name = data['Name']

        context = {
            "yes_no": yes_no,
            "emoji": emoji,
            "msg": msg,
            "name": name,
            "selected_algorithm": "decision_tree",
            "algorithm_name": "Decision Tree Algorithm",
        }
        return render(request,"predict.html", context)

    return render(request, "predict.html")

# """


def result_rfc(request):
    if request.method == "POST":
        data = request.POST
        features = [
            int(data['IQ Score']),
            float(data['Previous Semester Result']),
            float(data['CGPA']),
            int(data['Academic Performance']),
            label_encoders_rfc['Internship_Experience'].transform([data['Internship Experience']])[0],
            int(data['Extra-Curricular Score']),
            int(data['Communication Skills']),
            int(data['Projects Completed']),
        ]

        # Feature names must match training
        feature_names = [
           "IQ", "Prev_Sem_Result",	"CGPA",	"Academic_Performance",	"Internship_Experience",
            "Extra_Curricular_Score", "Communication_Skills", "Projects_Completed"
        ]

        df_input = pd.DataFrame([features], columns= feature_names)
        yes_no = model_rfc.predict(df_input)[0]
        # Choose emoji based on score
        if yes_no:
            emoji = "😌"
            msg = "Congratulations! as per our prediction you will be placed."
        else:
            emoji = "😟"
            msg = "Sorry as per our prediction you won't be placed."

        name = data['Name']

        context = {
            "yes_no": yes_no,
            "emoji": emoji,
            "msg": msg,
            "name": name,
            "selected_algorithm": "random_forest",
            "algorithm_name": "Random Forest Classifier Algorithm",
        }
        return render(request,"predict.html", context)

    return render(request, "predict.html")

