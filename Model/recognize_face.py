import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from Model.SQL import get_student_info

def recognize_face(face, db_path):
    try:
        dfs = DeepFace.find(face, db_path=db_path, enforce_detection=False, model_name='Facenet', detector_backend="retinaface")
        if len(dfs) > 0:
            dfs = np.array(dfs).tolist()
            df = pd.DataFrame(dfs)
            first_identity = df.iloc[0, 0]
            identity = os.path.basename(os.path.dirname(first_identity[0]))  # Extract the folder name
            return identity
        else:
            return "unknown"
    except Exception as e:
        print("Error in face recognition:", e)
        return "unknown"

def recognize_face_with_database(face, db_path):
    # Recognize the person's name from the face
    student_id = recognize_face(face, db_path)
    
    # If the student_id is unknown, return default values
    if student_id == "unknown":
        return "unknown", "None", "None", "None", "None"
    
    # Retrieve person's data from the database
    student_data = get_student_info(student_id)
    if student_data:
        first_name, last_name, academic_year, department = student_data
        return student_id, first_name, last_name, academic_year, department
    else:
        return "unknown", "None", "None", "None", "None"
