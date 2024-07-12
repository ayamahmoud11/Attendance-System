import os
from datetime import datetime
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from Model.recognize_face import recognize_face_with_database
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def send_email_with_photo(image_path, recognized_id):
    # Email configurations
    sender_email = 'espaya580@gmail.com'
    receiver_email = 'ayamahmoudk677@gmail.com'
    password = 'pxqz qwyj bkfc uypj'

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Unknown Student Detected'

    # Attach image
    with open(image_path, 'rb') as fp:
        img = MIMEImage(fp.read())
    img.add_header('Content-Disposition', 'attachment', filename=f'{recognized_id}_photo.jpg')
    msg.attach(img)

    # Send email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:  # Update SMTP server address
        smtp.starttls()
        smtp.login(sender_email, password)
        smtp.send_message(msg)
        smtp.quit()


def process_frame(frame, detection_model, tracker, tracked_faces, detection_results, db_path, csv_file):
    # Previous frame's detection results count
    prev_detection_count = len(tracked_faces)

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    num_predictions = len(result.object_prediction_list)
    dets = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(result.object_prediction_list):
        dets[ind, :4] = np.array(object_prediction.bbox.to_xyxy(), dtype=np.float32)
        dets[ind, 4] = object_prediction.score.value
        dets[ind, 5] = object_prediction.category.id

    tracks = tracker.update(dets, frame)

    if tracks.shape[0] != 0:
        xyxys = tracks[:, 0:4].astype('int')  # float64 to int
        ids = tracks[:, 4].astype('int')  # float64 to int
        confs = tracks[:, 5].round(decimals=2)
        clss = tracks[:, 6].astype('int')  # float64 to int

        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            recognized_id = None  # Initialize recognized_id with None
            # if new person
            if id not in tracked_faces:
                tracked_faces[id] = {"First Detection Time": datetime.now(), "Recognized ID": None, "xyxy": xyxy}

            # If detection count increased
            if len(tracked_faces) > prev_detection_count:
                face = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                recognized_id, first_name, last_name, academic_year, department = recognize_face_with_database(face, db_path)
                tracked_faces[id]["Recognized ID"] = recognized_id

                if recognized_id == "unknown":
                    # Send alert email
                    cv2.imwrite('unknown_person.jpg', face)  # Save the unknown face
                    send_email_with_photo('unknown_person.jpg', recognized_id)

                new_detection_result = pd.DataFrame({
                    "ID": [id],
                    "Recognized ID": [recognized_id],
                    "First name": [first_name],
                    "Last name": [last_name],
                    "Academic_year": [academic_year],
                    "Department": [department],
                    "First Detection Time": [tracked_faces[id]["First Detection Time"]],
                    "Last Detection Time": [datetime.now()],
                    "Confidence": [conf],
                    "Left Class": [False]
                })
                detection_results = pd.concat([detection_results, new_detection_result], ignore_index=True)

            # Update the frame with bounding box and recognized ID
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {id}, Recognized ID: {recognized_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    detection_results.to_csv(csv_file, index=False)
    return tracked_faces, detection_results



