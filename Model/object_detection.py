from sahi import AutoDetectionModel
import cv2


def initialize_detection_model(model_type, model_path, confidence_threshold, device='cpu'):
    return AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )

def update_detection_results(tracked_faces, detection_results, frame):
    for id, data in tracked_faces.items():
        x1, y1, x2, y2 = data["xyxy"]  # Access bounding box coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {id}, Recognized ID: {data["Recognized ID"]}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow(frame)
    return detection_results

