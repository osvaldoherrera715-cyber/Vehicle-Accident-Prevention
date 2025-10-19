from ultralytics import YOLO
import os
import cv2

model_path = r"C:\Users\osval\Accident-Prevention\best.pt"
source_path = r"C:\Users\osval\Accident-Prevention\TrialWeb.jpg"
output_folder = r"C:\Users\osval\Accident-Prevention\InferenceResults"
os.makedirs(output_folder, exist_ok=True)

print("Loading YOLO model...")
model = YOLO(model_path)


print("Running inference with confidence=0.25...")

# Annotate (boxes + class names)
results = model.predict(
    source=source_path,
    imgsz=1280,
    conf=0.25,
   # device=0,          # GPU if available, else 'cpu'
    save=True,         # saves annotated image
    show=False,        # we will handle display
    save_dir=output_folder,
    name="TrialRun_conf25",
    exist_ok=True
)

# Display annotated image
for r in results:
    # r.orig_img is the original image
    # r.plot() returns the annotated image
    annotated_img = r.plot()  # this contains boxes + labels
    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection", 1280, 720)  # adjust window size
    cv2.imshow("YOLO Detection", annotated_img)
    cv2.waitKey(0)  # wait until a key is pressed

cv2.destroyAllWindows()
print(f"✅ Inference completed. Annotated image saved in: {output_folder}")



