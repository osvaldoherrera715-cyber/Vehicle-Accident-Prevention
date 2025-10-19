from ultralytics import YOLO
import os

# -----------------------------
# 1️⃣ Paths
# -----------------------------
model_path = r"C:\Users\osval\Accident-Prevention\best.pt"  # your trained weights
source_path = r"C:\Users\osval\Accident-Prevention\TrialRun.jpg"  # video or image

# Base output folder
output_base = r"C:\Users\osval\Accident-Prevention\InferenceResults"
os.makedirs(output_base, exist_ok=True)

# -----------------------------
# 2️⃣ Load the trained model
# -----------------------------
model = YOLO(model_path)

# -----------------------------
# 3️⃣ Confidence intervals to test
# -----------------------------
conf_values = [0.25, 0.35, 0.45, 0.55]  # adjust as needed

# -----------------------------
# 4️⃣ Run inference for each confidence
# -----------------------------
for conf in conf_values:
    print(f"🔹 Running inference with confidence={conf}")
    
    # Create folder for this confidence
    output_folder = os.path.join(output_base, f"conf_{int(conf*100)}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Run YOLOv11 prediction
    model.predict(
        source=source_path,
        imgsz=1280,        # match training
        conf=conf,
        #device=0,          # set 0 for GPU, or 'cpu' if no CUDA
        save=True,
        show=False,        # set True to display each frame
        save_dir=output_folder  # save results in this folder
    )

print("✅ Inference completed for all confidence levels!")

