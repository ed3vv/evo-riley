from ultralytics import YOLO
import os
import shutil

#MAKE THIS WORK WITH MULTIPLE FOLDERS FOR INPUT

model = YOLO('models/best.pt')
input_folder = 'training_data/session_2025-12-27_18-35-33'
output_folder = 'auto_annotated'

os.makedirs(f'{output_folder}/images', exist_ok=True)
os.makedirs(f'{output_folder}/labels', exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.endswith(('.png', '.jpg')):
        continue
    
    img_path = f'{input_folder}/{filename}'
    results = model(img_path, conf=0.4)[0]
    
    # Copy image
    shutil.copy(img_path, f'{output_folder}/images/{filename}')
    
    # Save YOLO format labels
    label_name = filename.rsplit('.', 1)[0] + '.txt'
    with open(f'{output_folder}/labels/{label_name}', 'w') as f:
        for box in results.boxes:
            cls = int(box.cls)
            x, y, w, h = box.xywhn[0].tolist()  # normalized xywh
            f.write(f'{cls} {x} {y} {w} {h}\n')

print('Done. Upload auto_annotated/ to Roboflow')