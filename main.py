import cv2
import matplotlib.pyplot as plt

config_file = "model_config/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Class labels nuskaitymas į sąrašą
classLabels = []
label_file = "Labels.txt"
with open(label_file, "r") as f:
    classLabels.append(f.read().rstrip("\n").split("\n"))
    classLabels = classLabels[0]

# Modelio nustatymai
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Pasirenkamas vaizdo šaltinis
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    vid = cv2.VideoCapture(0)
if not vid.isOpened():
    raise IOError("Cannot open camera")

font_scale = 4
font = cv2.FONT_HERSHEY_PLAIN

# Paleidžiamas ciklas skaitantis ir klasifikuojantis kiekvieną video kadrą
while True:
    ret, frame = vid.read()

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)

    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()