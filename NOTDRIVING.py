import cv2
from gui_buttons import Buttons
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector



button = Buttons()
button.add_button("person", 0, 0)
button.add_button("cell phone", 200, 0)
button.add_button("orange", 0, 70)
button.add_button("banana", 0, 105)
button.add_button("spoon", 0, 140)
button.add_button("dog", 0, 175)
button.add_button("cup", 0, 210)
button.add_button("wine glass", 0, 245)
button.add_button("chair", 0, 280)
button.add_button("clock", 0, 315)
button.add_button("cat", 0, 350)
button.add_button("backpack", 0, 385)
colors = button.colors

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects list")
print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
detector = FaceMeshDetector(maxFaces=2)



def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)


cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

cv2.setMouseCallback("Frame", click_button)

while True:
    ret, frame = cap.read()


    active_buttons = button.active_buttons_list()

    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        color = colors[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, 255, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)


    button.display_buttons(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
