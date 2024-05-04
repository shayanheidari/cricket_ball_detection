import cv2
import time
import os

# setting up some hyper parameters
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
classes = 80
ball_class = 32
video_name = "0"

# reading class names from yolov4 class names file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# reading video with opencv VideoCapture module
cap = cv2.VideoCapture(f"../test_data/{video_name}.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

# loading the neural network with readNet method from dnn module in opencv
net = cv2.dnn.readNet("yolov4/yolov4.weights", "yolov4/yolov4.cfg")
# set preferable backends to gpu
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

if os.path.exists("results"):
    pass
else:
    os.mkdir("results")
result = cv2.VideoWriter(f"results/{video_name}_result.mp4",
                         cv2.VideoWriter_fourcc(*"avc1"),
                         10, size)

# building the model from neural net
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# run object detection
while cv2.waitKey(1) < 1:
    (grabbed, frame) = cap.read()
    if not grabbed:
        exit()
    start = time.time()
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    # set up detection box around detected objects
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        # draw box only if detected object is our target class
        if classid == ball_class:
            label = f"{class_names[classid]} : {score:.2f}"
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # setting up FPS
    fps = f"FPS: {(1 / (end - start))}"
    cv2.putText(frame, fps, (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    result.write(frame)
    cv2.imshow("output", frame)

cap.release()
result.release()
cv2.destroyAllWindows()

print("The video was successfully saved.")
