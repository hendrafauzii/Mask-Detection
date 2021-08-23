import cv2
import numpy as np
import tensorflow as tf

def face_detection(model, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,
                                 1.0,
                                 (224, 224),
                                 (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    locs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            startX, startY = max(0, startX), max(0, startY)
            endX, endY     = min(w - 1, endX), min(h - 1, endY)
            locs.append((startX, startY, endX, endY))

    return locs


def mask_detection(model, input_details, output_details, frame, xmin, ymin, xmax, ymax):
    img_face = frame[ymin:ymax, xmin:xmax]
    img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
    img_face = cv2.resize(img_face, (50, 50))
    img_face = img_face.astype(np.float32)
    img_face /= 255.

    model.set_tensor(input_details[0]['index'], [img_face])
    model.invoke()
    output = model.get_tensor(output_details[0]['index'])

    return 'Without Mask' if output[0] > 0.5 else 'With Mask'


# Load Mask Detection Model
maskNet = tf.lite.Interpreter(model_path = 'mask_model.tflite')
maskNet.allocate_tensors()
input_details = maskNet.get_input_details()
output_details = maskNet.get_output_details()

# Load Face Detection Model
prototxtPath = r"face_model/deploy.prototxt"
weightsPath = r"face_model/face_model.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False:
        print('Failed opened Camera')
        break

    bbox = face_detection(faceNet, frame)
    for xmin, ymin, xmax, ymax in bbox:
        label = mask_detection(maskNet, input_details, output_details, frame, xmin, ymin, xmax, ymax)
        if label == 'With Mask':
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()