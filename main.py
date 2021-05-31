
import numpy as np

# Requires OpenCV 3.4.2 or later version
import cv2

# Load weights and config files
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Read classes names
with open('coco.names', 'r') as f:
	classes = f.read().splitlines()

# print(classes)

# Read image
img = cv2.imread('image.jpg')
height, width, _ = img.shape

# Rescale image and convert to 4D Blob
blob = cv2.dnn.blobFromImage(img, 1./255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# for b in blob:
# 	for n, img_blob in enumerate(b):
# 		cv2.imshow(str(n), img_blob)

# Getting Yolo outputs from the last layer
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		class_id = np.argmax(scores)
		confidence = scores[class_id]

		# Only consider if detection confidence is greater than 50%
		if confidence > 0.5:
			center_x = int(detection[0] * width)
			center_y = int(detection[1] * height)
			w = int(detection[2] * width)
			h = int(detection[3] * height)

			x = int(center_x - w/2)
			y = int(center_y - h/2)

			boxes.append([x, y, w, h])
			confidences.append((float(confidence)))
			class_ids.append(class_id)

# print(len(boxes))

# Removes overlapping boxes that might be detecting the same object
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# print(indexes.flatten())

font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

# Display rectangle, class and confidence score on the image
for i in indexes.flatten():
	x, y, w, h = boxes[i]
	label = str(classes[int(class_ids[i])])
	confidence = str(round(confidences[i], 2))
	color = colors[i]
	cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
	cv2.putText(img, label + ' ' + confidence, (x, y - 10), font, 0.4, (255, 255, 255), 1)

# Display image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
