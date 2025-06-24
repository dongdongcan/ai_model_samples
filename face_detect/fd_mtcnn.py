from mtcnn.mtcnn import MTCNN
import cv2

face_detector = MTCNN()

img = cv2.imread("./face.jpeg")

faces = face_detector.detect_faces(img)

for face in faces:
    x, y, w, h = face["box"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for key, value in face["keypoints"].items():
        cv2.circle(img, value, 2, (0, 255, 0), -1)

processed_img = "./processed_image.jpg"
cv2.imwrite(processed_img, img)
print("succ dump processed img to:", processed_img)

# cv2.imshow('Detected Faces', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
