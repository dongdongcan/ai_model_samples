import dlib
import cv2

face_detector = dlib.get_frontal_face_detector()
img = cv2.imread("./face.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_detector(gray)

for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

processed_img = "./processed_image.jpg"
cv2.imwrite(processed_img, img)
print("succ dump processed img to:", processed_img)

# cv2.imshow('Detected Faces', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
