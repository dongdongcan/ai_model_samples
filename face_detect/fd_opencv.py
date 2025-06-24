import cv2

local_file_path = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(local_file_path)

img = cv2.imread("./face.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

processed_img = "./processed_image.jpg"
cv2.imwrite(processed_img, img)
print("succ dump processed img to:", processed_img)

# cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# cv2.imshow('Detected Faces', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
