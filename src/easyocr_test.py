import easyocr
import cv2

reader = easyocr.Reader(['en'])
path = '/home/loki/Downloads/61Dq3LRei9L.jpg'


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image
image = cv2.imread(path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (0, 0), 3)
sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
sharpened_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
image = sharpened_bgr

results = reader.readtext(image)
print(results)

for (bbox, text, prob) in results:
    top_left = tuple([int(val) for val in bbox[0]])
    bottom_right = tuple([int(val) for val in bbox[2]])

    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 6)

output = image
output = cv2.resize(output, (1280, 960))
cv2.imshow('Image with Bounding Boxes', output)
cv2.waitKey(0)  
cv2.destroyAllWindows()
