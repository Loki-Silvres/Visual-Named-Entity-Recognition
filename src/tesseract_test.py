import pytesseract
from PIL import Image
img= "/home/loki/AmazonML/data/images/test/A12Zn7WdVeL.jpg"
text = pytesseract.image_to_string(Image.open(img))
print(text)