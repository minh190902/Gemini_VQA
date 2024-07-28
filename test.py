from PIL import Image
import easyocr


image = Image.open('/home/kietha_inspirelab/Desktop/LLM/OCR/vietnamese/unseen_test_images/im1504.jpg')
reader = easyocr.Reader(lang_list=['vi'], recog_network='vi_custom')
result = reader.readtext(image)
ocr_text = "\n".join([res[1] for res in result])
print("Info of text in image: " + ocr_text)



