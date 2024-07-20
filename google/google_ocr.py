from google.cloud import vision
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    results = []
    for text in texts:
        description = text.description
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        results.append({"description": description, "bounds": vertices})
    if response.error.message:
        raise Exception(f'{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors')
    return results


def detect_handwriting(path):
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)  # Use document_text_detection for handwriting
    texts = response.text_annotations
    results = []
    for text in texts:
        description = text.description
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        results.append({"description": description, "bounds": vertices})
    if response.error.message:
        raise Exception(f'{response.error.message}\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors')
    return results

# detect_handwriting('PATH_TO_YOUR_IMAGE')

detect_text('./image.png')