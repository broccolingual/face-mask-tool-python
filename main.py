import time
import sys
import os
from pathlib import Path

import cv2
import face_recognition as fr
import numpy as np


def loadImage(path):
    return fr.load_image_file(path)


def loadMaskImage(path):
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def detectFaceLocations(loadedImage, model=None):
    return fr.face_locations(loadedImage, model=model)


def drawFaceOutline(loadedImage, faceLocations):
    for (top, right, bottom, left) in faceLocations:
        cv2.rectangle(loadedImage, (left, top),
                      (right, bottom), (0, 0, 255), 2)
    return loadedImage


def drawFaceMask(loadedImage, maskImage, faceLocations):
    for (top, right, bottom, left) in faceLocations:
        resizedMaskImage = cv2.resize(
            maskImage, dsize=(2*(right - left), 2*(bottom - top)))
        loadedImage = putSprite_npwhere(
            loadedImage, resizedMaskImage, (int(left - 0.5*(right - left)), int(top - 0.5*(bottom - top))))
    return loadedImage


def putSprite_npwhere(loadedImage, resizedMaskImage, pos):
    x, y = pos
    fh, fw = resizedMaskImage.shape[:2]
    bh, bw = loadedImage.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x+fw, bw), min(y+fh, bh)

    if not ((-fw < x < bw) and (-fh < y < bh)):
        return loadedImage

    f3 = resizedMaskImage[:, :, :3]
    f_roi = f3[y1-y:y2-y, x1-x:x2-x]
    roi = loadedImage[y1:y2, x1:x2]
    tmp = np.where(f_roi == (0, 0, 0), roi, f_roi)
    loadedImage[y1:y2, x1:x2] = tmp
    return loadedImage


def displayImage(loadedImage, outputPath):
    loadedImage = cv2.cvtColor(loadedImage, cv2.COLOR_BGR2RGB)
    cv2.imwrite(outputPath, loadedImage)


def main(imagePath):
    start = time.time()
    loadedImage = loadImage(imagePath)
    maskImage = loadMaskImage("img/bear.png")
    faceLocations = detectFaceLocations(loadedImage, model="cnn")
    print("Detected face locations")
    for i, k in enumerate(faceLocations):
        print(f"{i+1}. {k}")
    image = drawFaceMask(loadedImage, maskImage, faceLocations)
    os.makedirs("output", exist_ok=True)
    displayImage(image, f"output/masked_{Path(imagePath).name}")
    print("Successfully output image.")
    elapsed_time = time.time() - start
    print(f"Elapsed Time: {round(elapsed_time, 2)}s")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(1)
    try:
        main(sys.argv[1])
    except Exception as e:
        print(e)
