from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED
import glob
import math
import os
from pathlib import Path
import sys
import time

import cv2
import face_recognition as fr
import numpy as np
from PIL import Image


def loadImage(path):
    return fr.load_image_file(path)


def loadImageToGreyscale(path):
    return cv2.cvtColor(fr.load_image_file(path), cv2.COLOR_BGR2GRAY)


def loadMaskImage(path):
    return cv2.cvtColor(cv2.imread(path, -1), cv2.COLOR_BGR2RGB)


def smallResizeImage(loadedImage, ratio=0.5):
    return cv2.resize(loadedImage, dsize=None, fx=ratio, fy=ratio)


def detectFaceLocations(loadedImage, model=None, ratio=0.5):
    return fr.face_locations(smallResizeImage(loadedImage, ratio=ratio), model=model, number_of_times_to_upsample=1)


def detectFaceLandmarks(loadedImage, model=None):
    return fr.face_landmarks(loadedImage, model=model)


def drawFaceOutline(loadedImage, faceLocations, ratio=0.5):
    r_ratio = int(np.reciprocal(ratio))
    for (top, right, bottom, left) in faceLocations:
        top *= r_ratio
        right *= r_ratio
        bottom *= r_ratio
        left *= r_ratio

        cv2.rectangle(loadedImage, (left, top),
                      (right, bottom), (0, 0, 255), 2)
    return loadedImage


def drawEyePoints(loadedImage, faceLandmarks):
    for faceLandmark in faceLandmarks:
        cv2.line(
            loadedImage, calcEyePointCenter(
                faceLandmark["left_eye"]),
            calcEyePointCenter(
                faceLandmark["right_eye"]), (0, 255, 0), 2)
        angle = calcFaceAngleFromEyePoints(calcEyePointCenter(
            faceLandmark["left_eye"]), calcEyePointCenter(faceLandmark["right_eye"]))
        print(f"Angle: {round(angle, 1)}°")
    return loadedImage


def calcEyePointCenter(eye_point: list) -> tuple:
    x = int(0.5*(eye_point[0][0] + eye_point[1][0]))
    y = int(0.5*(eye_point[0][1] + eye_point[1][1]))
    return (x, y)


def calcFaceAngleFromEyePoints(left_eye: tuple, right_eye: tuple):
    x1, y1 = left_eye
    x2, y2 = right_eye
    return np.arctan((y1 - y2)/(x1 - x2))*180/math.pi


def drawFaceMask(loadedImage, maskImage, faceLocations, ratio=0.5):
    r_ratio = int(np.reciprocal(ratio))
    for (top, right, bottom, left) in faceLocations:
        top *= r_ratio
        right *= r_ratio
        bottom *= r_ratio
        left *= r_ratio

        resizedMaskImage = cv2.resize(
            maskImage, dsize=(2*(right - left), 2*(bottom - top)))
        loadedImage = putSprite_npwhere(
            loadedImage, resizedMaskImage, (int(left - 0.5*(right - left)), int(top - 0.5*(bottom - top))))
    return loadedImage


def convertPILToCV2(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def convertCV2ToPIL(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def isEyePointsInside(faceOutline: tuple, left_eye: tuple, right_eye: tuple) -> bool:
    top, right, bottom, left = faceOutline
    if (left < left_eye[0] < right) and (top < left_eye[1] < bottom) and (left < right_eye[0] < right) and (top < right_eye[1] < bottom):
        return True
    else:
        return False


def drawFaceAngledMask(loadedImage, maskImage, faceLocations, faceLandmarks, ratio=0.5):
    r_ratio = int(np.reciprocal(ratio))
    for i, (top, right, bottom, left) in enumerate(faceLocations):
        top *= r_ratio
        right *= r_ratio
        bottom *= r_ratio
        left *= r_ratio

        targetFaceLandmark = None

        # 顔の位置に合致する顔の特徴量があるかを判定
        for faceLandmark in faceLandmarks:
            if isEyePointsInside((top, right, bottom, left), faceLandmark["left_eye"][0], faceLandmark["right_eye"][0]):
                targetFaceLandmark = faceLandmark

        # 該当する顔の特徴量があれば、それに基づいてマスクの角度を回転して描画
        if targetFaceLandmark is not None:
            faceAngle = calcFaceAngleFromEyePoints(calcEyePointCenter(
                targetFaceLandmark["left_eye"]), calcEyePointCenter(targetFaceLandmark["right_eye"]))
            resizedMaskImage = cv2.resize(
                maskImage, dsize=(2*(right - left), 2*(bottom - top)))
            rotatedMaskImage = convertCV2ToPIL(
                resizedMaskImage).rotate(-faceAngle)  # PILの回転関数を使う為に変換
            loadedImage = putSprite_npwhere(
                loadedImage, convertPILToCV2(rotatedMaskImage), (int(left - 0.5*(right - left)), int(top - 0.5*(bottom - top))))  # 透過PNG用処理
            print(f"{i+1}. {(top, right, bottom, left)} | {round(-faceAngle, 1)}°")

        # 顔の特徴量が不足している場合は、角度を付けずにマスクを合成
        else:
            resizedMaskImage = cv2.resize(
                maskImage, dsize=(2*(right - left), 2*(bottom - top)))
            loadedImage = putSprite_npwhere(
                loadedImage, resizedMaskImage, (int(left - 0.5*(right - left)), int(top - 0.5*(bottom - top))))
            print(f"{i+1}. {(top, right, bottom, left)} | unknown")

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


def main(imagePath, model="hog"):
    os.makedirs("output", exist_ok=True)
    loadedImage = loadImage(imagePath)
    loadedGreyImage = loadImageToGreyscale(imagePath)
    maskImage = loadMaskImage("mask.png")
    faceLocations = detectFaceLocations(
        loadedGreyImage, model=model, ratio=0.5)
    faceLandmarks = detectFaceLandmarks(
        loadedGreyImage, model="small")

    print("-"*20 + "\n")
    print(Path(imagePath).name, loadedImage.shape)
    if len(faceLocations) == 0:
        print("No face was detected.\n")
    else:
        print(
            f"{len(faceLocations)} face detected. {len(faceLandmarks)} face landmark detected.")
        image = drawFaceAngledMask(
            loadedImage, maskImage, faceLocations, faceLandmarks, ratio=0.5)
        displayImage(image, f"output/masked_{Path(imagePath).name}")
        print("Successfully output image.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(1)
    try:
        start = time.time()
        img_list = glob.glob(
            f"{sys.argv[1]}/*.png") + glob.glob(f"{sys.argv[1]}/*.jpg")

        # single
        for img in img_list:
            main(img, model="cnn")

        # multi process
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     tasks = [executor.submit(main, img, model="hog")
        #              for img in img_list]
        #     wait(tasks, return_when=ALL_COMPLETED)
        #     print('All tasks completed.')
        elapsed_time = time.time() - start
        print(f"Elapsed Time: {round(elapsed_time, 1)}s")
    except Exception as e:
        print(e)
