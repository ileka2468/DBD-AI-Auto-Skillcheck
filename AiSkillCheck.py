import cv2 as cv
from WinCap import WindowCapture
from time import time
from fastai.vision.all import ImageDataLoaders, get_image_files, load_learner, Resize


path = r"./good"
def is_frame(x): return x[0].isupper()

dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_frame, item_tfms=Resize(224))

# Load the exported model you can train your own model with the images provided oor use the model I provided.
model_path = "good/dbdPklv2.pkl"
learn_inf = load_learner(model_path)


# If using actual DBD game use the process checking function inside the winCap class and run it to find process name. link to online skil checks: https://www.mistersyms.com/tools/gitgud/index
wincap = WindowCapture("Skill Check Simulator - Google Chrome")
loop_time = time()
while True:

    screenshot = wincap.get_Screen()

    print(screenshot.shape)
    # You may have to adjust the image cropping to center the skill check on your screen this should work for most 1080p screens
    cropped = screenshot[440:600, 860: 1060]
    cv.imshow('Computer Vision', cropped)

    is_frame, _, probabilities = learn_inf.predict(cropped)

    # Extract the probability of the "frame" class
    frame_prob = probabilities[1].item()

    # Convert the probability to a percentage with two decimal places
    frame_prob_pct = round(frame_prob * 100, 2)


    # print(f"Is frame? {is_frame}, Probability it's a frame: {frame_prob_pct}%")

    if frame_prob_pct >= 95:
        print("Push NOW")
        # here you can either interface an arduino solution to physically push the space bar or you can just use a python keyboard controller like pyAutoGUI or the built in keyboard library

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break