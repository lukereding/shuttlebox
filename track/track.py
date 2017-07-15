# load libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import progressbar
import pdb

def out_of_bounds(contour):
    M = cv2.moments(contour)
    cY = int(M["m01"] / M["m00"])
    if cY > 430 or cY < 60:
        return True
    else:
        return False

def distance_to_middle(contour):
    '''Calculate the distance between the contour and the middle of the tank along the x-axis.'''

    # hard-coded for stupid reassons
    middle_of_tank = (480, 272)

    # find the center of the contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # find distance
    dist = abs(cX - middle_of_tank[0])
    return dist

def get_video_info(capture):
    width, height = cap.get(3), cap.get(4)
    number_frames = int(cap.get(7))
    fps = cap.get(5)
    return width, height, number_frames, fps


def add_to_background(frame, old_background_image, amount = 0.05):
    """add frame to the background image called old_background_image, return updated image."""
    old_background_image = np.float32(old_background_image)
    cv2.accumulateWeighted(frame, old_background_image, amount)
    final = cv2.convertScaleAbs(old_background_image)
    return final

def find_background(capture):

    print("calculating the background image")

    bar = progressbar.ProgressBar()

    # get number of frames
    number_frames = int(capture.get(7))

    # get initial frames
    cap.set(1, 1)
    _, background = capture.read()

    for i in bar(range(1, 200)):
        capture.set(1, i)
        _, current_image = capture.read()
        background = add_to_background(current_image, background)
    return background

def test_video(capture):
    ret, frame = capture.read()
    if ret == False:
        sys.exit("OpenCV could not read this video.")

def get_aspect_ratio(contour):
    x,y,w,h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    return aspect_ratio

def draw_largest_contour(frame, previous_location, number_frames_without_fish):
    """Returns the frame with the largest contour drawn on it. """
    # take difference image, blur, convert to grey, threshold, find contours
    diff = cv2.subtract(frame, background)
    blurred = cv2.blur(diff, (5,5))
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # at this point, let's restrict our search for the fish
    if previous_location is not None or number_frames_without_fish < 20:
        height, width = gray.shape
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, previous_location, 100, 1, thickness=-1)
        cv2.circle(frame, previous_location, 100, 1, thickness=2)
        masked_data = cv2.bitwise_and(gray, gray, mask=circle_img)
    else:
        masked_data = gray

    _, thresh = cv2.threshold(masked_data, 20, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # we now add the step of excluding contours that are too big or too small
    contours_right_size = [c for c in contours if cv2.contourArea(c) < 700 and cv2.contourArea(c) > 20]

    # we also exclude contours that are too long
    contours_right_size_shape = [c for c in contours_right_size if get_aspect_ratio(c) < 10 and get_aspect_ratio(c) > 0.2]

    contours_filtered = [c for c in contours_right_size_shape if out_of_bounds(c) == False]

    # exclude contours that are too close to the bottom of the tank
    # contours_right_size_shape = [c for c in contours if get_aspect_ratio(c) < 10 and get_aspect_ratio(c) > 0.2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(contours_filtered) > 0:
        number_frames_without_fish = 0
        # if there are any contours remaining, we take the one closest to the middle of the tank
        c = min(contours_right_size_shape, key = distance_to_middle)
        cv2.drawContours(frame,[c],0,(124,23,199),-1)
        cv2.putText(frame, str(previous_location), (50,50), font, 1,(124,23,199),2,cv2.LINE_AA)
        M = cv2.moments(c)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            loc = (cx, cy)
        except:
            loc = previous_location
    else:
        number_frames_without_fish += 1
        cv2.putText(frame, str("ain't nobody home!!"), (200,200), font, 2,(124,23,199),2,cv2.LINE_AA)
        if number_frames_without_fish < 20:
            loc = previous_location
        else:
            loc = None
    return frame, loc, number_frames_without_fish



if __name__ == "__main__":

    # read in video, parse arguments
    cap = cv2.VideoCapture(sys.argv[1])

    # test to make sure opencv can read the video
    test_video(cap)

    # print things to the user
    width, height, number_frames, fps = get_video_info(cap)

    # find the background of the image
    background = find_background(cap)

    # Loop through frames
    frame_number = 1
    previous_location = None
    frames_missing = 0

    while frame_number < number_frames:
        print(frame_number)
        _, frame = cap.read()

        # update background image every x frames
        if frame_number % 200 ==  0:
            background = add_to_background(frame, background, 0.1)
            print("frame {}".format(frame_number))

        frame, previous_location, frames_missing = draw_largest_contour(frame, previous_location, frames_missing)

        frame_number += 1

        cv2.imshow('frame',frame)
        cv2.waitKey(20)
cv2.destroyAllWindows()
