# load libraries
import cv2
import numpy as np
import pandas as pd

import sys
import pdb
from pathlib import Path

import progressbar
import subprocess


def get_centroid(contour):
    try:
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx, cy)
    except:
        return None

def out_of_bounds(contour):
    x,y = get_centroid(contour)
    if y > 550 or y < 140 or x > 1150 or x < 150:
        return True
    else:
        return False

def distance_to_middle(contour):
    '''Calculate the distance between the contour and the middle of the tank along the x-axis.'''

    # hard-coded for stupid reassons
    middle_of_tank = (960, 540)

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

def find_background(capture, save_to_disk = True):

    print("calculating the background image")

    bar = progressbar.ProgressBar()

    # get number of frames
    number_frames = int(capture.get(7))

    # get initial frames
    cap.set(1, 1)
    _, background = capture.read()

    for i in bar(range(1, 2000, 10)):
        capture.set(1, i)
        _, current_image = capture.read()
        background = add_to_background(current_image, background)

    if save_to_disk:
        cv2.imwrite("background.jpg", background)
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
    diff = cv2.subtract(background, frame)
    blurred = cv2.blur(diff, (7,7))
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # at this point, let's restrict our search for the fish
    if previous_location is not None or number_frames_without_fish < 80:
        height, width = gray.shape
        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, previous_location, 100, 1, thickness=-1)
        cv2.circle(frame, previous_location, 100, 1, thickness=2)
        masked_data = cv2.bitwise_and(gray, gray, mask=circle_img)
    else:
        masked_data = gray

    _, thresh = cv2.threshold(masked_data, 10, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    #
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('gray', diff)

    # we now add the step of excluding contours that are too big or too small
    contours_right_size = [c for c in contours if cv2.contourArea(c) < 1000 and cv2.contourArea(c) > 100]

    # we also exclude contours that are too long
    contours_right_size_shape = [c for c in contours_right_size if get_aspect_ratio(c) < 10 and get_aspect_ratio(c) > 0.2]

    # or near the tank edges
    contours_filtered = [c for c in contours_right_size_shape if out_of_bounds(c) == False]

    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(contours_filtered) > 0:
        number_frames_without_fish = 0
        # if there are any contours remaining, we take the one closest to the middle of the tank
        best_guess = min(contours_filtered, key = distance_to_middle)
        cv2.drawContours(frame,[best_guess],0,(124,23,199),-1)
        M = cv2.moments(best_guess)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            loc = (cx, cy)
        except:
            loc = previous_location
        cv2.putText(frame, str(loc), (50,50), font, 1,(124,23,199),2,cv2.LINE_AA)
    else:
        number_frames_without_fish += 1
        # cv2.putText(frame, str("ain't nobody home!!"), (200,200), font, 2,(124,23,199),2,cv2.LINE_AA)
        if number_frames_without_fish < 80:
            loc = previous_location
        else:
            loc = None

    return frame, loc, number_frames_without_fish

def create_empty_df(number_rows):
    df = pd.DataFrame(index=np.arange(0, number_rows), columns=('frame', 'x', 'y', 'fish') )
    return df

def save_data(df, max_row, name):
    if max_row is not None:
        df = df.iloc[1:max_row]
    try:
        df = df.interpolate()
    except:
        x = "oops"
        # print("too many NAs to interpolate")

    df.to_csv("{}.csv".format(name))

if __name__ == "__main__":

    # read in video, parse arguments
    filename = sys.argv[1]
    print("using video: {}".format(filename))
    cap = cv2.VideoCapture(filename)

    # get video name
    name = filename.split('.', 1)[-2]

    # test to make sure opencv can read the video
    test_video(cap)

    # print things to the user
    width, height, number_frames, fps = get_video_info(cap)

    print("""
    video size: {} x {}
    total number of frames: {}
    fps: {}
    """.format(width, height, number_frames, fps))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('{}_tracked.avi'.format(name), fourcc, 30, (int(width), int(height)), True)

    # create empty dataframe to populate
    df = create_empty_df(number_frames)

    # find the background of the image
    background = find_background(cap)

    # Loop through frames
    frame_number = 1
    previous_location = None
    frames_missing = 0

    # reset capture index
    cap.set(1, frame_number)

    print("\n\ntracking the fish")

    bar = progressbar.ProgressBar()

    # while frame_number < number_frames:
    for i in bar(range(1, number_frames - 2)):
        frame_number = int(cap.get(1))

        _, frame = cap.read()

        # update background image every x frames
        if frame_number % 50 ==  0:
            background = add_to_background(frame, background, 0.05)
            # print("frame {}".format(frame_number))

        if frame_number % 200 == 0:
            save_data(df, frame_number-2, name)

        # print("previous location: {}".format(previous_location))
        frame, previous_location, frames_missing = draw_largest_contour(frame, previous_location, frames_missing)

        # add location to DataFrame
        if frames_missing > 0:
            df.loc[frame_number] = [frame_number, np.nan, np.nan, "missing"]
        elif frames_missing == 0:
            df.loc[frame_number] = [frame_number, previous_location[0], previous_location[1], "found"]

        previous_frame = frame

        out.write(frame)

        # cv2.imshow('frame',frame)
        cv2.waitKey(5)
cv2.destroyAllWindows()
save_data(df, None, name)

# this is stupid, but:
df = pd.read_csv("{}.csv".format(name))
df = df.interpolate()
csv_name = "{}_interpolated.csv".format(name)
df.to_csv(csv_name)
# plot the results
bashCommand = "Rscript plot_tracks.R {}".format(csv_name)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
