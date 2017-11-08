# load libraries
import cv2
import numpy as np
import pandas as pd

import sys
import pdb
from pathlib import Path
import json

import progressbar
import subprocess

# create a hard-coded dict that defines the contours of each zone
zones = {
    "LT_top": np.array([[88, 67], [461, 67], [464, 143], [200, 146], [203, 184], [94, 179]], dtype = np.int32),
    "LS": np.array([[94, 179], [203, 184], [216, 477], [108, 518]], dtype = np.int32),
    "LT_bottom": np.array([[108, 518], [216, 477], [222, 550], [477, 538], [481, 614], [110, 632]], dtype = np.int32),
    "LE": np.array([[222, 550], [477, 538], [464, 143], [200, 146], [203, 184], [216, 477]], dtype = np.int32),
    "CE": np.array([[464, 143], [779, 142], [792, 524], [477, 538]], dtype = np.int32),
    "CT_top": np.array([[461, 67], [775, 68], [779, 142], [464, 143]], dtype = np.int32),
    "CT_bottom": np.array([[477, 538], [481, 614], [795, 601], [792, 524]], dtype = np.int32),
    "RT_top": np.array([[775, 68], [779, 142], [1061, 139], [1062, 188], [1183, 161], [1180, 69]], dtype = np.int32),
    "RT_bottom": np.array([[792, 524], [795, 601], [1191, 583], [1186, 479], [1068, 467], [1068, 511]], dtype = np.int32),
    "RE": np.array([[779, 142], [1061, 139], [1068, 511], [792, 524]], dtype = np.int32),
    "RS": np.array([[1062, 188], [1183, 161], [1186, 479], [1068, 467]], dtype = np.int32)
}

def print_frame(frame_number):
    print("exiting on frame number {}".format(frame_number))

def check_zones(background, zone_dict):
    """Print background with the zones overlaid."""
    for zone, contour in zone_dict.items():
        print(zone)
        print(contour)
        # generate random color
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.drawContours(background, [contour], 0, color , -1)
    return background

def get_zone(point, zone_dict):
    """Return the zone the fish is currently in. Point must be a tuple."""
    if point is None:
        return "unknown"

    # loop through the dict
    location = "unknown"
    for zone, countour in zone_dict.items():
        dist = cv2.pointPolygonTest(countour, point, False)
        # if positive, point is inside the countour
        if dist > 0:
            location = zone
            break
    # hack off the "top" or "bottom"
    # note that if the point in not within any of the zones,
    # or at the border of zones,
    # this returns "unknown"
    return location.split("_")[0]

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
    if y > 600 or y < 110 or x > 1150 or x < 130:
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

def get_video_info(capture, video_name, save_json = True):
    width, height = cap.get(3), cap.get(4)
    number_frames = int(cap.get(7))
    fps = cap.get(5)

    if save_json:
        data = {
        'name' : video_name,
        'width' : width,
        'height' : height,
        'number_frames' : number_frames,
        'fps' : fps
        }
        with open("{}.txt".format(video_name), 'w') as outfile:
            json.dump(data, outfile)

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
    diff = cv2.subtract(frame, background)
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

    # cv2.imshow('thresh', thresh)
    # cv2.imshow('gray', diff)

    # we now add the step of excluding contours that are too big or too small
    contours_right_size = [c for c in contours if cv2.contourArea(c) < 1000 and cv2.contourArea(c) > 150]

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

    # record video of the tracked output?
    record = True

    # get video name
    name = filename.split('.', 1)[-2]

    # test to make sure opencv can read the video
    test_video(cap)

    # print things to the user
    width, height, number_frames, fps = get_video_info(cap, name)

    print("""
    video size: {} x {}
    total number of frames: {}
    fps: {}
    """.format(width, height, number_frames, fps))

    if record:
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
    for i in bar(range(1, number_frames - 2, 10)):

        # set the frame number to i and grab the frame
        frame_number = i
        cap.set(1) = frame_number

        _, frame = cap.read()

        # update background image every x frames
        if frame_number % 150 ==  0:
            background = add_to_background(frame, background, 0.05)
            # print("frame {}".format(frame_number))

        if frame_number % 200 == 0:
            save_data(df, frame_number-2, name)

        frame, previous_location, frames_missing = draw_largest_contour(frame, previous_location, frames_missing)

        # zone_location = get_zone(previous_location ,zones)

        # add location to DataFrame
        if frames_missing > 0:
            df.loc[frame_number] = [frame_number, np.nan, np.nan, "missing"]
        elif frames_missing == 0:
            df.loc[frame_number] = [frame_number, previous_location[0], previous_location[1], "found"]

        previous_frame = frame

        if record:
            out.write(frame)

        # cv2.imshow('frame',frame)
        cv2.waitKey(5)
cv2.destroyAllWindows()
if record:
    out.release()
save_data(df, None, name)

# this is stupid, but at the end, read back in the dataframe, interpolate points, find zones, then save as new '_interpolated' dataframe

# read in data frame
df = pd.read_csv("{}.csv".format(name))

# interpolate x and y point
df = df.interpolate()

# get the zone for each point
zone_list = [None] * df.shape[0]
for index, row in df.iterrows():
    point = (row['x'], row['y'])
    zone = get_zone(point, zones)
    zone_list[index] = zone

# add zone to DataFrame
df['zone'] = zone_list

# write the csv
csv_name = "{}_interpolated.csv".format(name)
df.to_csv(csv_name)

# plot the results
bashCommand = "Rscript plot_tracks.R {}".format(csv_name)
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
