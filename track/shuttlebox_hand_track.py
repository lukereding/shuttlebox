import cv2
import numpy as np
import pandas as pd
import sys
import os
'''

script for analyzing shuttlebox videos

useage: python shuttlebox_hand_track.py path_to_video

'''

def check_video(x):
    if not os.path.exists(x):
        sys.exit("{} does not exist. exiting.".format(x))

def read_video(video_filename):
    cap = cv2.VideoCapture(video_filename)

    # check the fps
    fps = cap.get(5)
    if fps != 30:
        sys.exit("tell kelly that video {} is {} fps.".format(video_filename, fps))
    return cap

def write_results(location_tuples, name, path):
    df = pd.DataFrame(pd.Series([x[0] for x in location_tuples]), columns = ['x'])
    y = pd.Series([x[1] for x in location_tuples])
    df['y'] = y
    df['frame_number'] = pd.Series(range(1, len(y) + 1))

    name = str(name) + '.csv'
    path_to_save = os.path.join(path, name)
    # print(path_to_save)
    df.to_csv(path_to_save)

def display_frame(capture, frame_number):

    locs = []

    def draw_circles(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(frame, (x,y), 8, (62, 80, 204), -1)
            return locs.append((x,y))

    capture.set(1, frame_number)
    _, frame = cap.read()

    window_name = "frame {}".format(frame_number)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_circles)

    # show the frame
    while True:
        cv2.imshow(window_name,frame)
        if cv2.waitKey(20) & 0xFF == 32 and locs != []:
            cv2.destroyAllWindows()
            return locs[0]


def csv_exists(filename, path):
    if os.path.exists(os.path.join(filename, path)):
        return True
    else:
        return False


if __name__ == "__main__":

    # read in the video name
    video_filename = sys.argv[1]

    startframe_arg = sys.argv[2]
    start_frame = int(startframe_arg)

    check_video(video_filename)
    cap = read_video(video_filename)

    path, filename = os.path.split(video_filename)

    # see whether the csv file already exists
    if csv_exists(filename, path):
        df = pd.read_csv(os.path.join(path, filename))
        x = df['x']
        y = df['y']
        frames = df['frame_number']
        loctions = list(zip(x,y))
        frames_scored = frames[-1] * 150
        start_frame = start_frame + frames_scored
        print("\nyou are starting to score on frame {}\n".format(start_frame))
    else:
        locations = []

    total_frames = int(cap.get(7))
    print("there are {} total frames".format(total_frames))

    counter = 0

    for frame_number in range(start_frame, total_frames, 150):
        try:
            location_current_frame = display_frame(cap, frame_number + 1)
        except:
            location_current_frame = (NA, NA)

        locations.append(location_current_frame)

        counter += 1

        if counter % 50 == 0 and frame_number != 0:
            write_results(locations, filename.split(".")[-2], path)
    write_results(locations, filename.split(".")[-2], path)
