# get a black image
# name it black.jpg
# make it the right size (same as the video)

ffmpeg -i black.jpg -vf scale=960x544 -y black_sized.jpg

# make a 1 second video of black
ffmpeg -framerate 1/1 -i black_sized.jpg -vcodec libx264 -r 30 -y black_1_s.mp4 &

## the shoal and felt videos are different sizes. no good!
## we also need to loop the felt video and make it 5 min long (so loop it five times)

ffmpeg -ss 00:00:02 -t 00:01:10 -i empty_shoal_6_use_this.mov -vcodec libx264 -r 30 -vf scale=960:544 -pix_fmt yuv420p -y empty.mp4 &

wait %1 %2

# loop the shoal stimulis video
echo "file 'empty.mp4'" > loopEmpty
for i in {1..4}; do printf "file '%s'\n" "empty.mp4" >> loopEmpty; done
ffmpeg -f concat -i loopEmpty -y empty_looped.mp4
rm loopEmpty


# cut down the shoal video to various lengths
ffmpeg -ss 00:00:00 -t 00:05:00 -i empty_looped.mp4 -y empty_300_s.mp4 &
ffmpeg -ss 00:00:00 -t 00:01:51 -i empty_looped.mp4 -y empty_111_s.mp4 &
ffmpeg -ss 00:00:00 -t 00:01:30 -i empty_looped.mp4 -y empty_90_s.mp4 &
ffmpeg -ss 00:00:00 -t 00:03:21 -i empty_looped.mp4 -y empty_201_s.mp4 &

wait %1 %2 %3 %4

# make the shoal video into a nice format
ffmpeg -i shoal_stimulus_5_20sec_clip_for_Luke.mov -vcodec libx264 -r 30 -pix_fmt yuv420p -y shoal.mp4

echo concatenating video 1

echo "file 'empty_300_s.mp4'" > conCatFile1
for i in {1..15}; do printf "file '%s'\n" "black_1_s.mp4" >> conCatFile1; printf "file '%s'\n" "shoal.mp4" >> conCatFile1; printf "file '%s'\n" "empty_201_s.mp4" >> conCatFile1; done
ffmpeg -f concat -i conCatFile1 -y final1.mp4 &


echo concatenating video 2

echo "file 'empty_300_s.mp4'" > conCatFile2
<<<<<<< HEAD
for i in {1..15}; do printf "file '%s'\n" "empty_111_s.mp4" >> conCatFile2; printf "file '%s'\n" "black_1_s.mp4" >> conCatFile2; printf "file '%s'\n" "shoal.mp4" >> conCatFile2; printf "file '%s'\n" "empty_90_s.mp4" >> conCatFile2; done
=======
for i in {1..15}; do printf "file '%s'\n" "empty_111_s.mp4" >> conCatFile2; printf "file '%s'\n" "black_1_s.mp4" >> conCatFile2; printf "file '%s'\n" "shoal.mp4" >> conCatFile2; >> conCatFile2; done
>>>>>>> 3922bff698e2364fccb39b83babf5d67f7445c8e
ffmpeg -f concat -i conCatFile2 -y final2.mp4 &

wait %1 %2

echo done
