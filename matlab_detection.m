clc;clear; close all;


fish_video = VideoReader('dataset\video_19\video_19.avi');
ROI_text_file = readtable('dataset\video_19\video_19_ROI.txt');
posy = str2double(ROI_text_file{1,1}{1, 1}(end-2:end));


forground_detector = vision.ForegroundDetector('NumTrainingFrames', 8,...
    'MinimumBackgroundRatio', 0.75, 'NumGaussians', 5, 'LearningRate', 0.001);
blober = vision.BlobAnalysis('AreaOutputPort',false,...
    'MinimumBlobArea', 30);

video_player = vision.VideoPlayer();
video_player.Position = [348.3333 283.3333 640 480];
mask_player = vision.VideoPlayer();
mask_player.Position = [948.3333 283.3333 640 480];

while hasFrame(fish_video)

    frame = readFrame(fish_video);
    mask = forground_detector(frame);
    [centroids, bboxes] = blober(mask);
    
    frame = insertShape(frame, "Rectangle", bboxes, "LineWidth",3);
    frame = insertMarker(frame,centroids,"plus");
    frame(posy, :, 1) = 255;
    video_player(frame);
    mask_player(mask);
    pause(0.1);
end


