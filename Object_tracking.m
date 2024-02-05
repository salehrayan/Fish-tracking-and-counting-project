clc;clear; close all;

%%%% LOAD DATASET STUFF
fish_video = VideoReader('dataset\video_19\video_19.avi');
ROI_text_file = readtable('dataset\video_19\video_19_ROI.txt');
labels = fileread('dataset\video_19\video_19_manual.txt');
posy = str2double(ROI_text_file{1,1}{1, 1}(end-2:end));
num_frames = fish_video.NumFrames;

s = jsondecode(labels);
label_frames = cell2mat(cellfun(@(x) str2double(regexp(x, '\d+', 'match')), fieldnames(s),...
    'UniformOutput', false)) +1;
label_vals = struct2array(s)';
labels = zeros([num_frames 1]);
labels(label_frames) = label_vals;
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% MoG segmenter and BLOB DETECTION
forground_detector = vision.ForegroundDetector('NumTrainingFrames', 5,...
    'MinimumBackgroundRatio', 0.75, 'NumGaussians', 5, 'LearningRate', 0.001);
blober = vision.BlobAnalysis('AreaOutputPort',false,...
    'MinimumBlobArea', 30);
%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% VIDEO PLAYER OBJECTS
video_player = vision.VideoPlayer();
video_player.Position = [348.3333 283.3333 640 480];
mask_player = vision.VideoPlayer();
mask_player.Position = [948.3333 283.3333 640 480];
vid_write = VideoWriter('Matlab_demo.mp4');
open(vid_write)
%%%%%%%%%%%%%%%%%%%%%%%

fishes_detected = struct('ID', [], 'frame_number', [], 'number_of_fish', [], 'area', []);
detection_history = cell(1,num_frames);
banned_IDs = struct('ID', []);

tracker = trackerGNN("MaxNumSensors", 1, 'MaxNumTracks', 60);
tracker.FilterInitializationFcn = @initcvkf;
tracker.ConfirmationThreshold = [3 3];
tracker.DeletionThreshold = [5 5];

% str = strel("disk", 5);
h = fspecial('log', [50 50], 7);

for frame_count=1:num_frames
    
    frame = readFrame(fish_video);
    mask = forground_detector(frame);
    [centroids, bboxes] = blober(mask);
%     bboxes = [];
%     centroids = [];
    num_measurements_in_frame = size(centroids,1);
    detections_in_frame = cell(num_measurements_in_frame,1);

    for detection_count=1:num_measurements_in_frame
        detections_in_frame{detection_count} = objectDetection(frame_count,...
            centroids(detection_count, :), "MeasurementNoise", diag([100 100]), ...
            ObjectAttributes = struct('BoundingBox', bboxes(detection_count, :)));
    end
    detection_history{frame_count} = detections_in_frame;

    if isLocked(tracker) || ~isempty(detection_history{frame_count})
        tracks = tracker(detection_history{frame_count}, frame_count);
    else
        tracks = objectTrack.empty;
    end

    for i=1:numel(tracks)
        if tracks(i).State(3) >= posy && ~any(tracks(i).TrackID == [fishes_detected.ID]) &&...
                ~any(tracks(i).TrackID == [banned_IDs.ID])
            if tracks(i).Age<=2
                banned_IDs(end+1).ID = tracks(i).TrackID;
                continue
            end
            fishes_detected(end+1).ID = tracks(i).TrackID;
            fishes_detected(end).frame_number = frame_count;
            area = tracks(i).ObjectAttributes.BoundingBox(3).*...
                    tracks(i).ObjectAttributes.BoundingBox(4);
            fishes_detected(end).area = area;
            fishes_detected(end).number_of_fish = 1;

            
        end
    end

    frame = insertTracksToFrame(frame, tracks);
    frame = insertText(frame, [0, 0], "Computer count: "+ num2str(sum([fishes_detected.number_of_fish]))+...
        ", Real count: "+ num2str(sum(labels(1:frame_count))),...
        BoxColor= "black", TextColor="yellow",BoxOpacity=1);

%     frame = insertShape(frame, "Rectangle", bboxes, "LineWidth",3);
%     frame = insertMarker(frame,centroids,"plus");
    frame(posy, :, 1) = 255;
%     video_player(frame);
%     mask_player(mask);
    imshow(frame, [])
%     vid_write(frame)
    writeVideo(vid_write, frame);
    
    pause(0.1);
end




function frame = insertTracksToFrame(frame, tracks)
numTracks = numel(tracks);
boxes = zeros(numTracks, 4);
ids = zeros(numTracks, 1, "int32");
predictedTrackInds = zeros(numTracks, 1);
for tr = 1:numTracks
    % Get bounding boxes.
    boxes(tr, :) = tracks(tr).ObjectAttributes.BoundingBox;
    boxes(tr, 1:2) = (tracks(tr).State(1:2:3))'-boxes(tr,3:4)/2;

    % Get IDs.
    ids(tr) = tracks(tr).TrackID;

    if tracks(tr).IsCoasted
        predictedTrackInds(tr) = tr;
    end
end

predictedTrackInds = predictedTrackInds(predictedTrackInds > 0);

% Create labels for objects that display the predicted rather
% than the actual location.
labels = cellstr(int2str(ids));

isPredicted = cell(size(labels));
isPredicted(predictedTrackInds) = {' predicted'};
labels = strcat(labels, isPredicted);

% Draw the objects on the frame.
frame = insertObjectAnnotation(frame, "rectangle", boxes, labels, ...
    TextBoxOpacity = 0.5);
end

