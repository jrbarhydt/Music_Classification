function [y,Fs] = musicclip(song_path,clip_length,r)
%This function accepts an audio file path in string format and creates a
%randomly located clip from the audio file, whose length and smaple rate
%can be chosen.

% Arguments:
%-----------------------------
%   song_path: complete file path (if not in default MATLAB directory)
%           r: rate of audio downsampling (i.e. only collect one sample per
%              'down_rate' datapoints
% clip_length: length, in seconds, of the audio clip to be returned
%
% Returns:
%-----------------------------
%           y: down-sampled, audio clip of desired length
%          Fs: clip frame-rate

    % default arguments
    if nargin < 2
       clip_length = 5; 
    end
    if nargin < 3
       r = 1; 
    end

[y,Fs] = audioread(song_path);
song_length = size(y,1)/Fs;
sample_start = 30 + rand()*(song_length-10-30);
fin = sample_start + clip_length ;
samples = [round(sample_start*Fs),round(fin*Fs-1)];
clear y Fs
[y,Fs] = audioread(song_path,samples);
y = (y(:,1)+y(:,2))/2;
y=downsample(y(:,1)',r);
end

