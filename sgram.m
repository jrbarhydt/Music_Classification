function [spectrogram] = sgram(song_path,down_rate,time_steps, gabor_w)
%This function accepts an audio file path in string format and performs fft
%analysis to create a random 5-second spectrogram matrix representing both
%time and frequency of the signal, using gaussian gabor windowing.

% Arguments:
%-----------------------------
%  song_path: complete file path (if not in default MATLAB directory)
%  down_rate: rate of audio downsampling (i.e. only collect one sample per
%             'down_rate' datapoints
% time_steps: number of points in time to run gabor window fft algorithm
%    gabor_w: arbitrary scaling value, corresponding to "narrowness" of the
%             gabor window (gaussian) function

    % default arguments
    if nargin < 2
       down_rate = 10; 
    end
    if nargin < 3
       time_steps = 20; 
    end
    if nargin < 4
        gabor_w = 1000;
    end
% original read and data gather
[y,Fs] = audioread(song_path);
song_length = size(y,1)/Fs;
% random sample location
sample_start = 30 + rand()*(song_length-10-30);
fin = sample_start + 5 ; %could change smaple length here
L = fin - sample_start;
% determine sample location
samples = [round(sample_start*Fs),round(fin*Fs-1)];
clear y Fs
% only read sample clip
[y,Fs] = audioread(song_path,samples); 
%downsample rate
r=down_rate;
audio=downsample(y(:,1)',r);
n=round(L*Fs/r);
% build timebase
t2=linspace(0,L,n+1); t=t2(1:n);
% build frequency base (2*pi periodic) and center-shift (0 at origin)
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);
% set gabor window slide location
tslide=linspace(0,L,time_steps);
% initialize spectrogram
spectrogram=zeros(time_steps,length(t));
for j=1:time_steps
    gabor=exp(-gabor_w*(t-tslide(j)).^2); %gabor expresssion at slide loc
    Sg=gabor.*audio; windowed_fft=fft(Sg); %Gabor-Signal FFT
    spectrogram(j,:)=abs(fftshift(windowed_fft)); %spectrogram (let -k=k)
end