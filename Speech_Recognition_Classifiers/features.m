function f=features(sig)
t=(0:length(sig)-1)/8000;
signal=filterdesign(sig);
f(1)=trapz(t,abs(signal));%Integrated EMG
f(2)=mean(abs(signal));%Mean absolute value
f(3)=rms(signal);
f(4)=var(signal);
f(5)=getwlfeat(signal);%waveform length;
f(6)=zerocross(signal);
f(7)=trapz(t,(abs(signal).^2)); %simple square integral;
f(8)=sum(diff(sign(diff(signal)))~=0);% Slope Sign Change 
f(9)=meanfreq(signal,8000);% mean frequency in terms of sampling rate
f(10)=medfreq(signal,8000);%median frequency
end