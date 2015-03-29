% Gabriel Levine
%Improved Spike-Sorting By Modeling Firing Statistics and Burst-Dependent Spike Amplitude Attenuation: A Markov Chain Monte Carlo Approach
%% Plot Distributions
plot(poisspdf(1:100,10))
hold on
plot(poisspdf(1:100,20),'r')
plot(poisspdf(1:100,40),'g')
legend('Neuron 1','Neuron 2','Neuron 3')
xlabel('ISI')
ylabel('Probability')
%% Generate Spike Trains (Uniform Amplitude)
clear all; close all;
%Generate Individual Spike Trains
neuronNum=2;
spikesPer=50;
%neurons.isi=randi([10 30],1,neuronNum);
neurons.isi=[10 20 40];
%neurons.Amplitudes=randi([1 1],1,neuronNum);
neurons.Amplitudes=[1 1 1];
scrsz = get(0,'ScreenSize');
figure('Position',scrsz);
for i=1:neuronNum
    neurons.spikes(i,:)=cumsum(poissrnd(neurons.isi(i),1,spikesPer));
    Amp=ones(1,spikesPer)*neurons.Amplitudes(i);
    subplot(neuronNum,1,i)
    stem(neurons.spikes(i,:),Amp,'Marker','none')
    ylim([0 max(neurons.Amplitudes)])
    xlim([0 max(max(neurons.spikes(1,:)))])
    title(['Neuron: ',num2str(i),'. Amplitude: ',num2str(neurons.Amplitudes(i)),'. Mean ISI: ',num2str(neurons.isi(i))])
    xlabel('Time (ms)')
    ylabel('Amplitude')
    box off;
end
%Combine Spike Trains
spikes=zeros(1,max(max(neurons.spikes)));
for i=1:neuronNum
    spikes(neurons.spikes(i,:))=spikes(neurons.spikes(i,:))+neurons.Amplitudes(i);
end
figure('Position',scrsz);
stem(spikes(1:500),'Marker','none')
title('Aggregate Spike Train')
xlabel('Time (ms)')
ylabel('Amplitude')
box off;
%% Fourier Spectrum
data=spikes(1:5000)
Npoints=length(data); % number of sampled points (best if this is a power of 2)
fs=1000;  % sampling frequency
nyquist = fs/2 ;    % nyquist frequency
range=(Npoints/2)+1; % set the range for the spectral plot to be 1/2 of the data size plus 1
t=1/fs*(1:Npoints); % set time vector of length Npoints
f=fs*(0:range-1)/Npoints; % set frequency vector
y=data-mean(data);
Y=fft(y, Npoints); % compute FFT
Pyy = 2*Y.*conj(Y)/Npoints; % compute power spectrum
plot(f, Pyy(1:range)); % plot the power spectrum vs frequency
xlabel('frequency (Hz)');ylabel('power (mV^2)');title('Power Spectra (2 Cells)');

%% Generate Candidate Trains
initTrains=initialSpikes(spikes(1:neurons.isi(1)*spikesPer),neuronNum);
figure('Position',scrsz);
for i=1:neuronNum
    subplot(neuronNum,1,i)
    stem(initTrains(i,:),'Marker','none')
    ylim([0 max(spikes)])
    xlabel('Time (ms)')
    ylabel('Amplitude')
    box off;
end
inds=find(spikes(1:neurons.isi(1)*spikesPer))
%candTrains=candidateSpikes(initTrains,inds,neuronNum);

%% Sort Spikes (w/ objfunc (naieve)
maxStep=50000;
improveStep=50000;
tol=10;
steps=0;
improve=0;
trains=initTrains;
Score=objfunc(trains,spikesPer,neuronNum,neurons);
t=linspace(0,12,maxStep);
SwapCount=[linspace(log(1.25),log(sum(sum(trains))),maxStep-maxStep/100),ones(1,maxStep/100)];
clear E
while (steps<maxStep)&(improve<improveStep)&(Score>tol)
    steps=steps+1;
    swapCount=(spikesPer*neuronNum)/exp(SwapCount(steps));
    T=2^(-t(steps));
    candTrains=candidateSpikes(trains,inds,neuronNum,swapCount);
    candScore=objfunc(candTrains,spikesPer,neuronNum,neurons);
    [Score,isiScore(steps),ampScore(steps)]=objfunc(trains,spikesPer,neuronNum,neurons);
    if Score<=candScore
        %        P=exp((Score-candScore)/T);
        improve=improve+1;
        %         if rand<P
        %             trains=candTrains;
        %             E(steps)=candScore;
        %         else
        E(steps)=Score;
        %         end
    else
        improve=0;
        trains=candTrains;
        E(steps)=candScore;
    end
    %    Pvec(steps)=P;
    candVec(steps)=candScore;
end
plot(E)
%% Sort Spikes (w/ likliehood func)
maxStep=100000;
improveStep=10000;
tol=10;
steps=0;
improve=0;
trains=initTrains;
bestTrains=trains;
Score=likelihood(trains,neurons);
t=linspace(1.4,0.001,maxStep);
SwapCount=[linspace(log(1.25),log(sum(sum(trains))),maxStep-maxStep/100),ones(1,maxStep/100)*log(sum(sum(trains)))];
clear E
E(1)=likelihood(trains,neurons);
figure;
while (steps<maxStep)&(improve<improveStep)&(Score>10)
    steps=steps+1;
    T=t(steps);
    swapCount=(sum(sum(trains)))/exp(SwapCount(steps));
    candTrains=candidateSpikes(trains,inds,neuronNum,swapCount);
    candScore=likelihood(candTrains,neurons);
    Score=likelihood(trains,neurons);
    if Score<=min(E)
        bestTrains=trains;
        improve=0;
        for i=1:neuronNum
            subplot(neuronNum+1,1,i)
            stem(bestTrains(i,:),'Marker','none')
        end
    else
        improve=improve+1;
    end
    if Score<=candScore
        P=exp((Score-candScore)/T);
        Pvec(steps)=P;
        if rand<P
            trains=candTrains;
            E(steps)=candScore;
        else
            E(steps)=Score;
        end
    elseif Score>candScore
        trains=candTrains;
        E(steps)=candScore;
    end
    candVec(steps)=candScore;
    
    if mod(steps,100)==0
        subplot(neuronNum+1,1,neuronNum+1)
        plot(E)
        drawnow
    end
end
plot(E)
%% Plot Solution vs Correct and Initial
clear E bestTrains neurons
load UniformAmp_3Neurons.mat
figure;
k=1;
for i=1:neuronNum
    subplot(neuronNum,3,k)
    k=k+1;
    stem(neurons.spikes(i,:),ones(1,spikesPer),'Marker','none')
    title('Correct')
    ylim([0 3])
    xlim([0 500])
    subplot(neuronNum,3,k)
    k=k+1;
    
    stem(bestTrains(i,:),'Marker','none')
    title('Final Solution')
    ylim([0 3])
    subplot(neuronNum,3,k)
    k=k+1;
    stem(initTrains(i,:),'Marker','none')
    title('Initial Solution')
    ylim([0 3])
end

figure;
for i=1:3
    subplot(2,3,i)
hist(diff(neurons.spikes(i,find(neurons.spikes(i,:)<500))),0:4:60)
title('Corrrect')
xlabel('ISI')
end
for i=1:3
    subplot(2,3,i+3)
hist(diff(find(bestTrains(i,:))),0:4:60)
title('Solution')
xlabel('ISI')
end

%% Generate Spike Trains (Varied Amplitudes)
clear all; close all;
%Generate Individual Spike Trains
neuronNum=2;
spikesPer=40;
%neurons.isi=randi([10 30],1,neuronNum);
neurons.isi=[10 40 40];
%neurons.Amplitudes=randi([1 1],1,neuronNum);
neurons.Amplitudes=[1 2 3];
scrsz = get(0,'ScreenSize');
figure('Position',scrsz);
for i=1:neuronNum
    neurons.spikes(i,:)=cumsum(poissrnd(neurons.isi(i),1,spikesPer));
    Amp=ones(1,spikesPer)*neurons.Amplitudes(i);
    subplot(neuronNum,1,i)
    stem(neurons.spikes(i,:),Amp,'Marker','none')
    ylim([0 max(neurons.Amplitudes)])
    xlim([0 max(max(neurons.spikes))])
    title(['Neuron: ',num2str(i),'. Amplitude: ',num2str(neurons.Amplitudes(i)),'. Mean ISI: ',num2str(neurons.isi(i))])
    xlabel('Time (ms)')
    ylabel('Amplitude')
    box off;
end
%Combine Spike Trains
spikes=zeros(1,max(max(neurons.spikes)));
for i=1:neuronNum
    spikes(neurons.spikes(i,:))=spikes(neurons.spikes(i,:))+neurons.Amplitudes(i);
end
figure('Position',scrsz);
stem(spikes,'Marker','none')
title('Aggregate Spike Train')
xlabel('Time (ms)')
ylabel('Amplitude')
box off;
%% Generate Candidate Trains
initTrains=initialSpikes(spikes(1:neurons.isi(1)*spikesPer),neuronNum);
figure('Position',scrsz);
for i=1:neuronNum
    subplot(neuronNum,1,i)
    stem(initTrains(i,:),'Marker','none')
    ylim([0 max(spikes)])
    xlabel('Time (ms)')
    ylabel('Amplitude')
    box off;
end
inds=find(spikes(1:neurons.isi(1)*spikesPer))
%candTrains=candidateSpikes(initTrains,inds,neuronNum);

%% Sort Spikes (w/ likliehoodAmp func)
maxStep=20000;
improveStep=10000;
tol=10;
steps=0;
improve=0;
trains=initTrains;
bestTrains=trains;
Score=likelihoodAmp(trains,neurons);
t=linspace(1.4,0.001,maxStep);
SwapCount=[linspace(log(1.25),log(sum(sum(trains))),maxStep-maxStep/100),ones(1,maxStep/100)*log(sum(sum(trains)))];
clear E ISIScore AmpScore
[E(1), ISIScore(1), AmpScore(1)]=likelihoodAmp(trains,neurons);

figure('Position',scrsz/2);
while (steps<maxStep)&(improve<improveStep)&(Score>10)
    steps=steps+1;
    T=t(steps);
    swapCount=(sum(sum(trains)))/exp(SwapCount(steps));
    %swapCount=2;
    candTrains=candidateSpikes(trains,inds,neuronNum,swapCount);
    candScore=likelihoodAmp(candTrains,neurons);
    [Score,ISIScore(steps),AmpScore(steps)]=likelihoodAmp(trains,neurons);
    if Score<=min(E)
        bestTrains=trains;
        improve=0;
        for i=1:neuronNum
            subplot(neuronNum+1,1,i)
            stem(bestTrains(i,:),'Marker','none')
        end
    else
        improve=improve+1;
    end
    if Score<=candScore
        P=exp((Score-candScore)/T);
        Pvec(steps)=P;
        if rand<P
            trains=candTrains;
            E(steps)=candScore;
        else
            E(steps)=Score;
        end
    elseif Score>candScore
        trains=candTrains;
        E(steps)=candScore;
    end
    candVec(steps)=candScore;
    
    if mod(steps,100)==0
        subplot(neuronNum+1,1,neuronNum+1)
        plot(E)
        hold on;
        plot(AmpScore,'r')
        plot(ISIScore,'k')
        box off;
        legend('Total Score','Amplitude Score','ISI Score','Location','southoutside','Orientation','horizontal')
        hold off;
        drawnow
    end
end
plot(E)
%% Plot Solution vs Correct and Initial
k=1;
for i=1:neuronNum
    subplot(neuronNum,3,k)
    k=k+1;
    stem(neurons.spikes(i,:),ones(1,spikesPer),'Marker','none')
    title('Correct')
    ylim([0 3])
    xlim([0 500])
    subplot(neuronNum,3,k)
    k=k+1;
    
    stem(bestTrains(i,:),'Marker','none')
    title('Final Solution')
    ylim([0 3])
    subplot(neuronNum,3,k)
    k=k+1;
    stem(initTrains(i,:),'Marker','none')
    title('Initial Solution')
    ylim([0 3])
end
%% Estimate Model Parameters from Aggregate Spike Train
%% Generate Spike Trains (Uniform Amplitude)
clear all; close all;
%Generate Individual Spike Trains
neuronNum=3;
spikesPer=4096;
%neurons.isi=randi([10 30],1,neuronNum);
neurons.isi=[10 20 40];
%neurons.Amplitudes=randi([1 1],1,neuronNum);
neurons.Amplitudes=[1 1 1 1 1];
scrsz = get(0,'ScreenSize');
%figure('Position',scrsz);
for i=1:neuronNum
    neurons.spikes(i,:)=cumsum(poissrnd(neurons.isi(i),1,spikesPer));
    Amp=ones(1,spikesPer)*neurons.Amplitudes(i);
% %    subplot(neuronNum,1,i)
% %    stem(neurons.spikes(i,:),Amp,'Marker','none')
%     ylim([0 max(neurons.Amplitudes)])
%     xlim([0 max(max(neurons.spikes))])
%     title(['Neuron: ',num2str(i),'. Amplitude: ',num2str(neurons.Amplitudes(i)),'. Mean ISI: ',num2str(neurons.isi(i))])
%     xlabel('Time (ms)')
%     ylabel('Amplitude')
%     box off;
end
%Combine Spike Trains
spikes=zeros(1,max(max(neurons.spikes)));
for i=1:neuronNum
    spikes(neurons.spikes(i,:))=spikes(neurons.spikes(i,:))+neurons.Amplitudes(i);
end
figure('Position',scrsz);
stem(spikes(1:400),'Marker','none')
title('Aggregate Spike Train')
xlabel('Time (ms)')
ylabel('Amplitude')
box off;

%% Fourier Analysis
data=spikes
Npoints=length(data); % number of sampled points (best if this is a power of 2)
fs=1000;  % sampling frequency
nyquist = fs/2 ;    % nyquist frequency
range=(Npoints/2)+1; % set the range for the spectral plot to be 1/2 of the data size plus 1
t=1/fs*(1:Npoints); % set time vector of length Npoints
f=fs*(0:range-1)/Npoints; % set frequency vector
y=data-mean(data);
Y=fft(y, Npoints); % compute FFT
Pyy = 2*Y.*conj(Y)/Npoints; % compute power spectrum
plot(f, Pyy(1:range)); % plot the power spectrum vs frequency
xlabel('frequency (Hz)');ylabel('power (mV^2)');title('Power Spectra (2 Cells)');
%% Bandpassed Fourier
f=fs*(0:range-100)/Npoints; % set frequency vector
y=data-mean(data);
Y=fft(y, Npoints); % compute FFT
Pyy = 2*Y.*conj(Y)/Npoints; % compute power spectrum
plot(f, Pyy(100:range)); % plot the power spectrum vs frequency
xlabel('frequency (Hz)');ylabel('power (mV^2)');title('Power Spectra (2 Cells)');
%% Peak Isolation
cutoff=Npoints/(1000/110);
[pks,locs] = findpeaks(Pyy(100:cutoff),'MINPEAKDISTANCE',800,'MINPEAKHEIGHT',0.6)
plot(f(100:cutoff),(Pyy(100:cutoff)))
hold on;
scatter(f(locs+100),pks,'^')
candISI=1000./f(locs+100);
candISI=unique(round(candISI));
%% Generate Candidate Trains
initTrains=initialSpikes(spikes(1:400),neuronNum);
figure('Position',scrsz);
for i=1:neuronNum
    subplot(neuronNum,1,i)
    stem(initTrains(i,:),'Marker','none')
    ylim([0 max(spikes)])
    xlabel('Time (ms)')
    ylabel('Amplitude')
    box off;
end
inds=find(spikes(1:400))
%candTrains=candidateSpikes(initTrains,inds,neuronNum);
%% Sort Spikes (w/ likliehood func) to find Params
candComb = combnk(candISI,neuronNum);
for isi=1:length(candComb)
for j=1:neuronNum
    params{isi}(j)=candComb(isi,j);
end
E{isi}=NaN(1,10000);
end
parfor isi=1:length(candComb)
maxStep=10000;
improveStep=1000;
tol=10;
steps=0;
improve=0;
trains=initTrains;
Score=likelihoodParam(trains,params{isi});
t=linspace(1.4,0.001,maxStep);
SwapCount=[linspace(log(1.25),log(sum(sum(trains))),maxStep-maxStep/100),ones(1,maxStep/100)*log(sum(sum(trains)))];
E{isi}(1)=likelihoodParam(trains,params{isi});
while (steps<maxStep)&(improve<improveStep)&(Score>tol)
    steps=steps+1;
    T=t(steps);
    swapCount=(sum(sum(trains)))/exp(SwapCount(steps));
    candTrains=candidateSpikes(trains,inds,neuronNum,swapCount);
    candScore=likelihoodParam(candTrains,params{isi});
    Score=likelihoodParam(trains,params{isi});
    if Score<=min(E{isi})
        bestTrains=trains;
        improve=0;
    else
        improve=improve+1;
    end
    if Score<=candScore
        P=exp((Score-candScore)/T);
        if rand<P
            trains=candTrains;
            E{isi}(steps)=candScore;
        else
            E{isi}(steps)=Score;
        end
    elseif Score>candScore
        trains=candTrains;
        E{isi}(steps)=candScore;
    end
end
end
%% Find Candidate Params
for i=1:length(E)
paramScore(i)=min(E{i});
end
scores=[candComb,paramScore'];
scores=sortrows(scores,neuronNum+1);
finalists=scores(1:25,1:3);
%% Test Finalist Parameters
candComb = finalists;
clear E params
for isi=1:length(candComb)
for j=1:neuronNum
    params{isi}(j)=candComb(isi,j);
end
E{isi}=NaN(1,10000);
end
parfor isi=1:length(candComb)
maxStep=30000;
improveStep=15000;
tol=10;
steps=0;
improve=0;
trains=initTrains;
Score=likelihoodParam(trains,params{isi});
t=linspace(1.4,0.001,maxStep);
SwapCount=[linspace(log(1.25),log(sum(sum(trains))),maxStep-maxStep/100),ones(1,maxStep/100)*log(sum(sum(trains)))];
E{isi}(1)=likelihoodParam(trains,params{isi});
while (steps<maxStep)&(improve<improveStep)&(Score>tol)
    steps=steps+1;
    T=t(steps);
    swapCount=(sum(sum(trains)))/exp(SwapCount(steps));
    candTrains=candidateSpikes(trains,inds,neuronNum,swapCount);
    candScore=likelihoodParam(candTrains,params{isi});
    Score=likelihoodParam(trains,params{isi});
    if Score<=min(E{isi})
        bestTrains=trains;
        improve=0;
    else
        improve=improve+1;
    end
    if Score<=candScore
        P=exp((Score-candScore)/T);
        if rand<P
            trains=candTrains;
            E{isi}(steps)=candScore;
        else
            E{isi}(steps)=Score;
        end
    elseif Score>candScore
        trains=candTrains;
        E{isi}(steps)=candScore;
    end
end
end

for i=1:length(E)
paramScore(i)=min(E{i});
end
scores=[finalists,paramScore'];
scores=sortrows(scores,neuronNum+1);
%% Display and Run with New Parameters
load ParameterTest
figure;
uitable('Data', scores, 'ColumnName', {'Mean1', 'Mean 2', 'Mean 3','Score'},'Position',[100 200 400 150]);
