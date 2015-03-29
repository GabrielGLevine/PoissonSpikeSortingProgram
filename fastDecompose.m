function emitTrains=fastDecompose(spikes,neuronNum,params);
emission=spikes(1:400);
emission(1)=1;
emitISI=diff(find(emission));
emitTrains=zeros(neuronNum,400);
emitTimes=ones(neuronNum,1)*emitISI(1);
for j=1:length(emitISI)-1
    for i=1:neuronNum
        score(i)=poisscdf(emitTimes(i),params(i));
    end
    [C,I]=max(score);
    emitTrains(I,j)=emitTimes(I);
    for i=1:neuronNum
        if i==I
            emitTimes(i)=emitISI(j+1);
        else
            emitTimes(i)=emitTimes(i)+emitISI(j+1)
        end
    end
end
end

%%
cumsum(nonzeros(trains(2,:))')