%Training and Validation using AlexNet

DatasetPath ='C:\Users\A\D\MATLAB\imresizegs2';
%Reading Images from Database Folder

images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numTrainFiles = 1200;
[TrainImages,TestImages] = splitEachLabel(images,numTrainFiles,'randomize');

net = alexnet;

layersTransfer = net.Layers(1:end-3);
numClasses = 2;

analyzeNetwork(net)
plot(net)

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MiniBatchsize',20,...
    'MaxEpochs', 8,...
     'InitialLearnRate',1e-4,...
     'Shuffle','every-epoch',...
     'ValidationData',TestImages,...
     'ValidationFrequency',10,...
     'Verbose', false,...
     'Plots','training-progress');
 
 netTransfer = trainNetwork(TrainImages,layers,options);
 
 YPred = classify(netTransfer,TestImages);
 Yvalidation = TestImages.Labels;
 
 accuracy = sum(YPred == Yvalidation)/numel(Yvalidation)
 plotconfusion(Yvalidation,YPred)
 


