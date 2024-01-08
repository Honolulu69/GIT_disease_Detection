%CNN Program

DatasetPath='C:\Users\A\Doc\MATLAB\imresizegs';

images = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

numTrainFiles = 1500; 
[TrainImages,TestImages] = splitEachLabel(images, numTrainFiles, 'randomize');


layers = [
    imageInputLayer([300 300 1], 'Name', 'Input')
    
    convolution2dLayer(3,8, 'Padding', 'same','Name','Conv_1')
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','Relu_1')
    dropoutLayer(0.25, 'Name' ,'drop_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','Maxpool_1')
    
   
    convolution2dLayer(3,16,'Padding', 'same','Name','Conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','Relu_2') 
    dropoutLayer(0.25, 'Name' ,'drop_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','Maxpool_2')
    
   
    convolution2dLayer(3,32,'Padding', 'same','Name','Conv_3')
    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','Relu_3')
    dropoutLayer(0.25, 'Name' ,'drop_3')
    
    maxPooling2dLayer(2,'Stride',2,'Name','Maxpool_3')
    
    
    convolution2dLayer(3,64,'Padding', 'same','Name','Conv_4')
    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','Relu_4')
    dropoutLayer(0.25, 'Name' ,'drop_4')
    
    fullyConnectedLayer(2,'Name', 'FC')
    softmaxLayer('Name','softmax');
    classificationLayer('Name', 'Output Classfication');
    ];

%%lgraph = layerGraph(layers);
%%plot(lgraph);
%-------------------------------------------------------------
options = trainingOptions('adam','InitialLearnRate', 0.001, 'MaxEpochs',16,'Shuffle','every-epoch',...
          'ValidationData',TestImages, 'ValidationFrequency',20,'Verbose',false, 'plots','training-progress');
      
      net = trainNetwork(TrainImages,layers,options);
      
      YPred = classify(net,TestImages);
      Yvalidation = TestImages.Labels;
      
      accuracy = sum(YPred == Yvalidation)/numel(Yvalidation)
      plotconfusion(Yvalidation,YPred)
      
      
      