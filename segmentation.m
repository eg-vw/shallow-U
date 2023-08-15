%training code

%for data augmentation
targetSize = [256 256];

%to construct the pixelLabelDatastore ignoring other class values
classes = ["flower","background"];
labelIDs = [1,3];

%test and train split prior to reading in
images = fullfile('C:\);
labels = fullfile('C:\');
tImages = fullfile('C:\');
tLabels = fullfile('C:\');

%converting to datastores
testImages = imageDatastore(tImages);
testLabels = pixelLabelDatastore(tLabels,classes,labelIDs);
testSet = combine(testImages,testLabels);

%replicating training images x3, converting, combining
numObservations = 3;
trainImages = repelem({images},numObservations,1);
trainLabels = repelem({labels},numObservations,1);
imds = imageDatastore(images);
pxds = pixelLabelDatastore(labels,classes,labelIDs);
segDataCombined = combine(imds,pxds);

%applying augmentation functions
augmentedTrainingData = transform(segDataCombined,@jitterImageColorAndWarp);
preprocessedTrainingData = transform(augmentedTrainingData,...
    @(data)centerCropImageAndLabel(data,targetSize));

%displaying pixel label counts
tbl = countEachLabel(pxds)

%calculating median for network weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

%defining layergraph for deepNetworkDesigner
unet_6 = layerGraph();

%network layers
layers = [

   %downsampling 1
    imageInputLayer([256 256 3],"Name","imageinput")
    convolution2dLayer([3 3],16,"Name","conv_1",'Padding','same')
    reluLayer("Name","relu_1")
    convolution2dLayer([3 3],16,"Name","conv_3",'Padding','same')
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding",'same',"Stride",[2 2])


    %downsampling 2
    convolution2dLayer([3 3],32,"Name","conv_7_1",'Padding','same')
    reluLayer("Name","relu_6_2")
    convolution2dLayer([3 3],32,"Name","conv_7_3",'Padding','same')
    reluLayer("Name","relu_6_1")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding",'same',"Stride",[2 2])


    %bridging layer
    convolution2dLayer([3 3],64,"Name","conv_10_1",'Padding','same')
    reluLayer("Name","relu_8_2")
    convolution2dLayer([3 3],64,"Name","conv_10_2",'Padding','same')
    reluLayer("Name","relu_8_3")
    transposedConv2dLayer([2 2],32,"Name","transposed-conv_1",'Stride',[2 2])


    %upsampling 1
    concatenationLayer(3,2,"Name","concat_104")
    convolution2dLayer([3 3],32,"Name",'conv_7_5','Padding','same')
    reluLayer("Name","relu_14_2")
    convolution2dLayer([3 3],32,"Name",'conv_7_2','Padding','same')
    reluLayer("Name","relu_14_1")
    transposedConv2dLayer([2 2],16,"Name","transposed-conv_3",'Stride',[2 2])


    %upsampling 2
    concatenationLayer(3,2,"Name","concat_4")
    convolution2dLayer([3 3],16,"Name",'conv_18','Padding','same')
    reluLayer("Name","relu_17")
    convolution2dLayer([3 3],16,"Name","conv_19",'Padding','same')
    reluLayer("Name","relu_18")

    %classification
    convolution2dLayer([1 1],2,"Name","conv_4",'Padding','same')
    softmaxLayer("Name","softmax_1")
    pixelClassificationLayer("Name","final_output")
    ];

%adding layers to layergraph
unet_6 = addLayers(unet_6,layers);

%cropping layers have ref connections, need to be defined afterwards
tempLayers = crop2dLayer("centercrop","Name","crop2d");
unet_6 = addLayers(unet_6,tempLayers);
tempLayers = crop2dLayer("centercrop","Name","crop2d_1");
unet_6 = addLayers(unet_6,tempLayers);

%connect non-sequential layers
unet_6 = connectLayers(unet_6,"relu_3","crop2d_1/in");
unet_6 = connectLayers(unet_6,"relu_6_1","crop2d/in");
unet_6 = connectLayers(unet_6,"transposed-conv_1","crop2d/ref");
unet_6 = connectLayers(unet_6,"transposed-conv_3","crop2d_1/ref");
unet_6 = connectLayers(unet_6,"crop2d","concat_104/in2");
unet_6 = connectLayers(unet_6,"crop2d_1","concat_4/in2");

%update classification layer with tbl weights
classLayer = pixelClassificationLayer('Name','labels','Classes',classes,'ClassWeights',classWeights);
unet_6 = replaceLayer(unet_6,'final_output',classLayer);
unet_6.Layers(end);

%plot(unet_6);

%training options
opts = trainingOptions('adam', ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateDropPeriod',2, ...
    'MaxEpochs',32, ...
    'Verbose',1, ...
    'VerboseFrequency',20, ...
    'MiniBatchSize',16, ...
    'Plots','training-progress');   

train the network 
f_net = trainNetwork(preprocessedTrainingData,unet_6,opts)

save
effNet = f_net;
save effNet;

%using semanticseg to generate results
results = semanticseg(testSet,effNet,'WriteLocation',tempdir);
[C,scores] = semanticseg(I,effNet);

%metrics inc. confusion matrix, miou, accuracy scores
metrics = evaluateSemanticSegmentation(results,testLabels);

%helper functions for data augmentation
function out = centerCropImageAndLabel(data,targetSize)
win = centerCropWindow2d(size(data{1}),targetSize);
out{1} = imcrop(data{1},win);
out{2} = imcrop(data{2},win);
end

function out = jitterImageColorAndWarp(data) 
I = data{1};
C = data{2};

I = jitterColorHSV(I,"Brightness",0.3,"Contrast",0.4,"Saturation",0.2);

tform = randomAffine2d("Scale",[0.8 1.5],"XReflection",true,'Rotation',[-30 30]);
rout = affineOutputView(size(I),tform);

augmentedImage = imwarp(I,tform,"OutputView",rout);
augmentedLabel = imwarp(C,tform,"OutputView",rout);

out = {augmentedImage,augmentedLabel};
end
