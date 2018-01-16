## Sign Language Recognition using WiFi and Convolutional Neural Networks

This website contains the datasets of Channel State Information (CSI) traces for the following paper.

[Yongsen Ma](http://www.cs.wm.edu/~yma/), [Gang Zhou](http://gzhou.blogs.wm.edu), Shuangquan Wang, Hongyang Zhao, and Woosub Jung. **SignFi: Sign Language Recognition using WiFi. ACM IMWUT (UbiComp '18)**.

### Files
This repository contains the following files.
| Files | Description | Size |
| ----- | ----------- | ---- |
| videos/ | This folder contains the videos for 276 basic sign words. These videos are used for the participant to learn and perform sign gestures during data collection. | 105.9 MB |
|dataset_home_276.mat | Downlink and uplink CSI traces for 276 sign words in the home environment. There are 2,760 instances of 276 sign gestures performed by one user. | 1.37 GB |
|dataset_lab_276.mat | Downlink and uplink CSI traces for 276 sign words in the lab environment. There are 5,520 instances of 276 sign gestures performed by one user.| 2.77 GB |
|dataset_lab_150.mat | Downlink CSI traces for 150 sign words in the lab environment. There are 7,500 instances of 150 sign gestures performed by five users. | 3.56 GB |
| signfi_cnn_example.m | MATLAB source code for training and testing using the dataset. | 3 KB |
| training_screen_shot.png | A screen shot of the training process | 541 KB |
| sign_labels.csv | Labels for 276 sign words used in the measurement. | 2 KB |
| README.md | Readme | 490 bytes |


### Data Collection Summary
The following table shows the data collection summary. Please check the paper for more details about experiment setup, measurement procedure, WiFi settings, etc.
| Environment | User | Age | Weight/Height | Data Collection Date (Number of Signs × Number of Repetitions) | Gesture Duration | Number of Instances |
| ----------- | ---- | --- | ------------- | -------------------------------------------------------------- | ---------------- | ------------------- |
| Lab  | User 1 | 39 | 90kg/170cm | Oct. 18, 2017 (25×10); Nov. 2, 2017 (125×10) | 1s-2.5s | 1,500 |
| Lab  | User 2 | 28 | 61kg/174cm | Oct. 18, 2017 (25×10); Oct. 30, 2017 (125×10) | 0.5s-1.5s | 1,500 |
| Lab  | User 3 | 31 | 55kg/168cm | Oct. 21, 2017 (25×10); Nov. 6, 2017 (125×10) | 0.5s-1.5s | 1,500 |
| Lab  | User 4 | 26 | 65kg/180cm | Oct. 23, 2017 (25×10); Oct. 31, 2017 (125×10) | 1s-2.5s | 1,500 |
| Lab  | User 5 | 29 | 68kg/171cm | Jul. 18, 2017 (166×20); Jul. 19, 2017 (110×20) | 0.5s-1.5s | 5,520 |
| Home | User 5 | 29 | 68kg/171cm | Jun. 8, 2017 (32×10); Jun. 25, 2017 (68×10); Jul. 4, 2017 (100×10); Jul. 11, 2017 (25×10); Jul. 12, 2017 (51×10) | 0.5s-1.5s | 2,760 |


### An Example
The following shows an example of running `[net_info, perf] = signfi_cnn_example(csi,label)`.
1. Load the dataset and prepare for training.
```Matlab
	load('dataset_lab_276.mat');
    tic;
	% prepare for training data
    csi_abs = abs(csi);
    csi_ang = angle(csi);
	csi_tensor = [csi_abs,csi_ang];
    word = categorical(label);
    t0 = toc; % pre-processing time
```
`tic` and `toc` are used to measure time consumption. The size of `csid_lab` is `size(csid_lab)=(200,30,3,5520)`. `csi_tensor` and `word` are the CSI input and labels for training.

2. Some parameter settings.
```Matlab
	[M,N,S,T] = size(csi_tensor);
    Nw = 276; % number of classes
    rng(42); % For reproducibility
    n_epoch = 10;
    learn_rate = 0.01;
    l2_factor = 0.01;
```

3. Divide the dataset into training and testing subsets.
```Matlab
K = 5;
    cv = cvpartition(T,'kfold',K); % 20% for testing
    k = 1; % for k=1:K
    trainIdx = find(training(cv,k));
    testIdx = find(test(cv,k));
    trainCsi = csi_tensor(:,:,:,trainIdx);
    trainWord = word(trainIdx,1);
    testCsi = csi_tensor(:,:,:,testIdx);
    testWord = word(testIdx,1);
    valData = {testCsi,testWord};
```

4. Set neural network layers and options.
```Matlab
	% Convolutional Neural Network settings
    layers = [imageInputLayer([M N S]);
              convolution2dLayer(4,4,'Padding',0);
              batchNormalizationLayer();
              reluLayer();
              maxPooling2dLayer(4,'Stride',4); 
              fullyConnectedLayer(Nw);
              softmaxLayer();
              classificationLayer()];

	% training options for the Convolutional Neural Network
    options = trainingOptions('sgdm','ExecutionEnvironment','parallel',...
                          'MaxEpochs',n_epoch,...
                          'InitialLearnRate',learn_rate,...
                          'L2Regularization',l2_factor,...
                          'ValidationData',valData,...
                          'ValidationFrequency',10,...
                          'ValidationPatience',Inf,...
                          'Shuffle','every-epoch',...
                          'Verbose',false,...
                          'Plots','training-progress');
```
5. Train the neural network and calculate recognition accuracy.
```Matlab
    [trainedNet,tr{k,1}] = trainNetwork(trainCsi,trainWord,layers,options);
    t1 = toc; % training end time

    [YTest, scores] = classify(trainedNet,testCsi);
    TTest = testWord;
    test_accuracy = sum(YTest == TTest)/numel(TTest);
    t2 = toc; % testing end time

    net_info = tr;
    perf = test_accuracy;
```

6. Plot the confusion matrix. Since there are too many classes, it is not reasonable to plot the whole confusion matrix. You can divide the classes into different categories and then plot the confusion matrix.
```Matlab
    % plot confusion matrix
    ttest = dummyvar(double(TTest))';
    tpredict = dummyvar(double(YTest))';
    [c,cm,ind,per] = confusion(ttest,tpredict);
    plotconfusion(ttest,tpredict);
```

The following figure shows an example of the training process.
![Training process](./training_screen_shot.png)