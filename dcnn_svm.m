function dcnn_svm(class1_dir, class2_dir)
    fprintf('Starting DCNN + Linear SVM Classification: %s vs %s\n', class1_dir, class2_dir);

    % 画像ファイル取得
    img_files1 = get_image_files(class1_dir);
    img_files2 = get_image_files(class2_dir);
    img_files = [img_files1; img_files2];

    num_samples = min(length(img_files1), length(img_files2));
    if num_samples == 0
        error('No images found in %s or %s', class1_dir, class2_dir);
    end
    labels = [ones(num_samples,1); -ones(num_samples,1)];

    % DCNN特徴量の計算
    fprintf('Extracting DCNN features...\n');
    features_dcnn = dcnn_feature_extraction(img_files);
    fprintf('DCNN feature extraction DONE\n');

    % 5-fold cross-validation
    fprintf('Starting classification...\n');
    accuracy_dcnn_svm = perform_svm_dcnn_classification(features_dcnn, labels);
    fprintf('DCNN + Linear SVM Accuracy: %.2f%%\n', mean(accuracy_dcnn_svm) * 100);
end

function features = dcnn_feature_extraction(img_files)
    net = vgg16;
    inputSize = net.Layers(1).InputSize(1:2);
    numImages = numel(img_files);
    totalFeatureDim = 4096;
    features = zeros(numImages, totalFeatureDim, 'single');
    fprintf('Extracting DCNN features using VGG16...\n');
    for i = 1:numImages
        try
            img = imread(img_files{i});
            img = imresize(img, inputSize);
            activations_fc7 = activations(net, img, 'fc7', 'OutputAs', 'rows');
            features(i, :) = activations_fc7;
        catch
            continue;
        end
    end
    fprintf('DCNN feature extraction DONE\n');
end

function files = get_image_files(folder)
    formats = {'*.jpg', '*.jpeg', '*.JPG', '*.JPEG'};
    files = {};
    for i = 1:length(formats)
        imgs = dir(fullfile(folder, formats{i}));
        img_paths = fullfile(folder, {imgs.name});
        files = [files, img_paths];
    end
    files = files';
end

function accuracy_dcnn_svm = perform_svm_dcnn_classification(features_dcnn, labels)
    num_folds = 5; % 5-fold クロスバリデーション
    cv = cvpartition(length(labels), 'KFold', num_folds);
    accuracy_dcnn_svm = zeros(num_folds,1);

    for k = 1:num_folds
        trainIdx = training(cv, k);
        testIdx = test(cv, k);

        % 線形SVMを学習
        dcnnModel = fitcsvm(features_dcnn(trainIdx,:), labels(trainIdx), ...
            'KernelFunction', 'linear', 'BoxConstraint', 10);

        preds = predict(dcnnModel, features_dcnn(testIdx,:));

        % 正解率の計算
        accuracy_dcnn_svm(k) = sum(preds == labels(testIdx)) / length(preds);
    end

    % 各 fold の分類精度の平均を返す
    accuracy_dcnn_svm = mean(accuracy_dcnn_svm);
end
