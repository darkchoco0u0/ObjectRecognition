function color_hist_knn(class1_dir, class2_dir)
    fprintf('Starting Color Histogram + KNN Classification: %s vs %s\n', class1_dir, class2_dir);

    % 画像ファイル取得
    img_files1 = get_image_files(class1_dir);
    img_files2 = get_image_files(class2_dir);
    img_files = [img_files1; img_files2];

    num_samples = min(length(img_files1), length(img_files2));
    if num_samples == 0
        error('No images found in %s or %s', class1_dir, class2_dir);
    end
    labels = [ones(num_samples,1); -ones(num_samples,1)];

    % カラーヒストグラム特徴抽出
    fprintf('Extracting Color Histogram features...\n');
    features_hist = extract_color_histograms(img_files);
    fprintf('Feature extraction DONE\n');

    % 5-fold cross-validation
    fprintf('Starting classification...\n');
    accuracy_knn = perform_knn_classification(features_hist, labels);
    fprintf('Color Histogram + KNN Accuracy: %.2f%%\n', mean(accuracy_knn) * 100);
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

function features = extract_color_histograms(img_files)
    numImages = numel(img_files);
    features = zeros(numImages, 64);
    for i = 1:numImages
        try
            img = imread(img_files{i});
            features(i, :) = extract_hist(img);
        catch
            continue;
        end
    end
end

function hist = extract_hist(img)
    img = imresize(img, [64, 64]);
    img = double(img) / 255;
    hist = histcounts(img(:), 64, 'Normalization', 'probability');
end

function accuracy_knn = perform_knn_classification(features_hist, labels)
    num_folds = 5; % 5-fold クロスバリデーション
    cv = cvpartition(length(labels), 'KFold', num_folds);
    accuracy_knn = zeros(num_folds,1);

    for k = 1:num_folds
        trainIdx = training(cv, k);
        testIdx = test(cv, k);

        % KNN モデルの学習
        mdl_knn = fitcknn(features_hist(trainIdx,:), labels(trainIdx));

        preds = predict(mdl_knn, features_hist(testIdx,:));

        % 正解率の計算
        accuracy_knn(k) = sum(preds == labels(testIdx)) / length(preds);
    end

    % 各foldの分類精度の平均を返す
    accuracy_knn = mean(accuracy_knn);
end