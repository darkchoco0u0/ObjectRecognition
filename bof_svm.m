function bof_svm(class1_dir, class2_dir)
    fprintf('Starting BoF + RBF SVM Classification: %s vs %s\n', class1_dir, class2_dir);

    % 画像ファイル取得
    img_files1 = get_image_files(class1_dir);
    img_files2 = get_image_files(class2_dir);
    img_files = [img_files1; img_files2];

    if isempty(img_files1) || isempty(img_files2)
        error('No images found in %s or %s', class1_dir, class2_dir);
    end
    labels = [ones(length(img_files1),1); -ones(length(img_files2),1)];

    % コードブック作成
    create_codebook(img_files);

    % BoF特徴量の計算
    fprintf('Creating BoF matrix...\n');
    features_bof = create_bof_matrix(img_files);
    fprintf('BoF matrix DONE\n');

    % 5-fold cross-validation
    fprintf('Starting classification...\n');
    accuracy_svm_rbf = perform_svm_bof_classification(features_bof, labels);
    fprintf('BoF + RBF SVM Accuracy: %.2f%%\n', mean(accuracy_svm_rbf) * 100);
end

function files = get_image_files(folder)
    formats = {'*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.bmp'};
    files = {};
    for i = 1:length(formats)
        imgs = dir(fullfile(folder, formats{i}));
        img_paths = fullfile(folder, {imgs.name});
        files = [files, img_paths];
    end
    files = files';
end

function create_codebook(image_files)
    fprintf('Extracting SURF features...\n');
    Features = [];
    for j = 1:length(image_files)
        try
            if ~exist(image_files{j}, 'file')
                continue;
            end
            I = imread(image_files{j});
            if size(I,3) == 3
                I = rgb2gray(I);
            end
            p = detectSURFFeatures(I);
            [f, ~] = extractFeatures(I, p);
            Features = [Features; f];
        catch
            fprintf('Skipping unreadable image: %s\n', image_files{j});
            continue;
        end
    end
    if size(Features, 1) > 50000
        Features = Features(randperm(size(Features, 1), 50000), :);
    end
    k = 500;
    [~, CODEBOOK] = kmeans(Features, k);
    save('codebook.mat', 'CODEBOOK');
end

function bof = create_bof_matrix(image_files)
    load('codebook.mat', 'CODEBOOK');
    k = size(CODEBOOK, 1);
    n = length(image_files);
    bof = zeros(n, k);
    for j = 1:n
        try
            if ~exist(image_files{j}, 'file')
                continue;
            end
            I = imread(image_files{j});
            if size(I,3) == 3
                I = rgb2gray(I);
            end
            p = detectSURFFeatures(I);
            [features, ~] = extractFeatures(I, p);
            for i = 1:size(features, 1)
                distances = sum((CODEBOOK - features(i, :)).^2, 2);
                [~, index] = min(distances);
                bof(j, index) = bof(j, index) + 1;
            end
        catch
            fprintf('Skipping unreadable image: %s\n', image_files{j});
            continue;
        end
    end
    bof = bof ./ sum(bof, 2);
    save('bof_matrix.mat', 'bof');
end

function accuracy_svm_rbf = perform_svm_bof_classification(features_bof, labels)
    num_folds = 5; % 5-fold クロスバリデーション
    cv = cvpartition(length(labels), 'KFold', num_folds);
    accuracy_svm_rbf = zeros(num_folds,1);

    for k = 1:num_folds
        trainIdx = training(cv, k);
        testIdx = test(cv, k);

        % SVMモデルを学習
        rbfModel = fitcsvm(features_bof(trainIdx,:), labels(trainIdx), ...
            'KernelFunction', 'rbf', 'KernelScale', 'auto');

        preds = predict(rbfModel, features_bof(testIdx,:));

        % 正解率の計算
        accuracy_svm_rbf(k) = sum(preds == labels(testIdx)) / length(preds);
    end

    % 各 fold の分類精度の平均を返す
    accuracy_svm_rbf = mean(accuracy_svm_rbf);
end
