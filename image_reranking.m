function image_reranking(keyword, n)
    if ~ismember(n, [25, 50])
        error('n must be either 25 or 50');
    end
    
    % 画像URLリストを取得 (ポジティブ画像をFlickrから、ネガティブ画像をランダム取得)
    positive_output_txt = 'flickr_positive_urls.txt';
    fetch_flickr_images_uec(keyword, 50, 4, positive_output_txt);
    
    negative_images_folder = '/MATLAB Drive/bgimg';
    negative_images = get_negative_images(negative_images_folder, 500);
    
    % DCNN特徴抽出 (VGG-16)
    pos_features = extract_dcnn_features_from_urls(readlines(positive_output_txt), 'vgg16');
    neg_features = extract_dcnn_features_batch(negative_images, 'vgg16');
    
    % SVM 学習データの準備
    X_train = [pos_features; neg_features];
    y_train = [ones(size(pos_features,1),1); -ones(size(neg_features,1),1)];
    
    % SVM モデル学習
    SVMModel = fitcsvm(X_train, y_train, 'KernelFunction', 'linear');
    save('svm_model.mat', 'SVMModel');
    
    % テストデータを収集
    flickr_output_txt = 'flickr_test_urls.txt';
    fetch_flickr_images_uec(keyword, 300,0, flickr_output_txt);
    
    % テストデータの特徴抽出と分類
    rerank_flickr_images(n, flickr_output_txt);
end

function rerank_flickr_images(n, txt_file)
    urls = readlines(txt_file);
    num_flickr_images = length(urls);
    
    % DCNN特徴抽出 (VGG-16)
    test_features = extract_dcnn_features_from_urls(urls, 'vgg16');
    
    % 事前学習済みSVMモデルの読み込み
    load('svm_model.mat', 'SVMModel');
    [label, score] = predict(SVMModel, test_features);
    
    % ソート
    [sorted_score, sorted_idx] = sort(score(:,2), 'descend');
    
    % 上位画像のURLを保存
    reranked_output_txt = 'reranked_flickr_urls.txt';
    fileID = fopen(reranked_output_txt, 'w');
    for i = 1:min(50, numel(sorted_idx))
        fprintf(fileID, '%s %f\n', urls(sorted_idx(i)), sorted_score(i));
    end
    fclose(fileID);
    fprintf('リランキング後の上位50枚の画像URLを %s に保存しました。\n', reranked_output_txt);
    
    % 上位画像を表示
    for i = 1:min(5, numel(sorted_idx))
        url = urls(sorted_idx(i));
        fprintf('Rank %d: %s %f\n', i, url, sorted_score(i));

        retry_count = 0;
        success = false;
        while retry_count < 3 && ~success
            try
                img = webread(url);
                success = true;
            catch
                fprintf('Warning: Failed to fetch %s (Attempt %d)\n', url, retry_count + 1);
                retry_count = retry_count + 1;
                pause(1);
            end
        end

        if success
            figure;
            imshow(img);
            title(sprintf('Rank %d: %s %f', i, url, sorted_score(i)));
        else
            fprintf('Error: Could not fetch %s after 3 attempts.\n', url);
        end

        pause(1);
    end
end

function fetch_flickr_images_uec(keyword, n, k, output_txt)
    base_url = "https://mm.cs.uec.ac.jp/tutorial/flickr.cgi";
    params = "?WORD=" + urlencode(keyword) + "&ORDER=" + k +"&PER_PAGE=" + num2str(n);
    
    % APIリクエスト
    url = base_url + params;
    response = webread(url);

    % XML形式のレスポンスから画像URLを抽出
    photo_pattern = '<photo id="([^"]+)" owner="[^"]+" secret="([^"]+)" server="([^"]+)" farm="([^"]+)"';
    tokens = regexp(response, photo_pattern, 'tokens');

    if isempty(tokens)
        error('No image URLs were extracted. Check the API response.');
    end

    % 画像URLを作成
    urls = cellfun(@(x) sprintf('https://live.staticflickr.com/%s/%s_%s.jpg', x{3}, x{1}, x{2}), tokens, 'UniformOutput', false);
    urls = urls(1:min(n, length(urls)));  % 上位n件を取得

    % ファイルに保存
    fileID = fopen(output_txt, 'w');
    fprintf(fileID, '%s\n', urls{:});
    fclose(fileID);

    fprintf('Flickr画像URLリストを %s に保存しました。\n', output_txt);
end

function features = extract_dcnn_features_from_urls(urls, model_name)
    net = eval(model_name);
    input_size = net.Layers(1).InputSize(1:2);
    num_features = 4096; 
    features = zeros(length(urls), num_features);
    
    for i = 1:length(urls)
        try
            img = imread(urls(i));
            img = imresize(img, input_size);
            features(i, :) = activations(net, img, 'fc7', 'OutputAs', 'rows');
        catch
            fprintf('Warning: Failed to process %s\n', urls(i));
        end
    end
end

function images = get_negative_images(folder, num_images)
    files = get_image_files(folder);
    num_files = length(files);
    if num_files < num_images
        error('Not enough images in the folder. Found %d, required %d.', num_files, num_images);
    end
    images = files(1:num_images);
end

function files = get_image_files(folder)
    formats = {'*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.bmp'};
    files = {};
    for i = 1:length(formats)
        imgs = dir(fullfile(folder, formats{i}));
        img_paths = fullfile(folder, {imgs.name});
        files = [files, img_paths];
    end
    files = files'; % 縦ベクトルに変換
end

function features = extract_dcnn_features_batch(images, model_name)
    net = eval(model_name);
    input_size = net.Layers(1).InputSize(1:2);
    num_features = 4096; % VGG-16 の特徴次元数
    batch_size = 50; % 50枚ずつ処理

    num_images = length(images);
    features = zeros(num_images, num_features);

    for i = 1:batch_size:num_images
        batch_end = min(i + batch_size - 1, num_images);
        batch_images = images(i:batch_end);
        batch_data = zeros([input_size, 3, length(batch_images)], 'single');

        for j = 1:length(batch_images)
            try
                img = imread(batch_images{j});
                img = imresize(img, input_size);
                batch_data(:,:,:,j) = single(img) / 255; 
            catch
                fprintf('Warning: Failed to process %s\n', batch_images{j});
            end
        end

        % VGG-16 の特徴抽出
        features(i:batch_end, :) = activations(net, batch_data, 'fc7', 'OutputAs', 'rows');
    end
end
