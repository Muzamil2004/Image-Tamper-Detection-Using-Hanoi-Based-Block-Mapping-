clc; clear all; close all;

%% ========== SETTINGS ==========
% Set recovery_mode = 1 to use original image 'a' for recovery (requires 'a' present)
% Set recovery_mode = 2 to reconstruct from embedded bits (no original needed)
recovery_mode = 1;

% Variance threshold for detecting near-constant blocks (e.g., black patch)
variance_threshold = 5;  % adjust if needed (lower -> stricter)

%% --- Input Image ---
a1 = imread('chest-x-ray1.png');
a1 = imresize(a1, [256 256], 'bilinear');
figure; imshow(a1); title('original image');          % 1
imwrite(a1,'image1.png');

a2 = double(a1);
[m,n] = size(a2);

%% --- Step 1: Zero out 2 LSB planes ---
a3 = my_de2bi(a2,8);   % custom de2bi
a3(:,1:2) = 0;
a4 = my_bi2de(a3);     % custom bi2de
a  = reshape(a4,[m,n]);  % 'a' is the pre-watermarked image with 2 LSBs cleared

figure; imshow(a/255); title('image after 2 LSB planes = 0');  % 2

%% --- Step 2: Check block compatibility ---
if rem(m,2)==0 && rem(n,2)==0
    disp('Image divisible into 2x2 blocks');
else
    disp('Image NOT divisible properly');
end

N = (m/2)*(n/2);   % Total blocks
disp(['Total 2x2 blocks = ',num2str(N)]);

%% --- Step 3: Generate Hanoi Shuffle ---
function idx = hanoi_shuffle(n)
    if n == 1
        idx = 1;
    else
        left = hanoi_shuffle(n-1);
        idx = [left, n, left];
    end
end

seq = hanoi_shuffle(ceil(log2(N)));
seq = unique(seq,'stable');

while length(seq) < N
    seq = [seq, setdiff(1:N, seq)];
end

I = seq(1:N);
disp(I(1:min(10,N)));

%% --- Step 4: Divide Image into 2x2 Blocks + Generate Watermark ---
row_vec = ones(1,m/2)*2;
col_vec = ones(1,n/2)*2;

c = mat2cell(a, row_vec, col_vec);

ww = cell(size(c));

for i = 1:N
    w = zeros(1,8);
    k = c{i};
    k_dct = myDCT2(k);    % custom DCT

    % -------------------------------
    % CHANGE: Use DC coefficient (1,1) for watermark generation
    % This binds the watermark to the block's frequency content.
    DC = abs(k_dct(1,1));   % absolute DC value
    % Map DC (numeric) into 6 bits by shifting / quantizing
    for j = 1:6
        % divide DC by 2^(6-j), then take integer and mod 2 -> bit
        w(j) = mod(floor(DC / 2^(6-j)), 2);
    end
    % -------------------------------

    temp = 0;
    for j = 1:6
        w7 = xor(w(j), temp);
        temp = w7;
    end

    w(7) = w7;
    w(8) = ~w7;

    ww{i} = w;
end

%% --- Step 5: Embed Watermark ---
b = zeros(m,n);
b1 = mat2cell(b, row_vec, col_vec);

for i = 1:N
    ii = I(i);
    k11 = c{ii};
    ww1 = ww{i};

    % embed bits into the two LSBs of each of the 4 pixels of the 2x2 block
    k12 = num2cell(k11);
    for j = 1:4
        pix = uint8(k12{j});
        pix = bitand(pix, uint8(252)); % clear 2 LSBs (11111100 = 252)
        twoBits = uint8(2*ww1(j) + ww1(j+4)); % MSB then LSB in those 2 bits
        k12{j} = double(bitor(pix, twoBits)); % place new 2 LSBs
    end

    b1{ii} = cell2mat(k12);
end

b2 = cell2mat(b1);
figure; imshow(b2/255); title('Watermarked Image');   % 3
imwrite(b2/255,'image2.png');

r = b2;  % Received image (no tampering in this run)
% If you want to simulate tampering for testing, uncomment below:
% r(100:150,100:150) = 0;  % simulate a big black tamper patch

%% --- Step 7: Tamper Localization Step 1 (Extract bits) ---
r1 = mat2cell(r,row_vec,col_vec);
ww2 = cell(size(c));

for i = 1:N
    r2 = r1{i};
    r3 = my_de2bi(r2,8);     % custom de2bi
    % The 2 LSB bits are in columns 1 and 2 (LSB is column 1)
    % Original code swaps these; keep same mapping to be consistent
    r4(:,1) = r3(:,2);
    r4(:,2) = r3(:,1);
    r5 = reshape(r4,[1,8]);
    ww2{i} = r5;
end

% Preliminary flag using watermark parity rules
flag1 = cell(size(c));
for i = 1:N
    f1 = ww2{i};
    kbit = 0;
    for j=1:6
        f2 = xor(f1(j), kbit);
        kbit = f2;
    end
    if f2 == f1(7) && f2 ~= f1(8)
        flag1{i} = 0; % consistent
    else
        flag1{i} = 1; % inconsistent -> suspicious
    end
end

% -------------------------------
% CHANGE: Add variance-based check (detect near-constant blocks like big black patch)
% A tampered black region will have near-zero variance -> force tamper.
for i = 1:N
    block = double(r1{i});
    v = var(block(:));
    if v < variance_threshold
        flag1{i} = 1; % force tamper detection for low-variance blocks
    end
end
% -------------------------------

flag_1 = cell2mat(flag1);
figure; imshow(flag_1); title('Flag-1');                 % 4
imwrite(flag_1,'image4.png');

% --- Create & show image after Step 1 detection (rflag1) ---
r_flag1 = cell(size(c));
for i = 1:N
    if flag1{i} == 0
        r_flag1{i} = r1{i};
    else
        r_flag1{i} = [1 1; 1 1];   % mark as white block for visualization
    end
end
rflag1 = cell2mat(r_flag1);
figure; imshow(rflag1/255); title('Image after Step 1 detection');  % 5
imwrite(rflag1/255,'image5.png');

%% --- Step 7.2: Generate WW3 for Step 2 Detection (recompute from block content) ---
ww3 = cell(size(c));

for i = 1:N
    k1 = r1{i};
    k2 = my_de2bi(k1,8);   % custom de2bi
    k2(:,1:2) = 0;         % clear 2 LSBs before recomputing
    k3 = my_bi2de(k2);     % custom bi2de
    k = reshape(k3,[2,2]);

    k_dct = myDCT2(k);     % custom DCT

    % -------------------------------
    % CHANGE: recompute using DC coefficient (1,1) instead of max
    DC = abs(k_dct(1,1));
    w = zeros(1,8);
    for j = 1:6
        w(j) = mod(floor(DC / 2^(6-j)), 2);
    end
    temp = 0;
    for j = 1:6
        w7 = xor(w(j), temp); temp = w7;
    end
    w(7) = w7;
    w(8) = ~w7;
    % -------------------------------

    ww3{i} = w;
end

%% --- Inverse Hanoi Mapping ---
I1 = zeros(1,N);
for i = 1:N
    I1(I(i)) = i;
end

%% --- Step 2 Flag (compare extracted vs recomputed) ---
flag2 = cell(size(c));
for i = 1:N
    ii = I1(i);
    if sum(ww2{i} == ww3{ii}) == 8
        flag2{i} = 1; % match
    else
        flag2{i} = 0; % mismatch -> tampered
    end
end

flag_2 = cell2mat(flag2);
figure; imshow(flag_2); title('Flag-2');                % 6
imwrite(flag_2,'image6.png');

% --- Create & show image after Step 2 detection (rflag2) ---
r_flag2 = cell(size(c));
for i = 1:N
    if flag2{i} == 0
        r_flag2{i} = r1{i};
    else
        r_flag2{i} = [0 0; 0 0];   % mark as black block for visualization
    end
end
rflag2 = cell2mat(r_flag2);
figure; imshow(rflag2/255); title('Image after Step 2 detection');  % 7
imwrite(rflag2/255,'image6b.png');

%% --- Step 3: Connectivity Check (refine) ---
[m1,n1] = size(flag_1);
flag_3 = flag_1;

for i=2:m1-1
    for j=2:n1-1
        if flag_1(i,j)==0
            T1 = sum([flag_1(i-1,j) flag_1(i-1,j+1) flag_1(i,j+1)]);
            T2 = sum([flag_1(i,j+1) flag_1(i+1,j+1) flag_1(i+1,j)]);
            T3 = sum([flag_1(i+1,j) flag_1(i+1,j-1) flag_1(i,j-1)]);
            T4 = sum([flag_1(i,j-1) flag_1(i-1,j-1) flag_1(i-1,j)]);
            if max([T1 T2 T3 T4])==3
                flag_3(i,j)=1;
            end
        end
    end
end

figure; imshow(flag_3); title('Flag-3');                 
imwrite(flag_3,'image7.png');

% --- Create & show image after Step 3 detection (rflag3) ---
r_flag3 = cell(size(c));
flag3 = num2cell(flag_3);  % keep same style as earlier code if used later
for i = 1:N
    if flag3{i} == 0
        r_flag3{i} = r1{i};
    else
        r_flag3{i} = [0 0; 0 0];
    end
end
rflag3 = cell2mat(r_flag3);
figure; imshow(rflag3/255); title('Image after Step 3 detection');  % 8
imwrite(rflag3/255,'image7b.png');

%% --- Step 4 Final Filtering ---
flag_4 = flag_3;

for i=2:m1-1
    for j=2:n1-1
        N8 = [flag_3(i-1,j-1) flag_3(i-1,j) flag_3(i-1,j+1) ...
              flag_3(i,j-1) flag_3(i,j+1) ...
              flag_3(i+1,j-1) flag_3(i+1,j) flag_3(i+1,j+1)];
        if sum(N8) > 4
            flag_4(i,j) = 1;
        else
            flag_4(i,j) = 0;
        end
    end
end

figure; imshow(flag_4); title('Flag-4');                 
imwrite(flag_4,'image8.png');

%% --- Tampered Block Map & Overlay (explicit visualization) ---
% Create a tamper map at pixel-level (2x2 blocks)
tamper_map = zeros(m,n);
k = 1;
for ii = 1:2:m
    for jj = 1:2:n
        if flag_4(k) == 1
            tamper_map(ii:ii+1, jj:jj+1) = 255;
        end
        k = k + 1;
    end
end

figure; imshow(tamper_map/255); title('Tampered Region Map (White = Tampered)');
imwrite(tamper_map/255, 'tamper_map.png');

% Overlay tampered regions on received image for easy viewing
highlighted = r;
k = 1;
for ii = 1:2:m
    for jj = 1:2:n
        if flag_4(k) == 1
            highlighted(ii:ii+1, jj:jj+1) = 255; % white highlight
        end
        k = k + 1;
    end
end

figure; imshow(highlighted/255); title('Tampered Regions Highlighted on Image');
imwrite(highlighted/255,'tampered_highlighted.png');


%% --- Step 8: Tamper Recovery (Debugged) ---
out = cell(size(r1));  % output block cell array

% Ensure flag_4 is a numeric column vector
tampered_blocks = reshape(flag_4, [], 1);  % 1 = tampered, 0 = not tampered

% Parameters for neighbor-based recovery
max_radius = 4;    % how far (in blocks) to search neighbours
weight_sigma = 1.5; % for distance weighting in neighbor average

% Precompute block grid sizes
num_r = m/2; 
num_c = n/2;
N = num_r * num_c;   % total number of blocks

% Loop over all blocks
for i = 1:N
    if flag_4(i) == 1   % tampered block => recover
        recovered_block = [];  % initialize

        % Strategy A: If original available and recovery_mode=1, use it
        if exist('recovery_mode','var') && recovery_mode == 1
            recovered_block = c{i};   % original block
        end

        % Strategy B: Neighbour-based weighted average
        if isempty(recovered_block)
            [r0, c0] = ind2sub([num_r, num_c], i);  % block row/col
            found = false;

            for R = 1:max_radius
                neigh_blocks = [];
                neigh_weights = [];

                for dr = -R:R
                    for dc = -R:R
                        if abs(dr)+abs(dc) > R, continue; end  % Manhattan shell
                        rr = r0 + dr; cc = c0 + dc;
                        if rr < 1 || rr > num_r || cc < 1 || cc > num_c, continue; end
                        idx = sub2ind([num_r, num_c], rr, cc);
                        if idx == i, continue; end
                        if flag_4(idx) == 0   % non-tampered neighbour
                            neigh = r1{idx};
                            dist = sqrt(dr^2 + dc^2);
                            w = exp(-(dist^2)/(2*weight_sigma^2));
                            neigh_blocks(:,:,end+1) = neigh; %#ok<SAGROW>
                            neigh_weights(end+1) = w; %#ok<SAGROW>
                        end
                    end
                end

                if ~isempty(neigh_weights)
                    W = reshape(neigh_weights,1,1,[]);
                    weighted = sum(bsxfun(@times, neigh_blocks, W), 3) ./ sum(W);
                    recovered_block = round(weighted);
                    found = true;
                    break;  % stop expanding search
                end
            end

            % Strategy C: fallback DC-based reconstruction
            if ~found
                r11 = r1{i};
                r12 = my_de2bi(r11,8);  % custom de2bi function
                r13(:,1) = r12(:,2);
                r13(:,2) = r12(:,1);
                r14 = reshape(r13,[1,8]);
                k14 = 0;
                for j = 1:6
                    k14 = k14 + r14(j) * 2^(6-j);
                end
                Dc1 = double(k14)^2;
                DC = myIDCT2([Dc1 0; 0 0]);  % custom IDCT2
                rec_block = round(DC);
                rec_block(rec_block < 0) = 0;
                rec_block(rec_block > 255) = 255;
                recovered_block = rec_block;
            end
        end

        % assign recovered block
        out{i} = recovered_block;

    else
        % not tampered - keep received block
        out{i} = r1{i};
    end
end

%% --- Convert blocks to image ---
out_img = cell2mat(out);
out_img_d = double(out_img);

%% --- Post-process: iterative diffusion fill to smooth tampered blocks ---
tamper_mask = zeros(m,n);
k = 1;
for rr = 1:2:m
    for cc = 1:2:n
        if flag_4(k) == 1
            tamper_mask(rr:rr+1, cc:cc+1) = 1;
        end
        k = k + 1;
    end
end

num_iter = 60;
for it = 1:num_iter
    A = out_img_d;
    up = [A(1,:); A(1:end-1,:)];
    down = [A(2:end,:); A(end,:)];
    left = [A(:,1), A(:,1:end-1)];
    right = [A(:,2:end), A(:,end)];
    neighbor_sum = up + down + left + right;
    neighbor_count = 4;
    new_vals = neighbor_sum / neighbor_count;
    % update only masked pixels
    out_img_d(tamper_mask==1) = 0.6*out_img_d(tamper_mask==1) + 0.4*new_vals(tamper_mask==1);
end

out_img_d(out_img_d < 0) = 0;
out_img_d(out_img_d > 255) = 255;
out_final = uint8(round(out_img_d));

%% --- Show & save result ---
figure; imshow(out_final/255); title('Tamper Recovered Image (Neighbour + Diffusion)');
imwrite(out_final/255,'image9.png');



%% ============= Custom Functions (NO TOOLBOX REQUIRED) ============= %%
function b = my_de2bi(x, bits)
    x = x(:);
    b = zeros(length(x), bits);
    for k = 1:bits
        b(:, k) = bitget(uint8(x), k);
    end
end

function x = my_bi2de(b)
    bits = size(b,2);
    x = zeros(size(b,1),1);
    for k = 1:bits
        x = x + b(:,k) * 2^(k-1);
    end
end

% ----------------- 1D DCT (Type-II) -----------------
function y = myDCT1D(x)
    x = double(x(:));
    N = length(x);
    y = zeros(N,1);
    for k = 0:N-1
        sum_val = 0;
        for n = 0:N-1
            sum_val = sum_val + x(n+1) * cos(pi*(n + 0.5) * k / N);
        end
        if k == 0
            alpha = sqrt(1/N);
        else
            alpha = sqrt(2/N);
        end
        y(k+1) = alpha * sum_val;
    end
end

% ----------------- 1D IDCT (Type-III) -----------------
function x = myIDCT1D(X)
    X = double(X(:));
    N = length(X);
    x = zeros(N,1);
    for n = 0:N-1
        sum_val = 0;
        for k = 0:N-1
            if k == 0
                alpha = sqrt(1/N);
            else
                alpha = sqrt(2/N);
            end
            sum_val = sum_val + alpha * X(k+1) * cos(pi*(n + 0.5) * k / N);
        end
        x(n+1) = sum_val;
    end
end

% ----------------- 2D DCT using custom 1D DCT -----------------
function D = myDCT2(B)
    B = double(B);
    [M, N] = size(B);
    D = zeros(M,N);
    % Apply 1D DCT to each row
    for i = 1:M
        D(i,:) = myDCT1D(B(i,:)).';
    end
    % Apply 1D DCT to each column
    for j = 1:N
        D(:,j) = myDCT1D(D(:,j));
    end
end

% ----------------- 2D IDCT using custom 1D IDCT -----------------
function B = myIDCT2(D)
    D = double(D);
    [M, N] = size(D);
    B = zeros(M,N);
    % Apply 1D IDCT to each row
    for i = 1:M
        B(i,:) = myIDCT1D(D(i,:)).';
    end
    % Apply 1D IDCT to each column
    for j = 1:N
        B(:,j) = myIDCT1D(B(:,j));
    end
end
