clear;
NumBlock = 20000;

x = -ones(1, NumBlock * 24);
ber = zeros(1, 8);
ber2 = zeros(1, 8);
EbNo = 1:8;

tic
parfor idx  = 1:8
    c = zeros(NumBlock, 12);
    y1 = awgn(x, EbNo(idx)+3, 'measured');
    y1 = (sign(y1) + 1)/2;
    ber2(idx) = sum(y1) / 24 / NumBlock;
    y = awgn(x, EbNo(idx), 'measured');
    y = (sign(y) + 1)/2;
    y = reshape(y, NumBlock, 24);

    for n = 1:NumBlock
        d = DecGolay(y(n, :));
        c(n, :) = d(1:12);
    end
    ber(idx) = sum(sum(c)) / 12 / NumBlock;    
end

semilogy(EbNo, ber, EbNo, ber2)
toc