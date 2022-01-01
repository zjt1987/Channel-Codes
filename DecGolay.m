function y = DecGolay(x)
     P = [0     1     1     1     1     1     1     1     1     1     1     1
          1     1     1     0     1     1     1     0     0     0     1     0
          1     1     0     1     1     1     0     0     0     1     0     1
          1     0     1     1     1     0     0     0     1     0     1     1
          1     1     1     1     0     0     0     1     0     1     1     0
          1     1     1     0     0     0     1     0     1     1     0     1
          1     1     0     0     0     1     0     1     1     0     1     1
          1     0     0     0     1     0     1     1     0     1     1     1
          1     0     0     1     0     1     1     0     1     1     1     0
          1     0     1     0     1     1     0     1     1     1     0     0
          1     1     0     1     1     0     1     1     1     0     0     0
          1     0     1     1     0     1     1     1     0     0     0     1];
    I = eye(12);
    H = [I P];    
    s = mod(H * x', 2);
    s1 = mod(P * s, 2);
    sp = mod(s + P, 2);
    s1p = mod(s1 + P', 2);
    
    c1 = sum(s) <= 3;
    c2 = find(sum(sp) <= 2);
    c3 = sum(s1) <= 3;
    c4 = find(sum(s1p) <= 2);
    
    if c1
        e = [s' zeros(1, 12)];
    elseif ~isempty(c2)        
        e = [transpose(sp(:, c2)) I(c2, :)];
    elseif c3
        e = [zeros(1, 12) s1'];
    elseif ~isempty(c4)
        e = [I(c4, :) transpose(s1p(:, c4))];
    else
        e = zeros(1, 24);
    end
    y = mod(x + e, 2);
end