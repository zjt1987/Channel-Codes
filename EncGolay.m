function y = EncGolay(x)
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
     G = [eye(12) P];
     y = mod(x*G, 2);     
end