function [q, r] = uqr(A)

% compute the unique qr decomposition of A
[q r] = qr(A);
d = diag(r);
[d id] = sort(d);
d = sum(d<0);
e = eye(size(A,1));

for i = 1:d
    e(id(i),id(i)) = -1;
end

q = q*e;
r = e*r;