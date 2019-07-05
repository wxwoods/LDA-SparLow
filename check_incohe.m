function [Mutual_cohe] = check_incohe(Dictionary)

Dictionary = Dictionary/(diag(sqrt(diag(Dictionary'*Dictionary))));%
%Er=sum((Data-Dictionary*CoefMatrix).^2,1); % remove identical atoms
G=Dictionary'*Dictionary; 
G = G-diag(diag(G));
Mutual_cohe = max(G(:));