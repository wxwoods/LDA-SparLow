
function W = block_betweenin_matrix(number_blocks, number_each_block)
% m is the size of data vector, n is the number of data vectors, 
    %n = number_block*number_each_block;
    %block_matrix = eye(number_each_block) - 1/number_each_block * ones(number_each_block,1) * ones(number_each_block,1)';
    block_matrix_sides = 1/sqrt(number_each_block) * ones(number_each_block,1); % m by c matrix
    W_sides = [];
    for i = 1:number_blocks
        W_sides = blkdiag(W_sides,block_matrix_sides);
    end
    
    block_matrix_mid = eye(number_blocks) - 1/number_blocks * ones(number_blocks,1) * ones(number_blocks,1)';
    
    W = number_each_block*W_sides * block_matrix_mid * W_sides';
end