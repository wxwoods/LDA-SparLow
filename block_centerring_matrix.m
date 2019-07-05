
function W = block_centerring_matrix(number_block, number_each_block)
    %n = number_block*number_each_block;
    block_matrix = eye(number_each_block) - 1/number_each_block * ones(number_each_block,1) * ones(number_each_block,1)';
    W = [];
    for i = 1:number_block
        W = blkdiag(W,block_matrix);
    end
end