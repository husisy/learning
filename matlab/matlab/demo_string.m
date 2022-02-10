%% string
% https://www.mathworks.com/help/matlab/matlab_prog/analyze-text-data-with-string-arrays.html
tmp0 = fileread("sonnets.txt"); %(char,(1,100266))
z0 = splitlines(string(tmp0)); %(string,(2625,1))

z0 = replace(z0, [".","?","!",",",";",":"], " "); %replace punctuations with space
z0 = strtrim(z0);
z0(z0=="") = []; %remove empty row

z1 = cell(size(z0));
for ind0 = 1:size(z0,1)
    z1{ind0} = split(z0(ind0));
end
z1 = lower(flat_array(z1));

[word,~,ind0] = unique(z1);
num_occurance = histcounts(ind0, size(word,1));

%sort by occurance
[num_occurance,ind0] = sort(num_occurance, 'descend');
word = word(ind0);

% figure
loglog(num_occurance)
xlabel('Rank of word (most to least common)');
ylabel('Number of Occurrences');

function ret = flat_array(z0)
assert(all(cellfun(@(x) size(x,2), z0)==1));
num0 = cellfun(@(x) size(x,1), z0);
ind_end = cumsum(num0);
ind_start = [1; ind_end(1:(end-1))+1];
ret = strings(sum(num0), 1);
for ind0 = 1:size(ind_end,1)
    ret(ind_start(ind0):ind_end(ind0)) = z0{ind0};
end
end
