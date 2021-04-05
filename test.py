import heapq
f = open('EnglishNews/News103271.txt', 'r')
print(f.read())

nums = [1, 8, 2, 23, 7, -4, 18, 23, 24, 37, 2]

# 最大的3个数的索引
max_num_index_list = map(nums.index, heapq.nlargest(3, nums))

# 最小的3个数的索引
min_num_index_list = map(nums.index, heapq.nsmallest(3, nums))

print(list(max_num_index_list))
print(list(min_num_index_list))
dict_1 = {}
for i in range(4):
    dict_1.update({i: i})
print(dict_1.keys())


seq = ("a", "b", "c")  # 字符串序列
print(' '.join(seq))
