import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 .tsv 文件
file_path = '/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/train.tsv'
data = pd.read_csv(file_path, sep='\t')
site_count = []
# 筛选 'site' 列中值为 3 的行
for i in range(0, 64):
    site_count.append(data[data['site'] == i].shape[0])

# 打印结果
print("Number of elements in 'site' column: ", site_count)    # [18, 244, 14, 866, 84, 40, 29, 12, 41, 18, 241, 49, 21, 0, 32, 0, 142, 216, 28, 54, 15, 10, 42, 20, 151, 66, 27, 31, 34, 10, 23, 0, 45, 7, 38, 21, 55, 36, 11, 21, 18, 21, 50, 34, 10, 12, 18, 12, 0, 31, 20, 12, 17, 8, 26, 24, 24, 0, 24, 0, 9, 14, 17, 14]

top_four = sorted(site_count, reverse=True)[:5]   
site_3 = sorted(site_count, reverse=True)[:1]

print("Top four elements:", top_four)    # [866, 244, 241, 216, 151]
print("Most elements:", site_3)    # [866]


indexed_data = list(enumerate(site_count))

# 根据值进行降序排序
sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1], reverse=True)

# 获取前四大元素的索引
top_four_indices = [index for index, value in sorted_indexed_data[:5]]  
site_3_indices = [index for index, value in sorted_indexed_data[:1]]

print("Indices of top four elements:", top_four_indices)   # [3, 1, 10, 17, 24]
print("Indices of site 3 elements:", site_3_indices)    # [3]


sites_of_interest = [1, 10, 17, 24]
site_3 = [3]
filtered_data_top2to5 = data[data['site'].isin(sites_of_interest)]
filtered_data_top1 = data[data['site'].isin(site_3)]
filtered_indices_top2to5 = filtered_data_top2to5.index.tolist()
filtered_indices_site3 = filtered_data_top1.index.tolist()
selected_site3 = np.random.choice(filtered_indices_site3, size=200, replace=False)
print('filtered_indices_top2to5: ', len(filtered_indices_top2to5))
print('length of filtered_indices_top1: ', len(selected_site3))
print('filtered_indices_top1: ', selected_site3)

balanced_indices = selected_site3.tolist() + filtered_indices_top2to5
print('length of balanced indices: ', len(balanced_indices))
print('balanced indices: ', balanced_indices)



x_arr = np.load('/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/openBHB/brain_age_with_site_removal-main/data/train.npy', mmap_mode="r")
y_arr = data[["age", "site"]].values
# x_arr_balanced = x_arr[balanced_indices]
# y_arr_balanced = y_arr[balanced_indices]
print("- x size [original]:", x_arr.shape)
print("- y size [original]:", y_arr.shape)
# print("- x size [balanced]:", x_arr_balanced.shape)
# print("- y size [balancced]:", y_arr_balanced.shape)

# 查看筛选后的数据
# print(filtered_data)
# print("Indices of filtered rows:", filtered_indices)

# plt.plot(site_count)

# # 添加标题和标签
# # plt.title('Line Graph of Data')
# plt.xlabel('site number')
# plt.ylabel('counts')

# # 显示图表
# plt.show()
# plt.savefig('/home/jiaxia/unet_test/contrastive-brain-age-prediction/src/site_distribution.jpg')