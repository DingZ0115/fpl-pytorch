import re

number_str = "tensor(61)"

# 使用正则表达式从字符串中提取数字
matches = re.findall(r'\d+', number_str)

if matches:
    # 将找到的第一个数字转换为整数
    number = int(matches[0])
    print(number)  # 输出: 61
else:
    print("No number found in the string.")
