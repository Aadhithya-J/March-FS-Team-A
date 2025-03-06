def punishment(n):
    nums = [1,9,10,36,45,55,82,91,99,100,235,297,369,370,379,414,657,675,703,756,792,909,918,945,964,990,991,999,1000]
    res = 1
    for i in range(1,n+1):
        if i in nums:
            prod = i * i
            res += prod
    return res
print(punishment(10))
#Time Complexity : O(n)
#Space Complexity: O(27)