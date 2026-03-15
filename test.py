# 建立一個 hello.py
name = input("你叫什麼名字？")
print(f"Hello, {name}!")

# 在 terminal 輸入：
# python hello.py

for i in range(1, 16):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
