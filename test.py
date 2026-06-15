import pyautogui
import time

# 指定要點擊的位置，改成你的座標
X = 655
Y = 818

# 每 3 分鐘點一次
INTERVAL = 60

# 安全機制：把滑鼠移到螢幕左上角可強制停止
pyautogui.FAILSAFE = True

print("開始自動點擊。把滑鼠移到螢幕左上角可停止。")
print(f"每 {INTERVAL} 秒點擊一次：({X}, {Y})")

try:
    while True:
        pyautogui.click(X, Y)
        print(f"已點擊 ({X}, {Y})")
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("已手動停止。")
except pyautogui.FailSafeException:
    print("滑鼠移到左上角，已安全停止。")


# try:
#     while True:
#         x, y = pyautogui.position()
#         print(f"目前位置：({x}, {y})", end="\r")
#         time.sleep(0.1)

# except KeyboardInterrupt:
#     print("\n結束。")