import pyautogui
import time
import random

# 關閉 fail-safe（避免滑鼠移到角落時程式中斷）
pyautogui.FAILSAFE = False

print("防閒置腳本啟動中... 按 Ctrl+C 可停止")

try:
    while True:
        # 取得目前滑鼠位置
        x, y = pyautogui.position()
        
        # 隨機微幅移動（避免完全相同的動作被偵測）
        offset_x = random.randint(-50, 50)
        offset_y = random.randint(-50, 50)
        
        # 移動滑鼠
        pyautogui.moveTo(x + offset_x, y + offset_y, duration=0.5)
        # 再移回原位
        pyautogui.moveTo(x, y, duration=0.5)

        print(f"滑鼠已移動 - {time.strftime('%H:%M:%S')}")
        # 方法一：模擬點擊（在當前位置點一下，更安全）
        pyautogui.click()


        # 方法二：按一下不影響的按鍵，例如 shift
        pyautogui.press('shift')

        # 等待 5 分鐘
        time.sleep(300)

except KeyboardInterrupt:
    print("\n腳本已停止")
