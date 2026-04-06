import yaml

# 這裡貼上你剛剛得到的 25 題 ID
correct_ids = [
    'test-1429', 'test-1264', 'test-1470', 'test-1084', 'test-1453', 
    'test-1710', 'test-1673', 'test-1028', 'test-1920', 'test-1708', 
    'test-1031', 'test-1833', 'test-1005', 'test-1009', 'test-1426', 
    'test-1442', 'test-1604', 'test-1106', 'test-1832', 'test-1114', 
    'test-1445', 'test-1776', 'test-1504', 'test-1697', 'test-1045'
]

error_ids = [
    'test-1593', 'test-1881', 'test-1001', 'test-1553', 'test-1577', 
    'test-1270', 'test-1490', 'test-1738', 'test-1979', 'test-1109', 
    'test-1951', 'test-1012', 'test-1810', 'test-1065', 'test-1842', 
    'test-1570', 'test-1008', 'test-1890', 'test-1625', 'test-1707', 
    'test-1825', 'test-1457', 'test-1880', 'test-1896', 'test-1666'
]

# 建立 YAML 資料結構
yaml_data = {
    'experiment_info': {
        'description': '自動抽樣的 50 題測試清單',
        'total_count': len(correct_ids) + len(error_ids),
        'status': 'ready_to_run'
    },
    'target_ids': {
        'correct_samples': correct_ids,
        'error_samples': error_ids
    }
}

# 寫入檔案
file_name = 'config.yaml'
with open(file_name, 'w', encoding='utf-8') as f:
    # sort_keys=False 可以保持我們定義的順序，不會被英文字母排序搞亂
    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"✅ 完成！'{file_name}' 已經乖乖躺在你的資料夾裡了。")