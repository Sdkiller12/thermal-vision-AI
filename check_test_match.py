
import os

rgb_dir = r"c:\Users\byteghost\Desktop\vision4\data\video_rgb_test\data"
thermal_dir = r"c:\Users\byteghost\Desktop\vision4\data\video_thermal_test\analyticsData"

def get_keys_from_dir(d):
    keys = set()
    if not os.path.exists(d):
        return keys 
    for f in os.listdir(d):
        parts = f.split('-')
        if len(parts) >= 4 and parts[0] == 'video':
            key = "-".join(parts[:4])
            keys.add(key)
    return keys

rgb_keys = get_keys_from_dir(rgb_dir)
thermal_keys = get_keys_from_dir(thermal_dir)

print(f"Test RGB Keys: {len(rgb_keys)}")
print(f"Test Thermal Keys: {len(thermal_keys)}")
common = rgb_keys.intersection(thermal_keys)
print(f"Test Common: {len(common)}")
