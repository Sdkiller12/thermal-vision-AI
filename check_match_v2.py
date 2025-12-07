
import os

rgb_dir = r"c:\Users\byteghost\Desktop\vision4\data\images_rgb_train\data"
thermal_dir = r"c:\Users\byteghost\Desktop\vision4\data\images_thermal_train\analyticsData"

def get_keys_from_dir(d):
    keys = set()
    if not os.path.exists(d):
        print(f"Dir not found: {d}")
        return keys
        
    for f in os.listdir(d):
        parts = f.split('-')
        # Expecting video-ID-frame-NUM-...
        if len(parts) >= 4 and parts[0] == 'video':
            key = "-".join(parts[:4])
            keys.add(key)
    return keys

rgb_keys = get_keys_from_dir(rgb_dir)
thermal_keys = get_keys_from_dir(thermal_dir)

print(f"RGB Keys: {len(rgb_keys)}")
print(f"Thermal Keys: {len(thermal_keys)}")
common = rgb_keys.intersection(thermal_keys)
print(f"Common: {len(common)}")
if common:
    print(f"Example: {list(common)[0]}")
