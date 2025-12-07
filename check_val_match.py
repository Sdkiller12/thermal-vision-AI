
import os

rgb_dir = r"c:\Users\byteghost\Desktop\vision4\data\images_rgb_val\data"
thermal_dir = r"c:\Users\byteghost\Desktop\vision4\data\images_thermal_val\analyticsData"

def get_keys_from_dir(d):
    keys = set()
    if not os.path.exists(d):
        # Try without subdirs?
        # Listing parent to see structure
        parent = os.path.dirname(d)
        if os.path.exists(parent):
             # Maybe struct is different
             pass
        return keys
        
    for f in os.listdir(d):
        parts = f.split('-')
        if len(parts) >= 4 and parts[0] == 'video':
            key = "-".join(parts[:4])
            keys.add(key)
    return keys

rgb_keys = get_keys_from_dir(rgb_dir)
thermal_keys = get_keys_from_dir(thermal_dir)

print(f"Val RGB Keys: {len(rgb_keys)}")
print(f"Val Thermal Keys: {len(thermal_keys)}")
common = rgb_keys.intersection(thermal_keys)
print(f"Val Common: {len(common)}")
