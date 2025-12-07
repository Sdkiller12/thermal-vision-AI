
import os

def get_keys(filename):
    try:
        with open(filename, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        # Extract Key: video-ID-frame-NUM
        keys = set()
        for x in files:
            parts = x.split('-')
            # Look for 'video' and 'frame'
            if 'video' in parts and 'frame' in parts:
                # Assuming format video-ID-frame-NUM-...
                # Check indices
                try:
                    v_idx = parts.index('video')
                    f_idx = parts.index('frame')
                    # ID is between video and frame?
                    # actually split by '-' might give ['video', 'ID', 'frame', 'NUM', ...]
                    # Let's reconstruct standard key
                    # video-ID-frame-NUM
                    # We need to be careful if ID has hyphens. 
                    # But usually ID is one block.
                    # Let's just take everything before the LAST dash?
                    # Or just take the first 4 parts if standard?
                    # file example: video-24ysbPEGoEKKDvRt6-frame-000195-WyEiQ9whdSabDBFz7.tiff
                    # Parts: [video, 24ysb..., frame, 000195, WyEi..., tiff]
                    # Key: video-24ysb...-frame-000195
                    if len(parts) >= 4:
                        key = "-".join(parts[:4])
                        keys.add(key)
                except ValueError:
                    pass
        return keys
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return set()

rgb_keys = get_keys('rgb_list.txt')
thermal_keys = get_keys('thermal_list.txt')

common = rgb_keys.intersection(thermal_keys)
print(f"RGB Files: {len(rgb_keys)}")
print(f"Thermal Files: {len(thermal_keys)}")
print(f"Common Keys: {len(common)}")

if len(common) > 0:
    print("Example common:", list(common)[0])
