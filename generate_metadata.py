import json

# Configuration
INPUT_FILE = r'C:\Users\nandh\deepfake_project\dataset\labels.txt'  # Path to your input file
OUTPUT_FILE = r'C:\Users\nandh\deepfake_project\metadata.json'

def process_line(line):
    """Process a single line and return (video_path, label) or None for invalid lines"""
    parts = line.strip().split()
    if not parts or len(parts) < 2:
        return None  # Skip empty lines
    
    # Extract label from first word (expecting format "haveX")
    label_char = parts[0][-1] if parts[0].startswith('have') else None
    
    # Extract video path (handle paths with spaces)
    video_path = ' '.join(parts[1:])
    
    if label_char == '1':
        return video_path, "REAL"
    elif label_char == '0':
        return video_path, "FAKE"
    return None

# Process all lines
metadata = {}
with open(INPUT_FILE, 'r') as f:
    for i, line in enumerate(f):
        result = process_line(line)
        if result:
            video_path, label = result
            metadata[video_path] = {"label": label}
        
        # Optional: Show progress every 100 lines
        if i % 100 == 0:
            print(f"Processed {i} lines...")

# Save to JSON
with open(OUTPUT_FILE, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Success! Generated {OUTPUT_FILE} with {len(metadata)} entries")
