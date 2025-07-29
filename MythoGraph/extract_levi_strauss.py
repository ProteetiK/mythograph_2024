import os
import csv

folder = "LeviStrauss_Gold_Text"
output_csv = "LeviStrauss_Gold_Text.csv"

file_paths = []
for root, _, files in os.walk(folder):
    for file in files:
        if file.endswith(".txt"):
            full_path = os.path.join(root, file)
            try:
                mtime = os.path.getmtime(full_path)
                file_paths.append((full_path, mtime))
            except Exception as e:
                print(f"Error accessing {full_path}: {e}")

file_paths.sort(key=lambda x: x[1])

data = []
for file_path, _ in file_paths:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                data.append({
                    "Title": os.path.basename(file_path),
                    "Content": content
                })
            else:
                print(f"Empty content: {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if data:
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Title", "Content"])
        writer.writeheader()
        writer.writerows(data)
    print(f"CSV written to {output_csv} with {len(data)} files (sorted by last modified time).")
else:
    print("No data to write.")