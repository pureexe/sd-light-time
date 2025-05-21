import json
from collections import Counter
import matplotlib.pyplot as plt

# Load JSON data
with open("/ist/ist-share/vision/pakkapon/relight/sd-light-time/src/20250510_webdataset_support/filenames_no_resample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Count how many times each string appears
string_counts = Counter(data)

# Count how many strings appear exactly N times
repeat_counts = Counter(string_counts.values())

# Sort by repeat count (1 time, 2 times, ...)
sorted_repeats = sorted(repeat_counts.items())

# Separate x and y values
x = [count for count, _ in sorted_repeats]
y = [num_strings for _, num_strings in sorted_repeats]

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.bar(x, y, color="skyblue")

# Add labels on top of bars
for bar, label in zip(bars, y):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(label), ha='center', va='bottom', fontsize=8)

plt.title("Frequency of String Repeats")
plt.xlabel("Number of Times a String Appears")
plt.ylabel("Number of Unique Strings")
plt.tight_layout()

# Save to PNG
plt.savefig("string_repeat_distribution.png", dpi=300)
plt.close()

print("Bar graph saved as string_repeat_distribution.png")
