import os
import sys

def convert_scenes_to_video(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each scene directory
    for scene_name in os.listdir(input_dir):
        scene_path = os.path.join(input_dir, scene_name)
        if not os.path.isdir(scene_path):
            continue

        # Output scene folder
        output_scene_dir = os.path.join(output_dir, scene_name)
        os.makedirs(output_scene_dir, exist_ok=True)

        # Input file pattern and output video path
        input_pattern = os.path.join(scene_path, "dir_%d_mip2.png")
        output_video = os.path.join(output_scene_dir, f"{scene_name}_ground_truth.mp4")

        # Run ffmpeg using os.system with framerate set to 10
        command = f'ffmpeg -y -framerate 10 -i "{input_pattern}" -c:v libx264 -pix_fmt yuv420p "{output_video}"'
        print(f"Processing scene: {scene_name}")
        os.system(command)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_scenes.py <input_dir> <output_dir>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    convert_scenes_to_video(input_directory, output_directory)
