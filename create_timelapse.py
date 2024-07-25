import cv2
import os
import argparse

def create_timelapse(input_dir, output_file, fps):
    # Get list of files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    files.sort()  # Ensure files are in order

    if not files:
        print("No PNG files found in the directory.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(input_dir, files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for filename in files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        out.write(img)

    out.release()
    print(f"Timelapse video saved as {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create a timelapse video from PNG images.")
    parser.add_argument('input_dir', type=str, help='Directory containing PNG images.')
    parser.add_argument('output_file', type=str, nargs='?', default=None, help='Output video file path (optional).')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for the output video (default: 60).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set default output file if not provided
    if args.output_file is None:
        output_dir = os.path.join(os.getcwd(), 'timelapse_output')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'timelapse.mp4')
    else:
        output_file = args.output_file

    create_timelapse(args.input_dir, output_file, args.fps)
