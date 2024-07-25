import cv2
import os
import argparse

def create_timelapse(input_dir, output_file, fps, zoom):
    # Get list of files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    # Sort files by their numerical value in the filename
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
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
        
        # Apply digital zoom
        if zoom != 1.0:
            img = apply_zoom(img, zoom)
        
        out.write(img)

    out.release()
    print("Timelapse video saved as {}".format(output_file))

def apply_zoom(img, zoom):
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = int(width // (2 * zoom)), int(height // (2 * zoom))

    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y

    cropped = img[min_y:max_y, min_x:max_x]
    resized = cv2.resize(cropped, (width, height))
    return resized

def get_next_run_number(base_dir, prefix="timelapse"):
    files = [f for f in os.listdir(base_dir) if f.startswith(prefix) and f.endswith('.mp4')]
    if not files:
        return 1
    run_numbers = [int(f[len(prefix) + 1:-4]) for f in files if f[len(prefix) + 1:-4].isdigit()]
    if not run_numbers:
        return 1
    return max(run_numbers) + 1

def parse_args():
    parser = argparse.ArgumentParser(description="Create a timelapse video from PNG images.")
    parser.add_argument('input_dir', type=str, help='Directory containing PNG images.')
    parser.add_argument('output_dir', type=str, nargs='?', default=None, help='Output directory path (optional).')
    parser.add_argument('-f', '--fps', type=int, default=60, help='Frames per second for the output video (default: 60).')
    parser.add_argument('-z', '--zoom', type=float, default=1.0, help='Digital zoom factor for the images (default: 1.0).')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'timelapse_output')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    run_number = get_next_run_number(output_dir)
    output_file = os.path.join(output_dir, f'timelapse_{run_number}.mp4')

    create_timelapse(args.input_dir, output_file, args.fps, args.zoom)
