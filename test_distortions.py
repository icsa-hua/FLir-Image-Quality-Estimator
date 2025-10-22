from collections import deque
import ctypes
import cv2
import torch
from distortions import *
from fliqe import OnlineFLIQE


if __name__ == "__main__":
    fliqe = OnlineFLIQE(smoothing_window=150)
    # Get screen size (Windows)
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    print(f"Screen size: {screen_width}x{screen_height}")

    cap = cv2.VideoCapture("./data/1.MP4")
    ret, frame = cap.read()
    print(f"Video frame size: {frame.shape[1]}x{frame.shape[0]}")
    # Use a larger size, e.g., 80% of original
    scale_factor = 0.8
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    print(f"Resized video frame size: {frame_width}x{frame_height}, FPS: {fps}")
    small_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    grid_size = 3
    out = cv2.VideoWriter("distortions.mp4", fourcc, fps, (frame_width * grid_size, frame_height * grid_size))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    distortions = {
        "Original": {'func':lambda img: img},
        "Lens Blur": {'func':lambda img: LensBlur(ksize=11)(img)[0]},
        "Motion Blur": {'func':lambda img: MotionBlur(degree=15, angle=45)(img)[0]},
        "Overexposure": {'func':lambda img: Overexposure(factor=2.5)(img)[0]},
        "Underexposure": {'func':lambda img: Underexposure(factor=0.3)(img)[0]},
        "Compression": {'func':lambda img: Compression(quality=5)(img)[0]},
        "Ghosting": {'func':lambda img: Ghosting(shift=10, alpha=0.6)(img)[0]},
        # "Blackout": {'func':lambda img: Blackout()(img)[0]},
        "Noise": {'func':lambda img: GaussianNoise(mean=0, std=25)(img)[0]},
        # "Color Distortion": lambda img: ColorDistortion()(img)[0],
        # "Glare": lambda img: Glare()(img)[0],
        # "Flicker": lambda img: Flicker(factor=1.8)(img)[0],
        # "Freeze": lambda img: FrameFreeze()(img)[0],
        # "Obstruction": lambda img: Obstruction()(img)[0],
        # "Crop": {'func':lambda img: Crop()(img)[0]}
        "Aliasing": {'func':lambda img: Aliasing(factor=4)(img)[0]}
    }
    for k in distortions.keys():
        fliqe.create_session(k)

    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1
        # if processed_frames < 150:
        #     continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing frame {processed_frames}/{total_frames}", end='\r')
        frame_small = cv2.resize(frame, small_size)
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        # Apply distortions and annotate
        annotated_versions = []
        for dist_name, dist in distortions.items():
            img = dist['func'](frame_small)
            annotated = img.copy()
            img = img.astype(np.float32)

            # FLIQE score
            quality_score = fliqe.estimate_smoothed_quality(dist['func'](frame), session_id=dist_name)

            color = fliqe.get_color(fliqe.get_smoothed_quality(dist_name))
            # Draw black rectangles as background for text
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 10), (350, 100), (0, 0, 0), -1)
            alpha = 0.5  # Lower opacity (0.0 transparent, 1.0 opaque)
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            cv2.putText(annotated, dist_name, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"FLIQE: ", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"{fliqe.get_smoothed_quality(dist_name):.2f}", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            annotated_versions.append(annotated)


        # Arrange in a nxm grid (nxm cells, last one black)
        cell_h, cell_w = annotated_versions[0].shape[:2]
        grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                idx = i * grid_size + j
                if idx < len(annotated_versions):
                    row.append(annotated_versions[idx])
                else:
                    # Fill empty cell with black
                    row.append(np.zeros_like(annotated_versions[0]))
            grid.append(np.hstack(row))
        combined_frame = np.vstack(grid)
        out.write(combined_frame)
        
        # Maintain aspect ratio while fitting to screen
        ch, cw = combined_frame.shape[:2]
        scale = min(screen_width / cw, screen_height / ch)
        new_w, new_h = int(cw * scale), int(ch * scale)
        resized = cv2.resize(combined_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to center
        top = (screen_height - new_h) // 2
        bottom = screen_height - new_h - top
        left = (screen_width - new_w) // 2
        right = screen_width - new_w - left
        combined_frame_padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        cv2.imshow("Distorted Frames", combined_frame_padded)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("distorted_frames.png", combined_frame_padded)  # Uncomment to save the image
            break
        
        # if processed_frames == 600:
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()