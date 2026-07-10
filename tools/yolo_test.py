import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the core YOLO26 segmentation model
model = YOLO("yolo26m-seg.pt")

# 2. Initialize the live webcam stream
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error opening webcam"

print("Press 'q' to quit the live feed.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Webcam frame empty or disconnected.")
        break

    # 3. Run inference strictly filtering for class 0 (Person)
    results = model(frame, classes=[0], verbose=False)
    
    # Create an entirely black background image matching the frame size
    black_background = np.zeros_like(frame)
    
    # Check if any humans were detected and if masks exist
    if results[0].masks is not None:
        # Combine all detected human masks into a single binary mask
        # We scale it to the original frame size
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for mask in results[0].masks.data:
            # Move mask to CPU, convert to numpy, and resize to match frame dimensions
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            
            # Merge with the combined mask
            combined_mask = cv2.bitwise_or(combined_mask, mask_resized.astype(np.uint8))
            
        # 4. Copy the human pixels onto the black background using the mask
        # Where combined_mask == 1, keep original frame; everywhere else stays black.
        cv2.copyTo(frame, combined_mask, black_background)
        
        # Optional: Render the default bounding boxes/labels on top of the isolated segment
        # annotated_frame = results[0].plot(img=black_background)
        # cv2.imshow("YOLO26 Human Isolation", annotated_frame)
    else:
        # If no humans are detected, the output remains completely black
        pass

    # 5. Display the isolated result
    cv2.imshow("YOLO26 Human Isolation", black_background)

    # Break the loop gracefully if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()