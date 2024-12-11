import cv2
import numpy as np

def detect_color(hsv):
    # Define color ranges and corresponding names in HSV color space
    color_ranges = [
        ("Red", [(0, 50, 50), (10, 255, 255)], [(170, 50, 50), (180, 255, 255)]),
        ("Orange", [(11, 50, 50), (25, 255, 255)]),
        ("Yellow", [(26, 50, 50), (34, 255, 255)]),
        ("Green", [(35, 50, 50), (85, 255, 255)]),
        ("Blue", [(86, 50, 50), (125, 255, 255)]),
        ("Purple", [(126, 50, 50), (155, 255, 255)]),
        ("Pink", [(145, 50, 50), (165, 255, 255)]),  # Light red/pink range
        ("Cyan", [(80, 50, 50), (100, 255, 255)]),   # Greenish-blue range
        ("Brown", [(10, 50, 50), (20, 255, 100)]),   # Light brown range
        ("Beige", [(15, 10, 180), (25, 40, 255)]),   # Very light brown
        ("Lime", [(35, 100, 100), (75, 255, 255)]),  # Bright green
        ("White", [(0, 0, 200), (180, 30, 255)]),
        ("Gray", [(0, 0, 50), (180, 30, 200)]),
        ("Black", [(0, 0, 0), (180, 50, 50)]),
        ("Teal", [(80, 50, 50), (100, 255, 100)]),   # Blue-green
        ("Turquoise", [(85, 50, 50), (110, 255, 255)]), # Light cyan
        ("Violet", [(130, 50, 50), (160, 255, 255)]), # Dark purple
        ("Magenta", [(140, 50, 50), (160, 255, 255)]), # Pinkish purple
        ("Indigo", [(120, 50, 50), (140, 255, 255)]), # Deep blue
        ("Light Blue", [(100, 50, 50), (130, 255, 255)]), # Light blue
        ("Dark Green", [(35, 40, 40), (85, 255, 100)]), # Darker green
        ("Dark Blue", [(86, 40, 40), (125, 255, 100)]), # Darker blue
        ("Dark Red", [(0, 50, 50), (10, 100, 100)]), # Dark red
        ("Dark Orange", [(11, 50, 50), (25, 100, 100)]), # Dark orange
        ("Dark Yellow", [(26, 50, 50), (34, 100, 100)]), # Dark yellow
        ("Dark Purple", [(126, 40, 40), (155, 100, 100)]), # Dark purple
        ("Gold", [(25, 50, 50), (35, 255, 255)]), # Gold-yellow
        ("Silver", [(0, 0, 150), (180, 20, 200)]), # Light gray/silver
        ("Tan", [(20, 20, 100), (30, 100, 180)]), # Tan/brownish
        ("Maroon", [(0, 50, 50), (10, 100, 100)]), # Maroon/dark red
        ("Olive", [(35, 50, 50), (55, 150, 150)]), # Olive green
        ("Sea Green", [(50, 50, 50), (90, 255, 255)]), # Sea green
        ("Mint", [(60, 50, 50), (90, 255, 180)]), # Mint green
        ("Peach", [(5, 50, 50), (15, 255, 255)]), # Peachy pink
        ("Lavender", [(130, 30, 150), (160, 100, 255)]), # Light purple
        ("Copper", [(10, 50, 50), (20, 150, 150)]),
    ]

    for color_name, *ranges in color_ranges:
        for lower, upper in ranges:
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(np.uint8([[hsv]]), lower_bound, upper_bound)
            if mask[0][0] > 0:  # Check if the HSV value falls within the range
                return color_name
    return "Unknown"

def main():
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame and convert to HSV color space
        resized_frame = cv2.resize(frame, (640, 480))
        hsv_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)

        # Define region of interest (center of frame)
        height, width, _ = resized_frame.shape
        cx, cy = width // 2, height // 2
        roi = hsv_frame[cy - 10:cy + 10, cx - 10:cx + 10]

        # Average HSV color in ROI
        avg_hsv = np.mean(roi, axis=(0, 1))
        avg_hsv = np.round(avg_hsv).astype(int)

        # Debugging: Print average HSV value
        print(f"BY v1_creator ROI Avg HSV: {avg_hsv}")

        color = detect_color(avg_hsv)

        # Draw circle and display detected color
        cv2.circle(resized_frame, (cx, cy), 10, (255, 255, 255), 2)
        cv2.putText(resized_frame, f"Color: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("Color Detection", resized_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
