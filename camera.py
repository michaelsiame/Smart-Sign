import cv2

# Open the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break
    # Display the resulting frame
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
