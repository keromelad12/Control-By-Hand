import cv2
import mediapipe as mp
import pyautogui
import time

#  MEDIAPIPE 
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path='hand_landmarker.task'
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

#  CAMERA 
cam = cv2.VideoCapture(0)
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

#  ZONES 
TOP_LIMIT = 0.40
BOTTOM_LIMIT = 0.60

#  SMOOTHING 
smooth_y = None
alpha = 0.85

#  STATE 
state = "middle"

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ✅ Convert BGR → RGB before wrapping in mp.Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb  # ✅ Must be RGB, not BGR
    )

    timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp)

    # ─── DRAW ZONES ────────────────────────
    top_px = int(TOP_LIMIT * h)
    bottom_px = int(BOTTOM_LIMIT * h)
    cv2.line(frame, (0, top_px), (w, top_px), (0, 255, 0), 2)
    cv2.line(frame, (0, bottom_px), (w, bottom_px), (0, 255, 0), 2)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # INDEX FINGER TIP
        index_tip = hand[8]
        x = int(index_tip.x * w)
        y_norm = index_tip.y

        # SMOOTHING
        if smooth_y is None:
            smooth_y = y_norm
        else:
            smooth_y = alpha * smooth_y + (1 - alpha) * y_norm

        y = int(smooth_y * h)
        cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

        # SCROLL LOGIC
        if smooth_y < TOP_LIMIT:
            if state != "top":
                pyautogui.scroll(-200)
                print("Scroll DOWN")
                state = "top"
        elif smooth_y > BOTTOM_LIMIT:
            if state != "bottom":
                pyautogui.scroll(200)
                print("Scroll UP")
                state = "bottom"
        else:
            state = "middle"

    cv2.imshow("Gesture Scroll", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()