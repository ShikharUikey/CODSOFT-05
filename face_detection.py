import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk


# -----------------------------
# Face Cascade
# -----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# -----------------------------
# Window
# -----------------------------

root = tk.Tk()

root.title("Face Detection AI")
root.geometry("1100x720")
root.configure(bg="#1e1e1e")


# -----------------------------
# Global Variables
# -----------------------------

current_image    = None
current_cv_image = None
photo            = None
webcam_running   = False   # BUG FIX : was missing — needed to stop webcam loop cleanly
camera           = None    # BUG FIX : global camera ref so we can release it on Stop


# -----------------------------
# Detect Faces
# -----------------------------

def detect_faces(image):
    """
    BUG FIX : minNeighbors raised from 4 → 6  (was causing false positives on
               hands / body parts as seen in the screenshot)
    BUG FIX : minSize raised from (30,30) → (60,60) so tiny non-face blobs
               are ignored (group photo has a lot of those)
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # optional: equalise histogram so darker faces are detected better
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,      # BUG FIX : was 4 → too many false positives
        minSize=(60, 60)     # BUG FIX : was (30,30) → detected hands/objects
    )

    result = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    count_label.config(
        text=f"Faces Detected : {len(faces)}"
    )

    status_label.config(
        text="Status : Detection Complete"
    )

    return result


# -----------------------------
# Show Image
# -----------------------------

def show_image(image):

    global photo

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(rgb)
    img.thumbnail((760, 560))

    photo = ImageTk.PhotoImage(img)

    image_label.config(
        image=photo,
        text=""
    )

    image_label.image = photo


# -----------------------------
# Upload Image
# -----------------------------

def upload_image():

    global current_image
    global current_cv_image

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.webp")
        ]
    )

    if file_path == "":
        return

    image = cv2.imread(file_path)

    if image is None:
        messagebox.showerror(
            "Error",
            "Unable to open image."
        )
        return

    current_image    = image.copy()
    current_cv_image = detect_faces(image)

    show_image(current_cv_image)


# -----------------------------
# Webcam  (BUG FIX : was using a blocking while-loop which froze the UI.
#           Now uses root.after() so tkinter stays responsive.)
# -----------------------------

def open_webcam():

    global webcam_running
    global camera
    global current_cv_image

    if webcam_running:          # already running — ignore double-click
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        messagebox.showerror(
            "Error",
            "Unable to open webcam."
        )
        return

    webcam_running = True

    # swap button text so user can stop it
    camera_button.config(
        text="⏹ Stop Webcam",
        command=stop_webcam,
        bg="#DC2626"
    )

    status_label.config(text="Status : Webcam Running")

    _webcam_loop()              # kick off the non-blocking loop


def _webcam_loop():
    """Called repeatedly via root.after — does NOT block the UI."""

    global current_cv_image

    if not webcam_running:
        return

    success, frame = camera.read()

    if success:
        current_cv_image = frame.copy()

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,     # BUG FIX : same fix as detect_faces()
            minSize=(60, 60)    # BUG FIX : same fix as detect_faces()
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                current_cv_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

        count_label.config(
            text=f"Faces Detected : {len(faces)}"
        )

        show_image(current_cv_image)

    # schedule next frame (~30 fps)
    root.after(33, _webcam_loop)


def stop_webcam():
    """Stop the webcam loop and release the camera."""

    global webcam_running
    global camera

    webcam_running = False

    if camera is not None:
        camera.release()
        camera = None

    cv2.destroyAllWindows()

    # restore button
    camera_button.config(
        text="📷 Open Webcam",
        command=open_webcam,
        bg="#16A34A"
    )

    status_label.config(text="Status : Ready")

    count_label.config(text="Faces Detected : 0")


# -----------------------------
# Save Image
# -----------------------------

def save_image():

    global current_cv_image

    if current_cv_image is None:
        messagebox.showwarning(
            "Warning",
            "No image available to save."
        )
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[
            ("JPEG Image", "*.jpg"),
            ("PNG Image",  "*.png")
        ]
    )

    if file_path == "":
        return

    cv2.imwrite(file_path, current_cv_image)

    messagebox.showinfo(
        "Saved",
        "Image saved successfully!"
    )


# -----------------------------
# Clear Image
# BUG FIX : function was accidentally indented INSIDE save_image — it was
#            never reachable as a standalone function before.
# -----------------------------

def clear_image():

    global current_image
    global current_cv_image
    global photo

    current_image    = None
    current_cv_image = None
    photo            = None

    image_label.config(
        image="",
        text="Image Preview"
    )

    image_label.image = None

    count_label.config(text="Faces Detected : 0")

    status_label.config(text="Status : Ready")


# -----------------------------
# Title
# -----------------------------

title = tk.Label(
    root,
    text="🧠 FACE DETECTION AI",
    bg="#1e1e1e",
    fg="#00C2FF",
    font=("Helvetica", 28, "bold")
)

title.pack(pady=(20, 5))

subtitle = tk.Label(
    root,
    text="AI Powered Face Detection using OpenCV",
    bg="#1e1e1e",
    fg="#bbbbbb",
    font=("Helvetica", 12)
)

subtitle.pack()


# -----------------------------
# Main Layout
# -----------------------------

main_frame = tk.Frame(root, bg="#1e1e1e")

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=20
)


# -----------------------------
# Left Panel
# -----------------------------

left_frame = tk.Frame(
    main_frame,
    bg="#252526",
    bd=2,
    relief="ridge"
)

left_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10
)


# -----------------------------
# Right Panel
# -----------------------------

right_frame = tk.Frame(
    main_frame,
    bg="#252526",
    bd=2,
    relief="ridge",
    width=260
)

right_frame.pack(
    side="right",
    fill="y",
    padx=10
)

right_frame.pack_propagate(False)


# -----------------------------
# Image Preview
# -----------------------------

image_label = tk.Label(
    left_frame,
    text="Image Preview",
    bg="#252526",
    fg="white",
    font=("Helvetica", 14)
)

image_label.pack(expand=True)


# -----------------------------
# Status Labels
# -----------------------------

status_label = tk.Label(
    right_frame,
    text="Status : Ready",
    bg="#252526",
    fg="#00ff99",
    font=("Helvetica", 12, "bold")
)

status_label.pack(pady=20)

count_label = tk.Label(
    right_frame,
    text="Faces Detected : 0",
    bg="#252526",
    fg="white",
    font=("Helvetica", 13)
)

count_label.pack(pady=10)


# -----------------------------
# Upload Button
# -----------------------------

upload_button = tk.Button(
    right_frame,
    text="📁 Upload Image",
    command=upload_image,
    width=20,
    height=2,
    bg="#0078D4",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    cursor="hand2"
)

upload_button.pack(pady=15)


# -----------------------------
# Webcam Button
# -----------------------------

camera_button = tk.Button(
    right_frame,
    text="📷 Open Webcam",
    command=open_webcam,
    width=20,
    height=2,
    bg="#16A34A",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    cursor="hand2"
)

camera_button.pack(pady=15)


# -----------------------------
# Save Button
# -----------------------------

save_button = tk.Button(
    right_frame,
    text="💾 Save Image",
    command=save_image,
    width=20,
    height=2,
    bg="#E67E22",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    cursor="hand2"
)

save_button.pack(pady=15)


# -----------------------------
# Clear Button
# -----------------------------

clear_button = tk.Button(
    right_frame,
    text="🗑 Clear",
    command=clear_image,
    width=20,
    height=2,
    bg="#DC2626",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    cursor="hand2"
)

clear_button.pack(pady=15)


# -----------------------------
# Run Application
# -----------------------------

root.mainloop()