import cv2
import speech_recognition as sr
import threading

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

caption_text = "Listening..."
new_caption = threading.Event()
last_text = ""

# Function to display fading text for smoother transitions
def draw_fading_text(frame, text, position, alpha=0.6):
    overlay = frame.copy()
    cv2.putText(overlay, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# Function to handle face detection and display captions
def detect_faces_and_display_captions():
    global caption_text, last_text
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the latest recognized caption smoothly
        if caption_text != last_text:
            frame = draw_fading_text(frame, caption_text, (10, frame.shape[0] - 20))
            last_text = caption_text
        else:
            frame = draw_fading_text(frame, caption_text, (10, frame.shape[0] - 20), alpha=0.3)
        
        # Show the frame with detected faces and captions
        cv2.imshow('Face Detection with Real-Time Captions', frame)
        
        # Break on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function for continuous speech recognition
def recognize_speech_and_update_captions():
    global caption_text
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    # Continuous listening loop
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for speech...")

        while True:
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                speech_text = recognizer.recognize_google(audio)
                
                # Avoid repeating the same caption
                if speech_text.strip() and speech_text != caption_text:
                    caption_text = speech_text
                    print(f"Recognized: {speech_text}")
                
                new_caption.set()
            except sr.WaitTimeoutError:
                pass  # No speech detected, continue listening
            except sr.UnknownValueError:
                caption_text = "Listening..."
                new_caption.clear()
            except sr.RequestError as e:
                caption_text = "Speech Service Error"
                print(f"Speech Recognition service error: {e}")
                new_caption.clear()

# Main logic for threading
if __name__ == "__main__":
    # Create and start threads
    face_thread = threading.Thread(target=detect_faces_and_display_captions)
    speech_thread = threading.Thread(target=recognize_speech_and_update_captions)
    
    face_thread.start()
    speech_thread.start()
    
    # Wait for threads to complete
    face_thread.join()
    speech_thread.join()
