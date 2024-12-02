import cv2
import speech_recognition as sr
import threading

# Initialize face detection with OpenCV's Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to display live video feed with face detection
def detect_faces_and_display_captions(caption_text):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for better face detection performance
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Display the real-time caption
        cv2.putText(frame, caption_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Face Detection with Live Captions', frame)
        
        # Stop on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to recognize speech and update captions in real-time
def recognize_speech_and_update_captions():
    recognizer = sr.Recognizer()
    global caption_text
    
    while True:
        with sr.Microphone() as source:
            try:
                print("Listening for speech...")
                audio = recognizer.listen(source, timeout=5)
                speech_text = recognizer.recognize_google(audio)
                print(f"Recognized: {speech_text}")
                caption_text = speech_text
            except sr.UnknownValueError:
                print("Could not understand audio.")
                caption_text = "Listening..."
            except sr.WaitTimeoutError:
                print("No speech detected.")
            except sr.RequestError as e:
                print(f"Speech Recognition service error: {e}")
                caption_text = "Error with Speech Recognition service."

if __name__ == "__main__":
    # Global caption text variable shared across threads
    caption_text = "Listening..."
    
    # Create threads for face detection and speech recognition
    face_thread = threading.Thread(target=detect_faces_and_display_captions, args=(lambda: caption_text,)())
    speech_thread = threading.Thread(target=recognize_speech_and_update_captions)
    
    # Start threads
    face_thread.start()
    speech_thread.start()
    
    # Wait for both threads to finish
    face_thread.join()
    speech_thread.join()
