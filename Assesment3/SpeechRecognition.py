import speech_recognition as sr
import threading

class SpeechRecognition:
    def __init__(self):
        # Initialize the recognizer and microphone functions from sr
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def recognize_speech(self):
        # Listen to the microphone for speech and prints recognized test
        # (Ref. Speech Recognition examples with Python - Python. (n.d.). https://pythonprogramminglanguage.com/speech-recognition/)
        with self.microphone as source:
            # Adjust the recognizer sensitivity to ambient noise levels
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening for speech...")

            while True:
                try:
                    # Listen for speech input from the microphone and variables for timeout and maximum time of listen
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)
                    # Recognize speech using Google speech recognition API
                    speech_text = self.recognizer.recognize_google(audio)
                    # Print the recognized text
                    if speech_text.strip():
                        print(f"Recognized: {speech_text}")
                # No speech detected, continue listening
                except sr.WaitTimeoutError:
                    pass 
                # Speech unintelligible, continue listening
                except sr.UnknownValueError:
                    print("Apologies, the audio wasn't clear enough.")
                    pass  
                except sr.RequestError as e:
                    print("There was an issue retrieving results. Error: {0}".format(e))
                    break  # Exit the loop on service error

    def start_recognition(self):
        """ We need to separate the recognition in another thread, in this way it runs by itself and is not 
        blocked by the main program, so basically it can continue listening while trying to recognize other speech"""
        recognition_thread = threading.Thread(target=self.recognize_speech)
        # Ensure the thread exits when the main program exits
        recognition_thread.daemon = True  
        recognition_thread.start()
        # Return the thread instance
        return recognition_thread  

# Validation function for speech recognition (word % comparison)
def validate_speech_recognition(validation_set):
    recognizer = sr.Recognizer()
    for audio_file, correct_transcription in validation_set.items():
        try:
            # Load the audio file from sr
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
                
            # Recognize speech using Google API
            recognized_text = recognizer.recognize_google(audio)
            # Try to normalize text
            recognized_words = recognized_text.lower().split()
            correct_words = correct_transcription.lower().split()

            # Initialize the count of correct words
            correct_count = 0

            # Iterate over pairs of recognized and correct words
            for rec_word, corr_word in zip(recognized_words, correct_words):
                # we compare the words and if match increase count
                if rec_word == corr_word:
                    correct_count += 1

            # Handle cases where the number of words in recognized text and correct transcription do not match
            total_words = max(len(recognized_words), len(correct_words))

            # Calculate accuracy percentage
            if total_words > 0:
                accuracy_percentage = (correct_count / total_words) * 100
            else:
                accuracy_percentage = 0

            # Print Results
            print(f"{audio_file}: RECOGNIZED TEXT: {recognized_text}")
            print(f"CORRECT TEXT: {correct_transcription}")
            print(f"Accuracy: {accuracy_percentage:.2f}% ({correct_count}/{total_words} correct words)")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

if __name__ == "__main__":
    # Initialize the speech recognition system
    speech_recognition = SpeechRecognition()

    # Start the recognition thread
    recognition_thread = speech_recognition.start_recognition()

    # Validation set from kaggle 
    #(ref. sample audio files for speech recognition. (2020, August 14). https://www.kaggle.com/datasets/pavanelisetty/sample-audio-files-for-speech-recognition/data?select=jackhammer.wav)
    validation_set = {
        r"C:\Users\skill\source\repos\AIApplications\Assesment3\known_speech\harvard.wav": "the stale smell of old beer lingers it takes heat to bring out the o'dor a cold dip restores health and zest a salt pickle tastes fine with ham tacos al pastor are my favorite a zestful food is the hot cross bun.",
        r"C:\Users\skill\source\repos\AIApplications\Assesment3\known_speech\jackhammer.wav": "the stale smell of old beer lingers",
    }
    
    # Run validation after starting recognition
    print("Starting validation...")
    validate_speech_recognition(validation_set)

    # Keep the main program running while speech recognition happens in the background
    try:
        while True:
            pass  # Keep the main program running
    except KeyboardInterrupt:
        print("Program interrupted by user.")
