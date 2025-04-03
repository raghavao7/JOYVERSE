import { useState, useRef, useCallback } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "./App.css";

const questions = [
  { q: "What color is the sky?", options: ["Blue", "Red", "Green"], answer: "Blue" },
  { q: "What animal says 'meow'?", options: ["Dog", "Cat", "Bird"], answer: "Cat" },
  { q: "What comes after Tuesday?", options: ["Monday", "Wednesday", "Friday"], answer: "Wednesday" },
  { q: "Which is a fruit?", options: ["Apple", "Car", "Book"], answer: "Apple" },
];

const themeGifs = {
  neutral: "https://media1.tenor.com/m/vbvqOPixsQ0AAAAC/dap.gif",
  happy: "https://media1.tenor.com/m/vbvqOPixsQ0AAAAC/dap.gif",
  sad: "https://media1.tenor.com/m/Q5kN2F2e8pAAAAAC/doraemon-cry.gif",
  angry: "https://media1.tenor.com/m/_7iV64VvkaMAAAAC/doraemon-angry.gif",
  surprised: "https://media1.tenor.com/m/5tL8Xz9vD8QAAAAC/doraemon-shocked.gif",
  confused: "https://media1.tenor.com/m/44CbaYfUL98AAAAd/bird-cartoon.gif",
  bored: "https://media1.tenor.com/m/7Y5z5u5z5zQAAAAC/doraemon-bored.gif",
};

function App() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [theme, setTheme] = useState("neutral");
  const [emotion, setEmotion] = useState("N/A");
  const [action, setAction] = useState("N/A");
  const [feedback, setFeedback] = useState("");
  const [quizStarted, setQuizStarted] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);
  const [quizData, setQuizData] = useState([]); // Store all quiz data
  const webcamRef = useRef(null);

  const captureFrame = useCallback(() => {
    return webcamRef.current.getScreenshot();
  }, [webcamRef]);

  const captureMultipleFrames = useCallback((callback) => {
    let frames = [];
    let captures = 0;
    const interval = setInterval(() => {
      frames.push(captureFrame());
      captures++;
      if (captures === 4) {
        clearInterval(interval);
        callback(frames);
      }
    }, 1000);
  }, [captureFrame]);

  const analyzeEmotions = async (frames) => {
    const formData = new FormData();
    frames.forEach((frame, index) => {
      formData.append(`frame${index}`, dataURLtoBlob(frame));
    });
    formData.append("previous_actions", `${theme},question${currentQuestion}`);

    try {
      const response = await axios.post("http://127.0.0.1:5000/analyze-frames", formData);
      const { emotion, emotion_probs, action } = response.data;
      setEmotion(emotion || "N/A");
      setAction(action || "N/A");
      adjustQuiz(action, emotion);

      // Add current question data to quizData array
      setQuizData((prev) => [
        ...prev,
        {
          question: questions[currentQuestion].q,
          emotions: emotion_probs,
          action,
          theme,
        },
      ]);

      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1);
      } else {
        setQuizCompleted(true);
      }
    } catch (error) {
      console.error("Fetch Error:", error);
    }
  };

  const dataURLtoBlob = (dataURL) => {
    const [header, data] = dataURL.split(",");
    const mime = header.match(/:(.*?);/)[1];
    const binary = atob(data);
    const array = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
    return new Blob([array], { type: mime });
  };

  const adjustQuiz = (action, emotion) => {
    const emotionToTheme = {
      happy: "happy",
      sad: "sad",
      angry: "angry",
      surprised: "surprised",
      neutral: "neutral",
      confused: "confused",
      bored: "bored",
    };
    const newTheme = emotionToTheme[emotion] || "neutral";
    setTheme(newTheme);
    console.log(`Emotion: ${emotion}, Action: ${action}, Theme: ${newTheme}`);
  };

  const checkAnswer = (selected, correct) => {
    if (selected === correct) {
      setFeedback("Great job! Analyzing your emotions...");
      captureMultipleFrames((frames) => {
        setFeedback("Great job!");
        analyzeEmotions(frames);
      });
    } else {
      setFeedback("Try again!");
    }
  };

  const startQuiz = () => {
    setCurrentQuestion(0);
    setTheme("neutral");
    setFeedback("");
    setEmotion("N/A");
    setAction("N/A");
    setQuizStarted(true);
    setQuizCompleted(false);
    setQuizData([]); // Reset quiz data for new session
  };

  const saveQuizSession = async () => {
    const playerId = `player_${Date.now()}`; // Simple unique ID based on timestamp
    const sessionData = { playerId, quizData };
    try {
      await axios.post("http://localhost:3000/api/save-quiz-session", sessionData);
      console.log("Quiz session saved:", sessionData);
    } catch (error) {
      console.error("Error saving quiz session:", error);
    }
  };

  return (
    <div className={`theme-${theme}`}>
      <h2>JoyVerse: Adaptive Quiz for Dyslexic Kids</h2>
      <Webcam audio={false} ref={webcamRef} screenshotFormat="image/jpeg" width={320} height={240} />
      {!quizStarted && <button onClick={startQuiz}>Start Quiz</button>}

      {quizStarted && !quizCompleted && (
        <>
          <h3>Current Question:</h3>
          <p id="question">Question {currentQuestion + 1}: {questions[currentQuestion].q}</p>
          <div id="options">
            {questions[currentQuestion].options.map((opt) => (
              <button
                key={opt}
                className="option-btn"
                onClick={() => checkAnswer(opt, questions[currentQuestion].answer)}
              >
                {opt}
              </button>
            ))}
          </div>
          <p id="feedback">{feedback}</p>
          <div id="gif-display">
            <img id="theme-gif" src={themeGifs[theme]} alt="Theme GIF" />
          </div>
        </>
      )}

      {quizCompleted && (
        <>
          <h3>Quiz Complete!</h3>
          <p>Well done!</p>
          <button onClick={() => { saveQuizSession(); window.location.href = "/admin"; }}>
            Get Report
          </button>
        </>
      )}

      <h3>Analysis:</h3>
      <p><b>Detected Emotion:</b> {emotion}</p>
      <p><b>Next Action:</b> {action}</p>
      <p><b>Current Theme:</b> {theme.charAt(0).toUpperCase() + theme.slice(1)}</p>
    </div>
  );
}

export default App;