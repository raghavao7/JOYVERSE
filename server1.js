const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

// MongoDB Connection
mongoose.connect("mongodb://localhost:27017/joyverse", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Player Quiz Session Schema
const quizSessionSchema = new mongoose.Schema({
  playerId: { type: String, required: true }, // Unique identifier for the player (e.g., timestamp or UUID)
  quizData: [
    {
      question: String,
      emotions: {
        happy: Number,
        sad: Number,
        angry: Number,
        surprised: Number,
        neutral: Number,
        confused: Number,
        bored: Number,
      },
      action: String,
      theme: String,
    },
  ],
  timestamp: { type: Date, default: Date.now },
});

const QuizSession = mongoose.model("QuizSession", quizSessionSchema);

// Save Quiz Session Data
app.post("/api/save-quiz-session", async (req, res) => {
  const { playerId, quizData } = req.body;
  try {
    const session = new QuizSession({ playerId, quizData });
    await session.save();
    res.status(201).json({ message: "Quiz session saved" });
  } catch (error) {
    res.status(500).json({ error: "Failed to save quiz session" });
  }
});

// Get All Quiz Sessions
app.get("/api/quiz-sessions", async (req, res) => {
  try {
    const sessions = await QuizSession.find().sort({ timestamp: -1 });
    res.json(sessions);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch quiz sessions" });
  }
});

app.listen(3000, () => {
  console.log("Express server running on port 3000");
});