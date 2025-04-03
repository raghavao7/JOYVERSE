import { useEffect, useState } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import "./Admin.css";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function Admin() {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await axios.get("http://localhost:3000/api/quiz-sessions");
        setSessions(response.data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching sessions:", error);
        setLoading(false);
      }
    };
    fetchSessions();
  }, []);

  const emotions = ["happy", "sad", "angry", "surprised", "neutral", "confused", "bored"];

  const getChartData = (quizData) => ({
    labels: quizData.map((entry) => entry.question),
    datasets: emotions.map((emotion, index) => ({
      label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      data: quizData.map((entry) =>
        entry.emotions && typeof entry.emotions[emotion] === "number"
          ? entry.emotions[emotion] * 100
          : 0
      ),
      backgroundColor: [
        "#FF6384",
        "#36A2EB",
        "#FFCE56",
        "#4BC0C0",
        "#9966FF",
        "#FF9F40",
        "#C9CBCF",
      ][index],
    })),
  });

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "Emotion Distribution per Question (%)" },
    },
    scales: {
      y: { beginAtZero: true, max: 100, title: { display: true, text: "Percentage (%)" } },
    },
  };

  return (
    <div className="admin">
      <h2>Admin Report: Player Quiz Sessions</h2>
      {loading ? (
        <p>Loading data...</p>
      ) : sessions.length === 0 ? (
        <p>No sessions available yet.</p>
      ) : (
        sessions.map((session) => (
          <div key={session._id} className="session">
            <h3>Player: {session.playerId}</h3>
            <p>Completed: {new Date(session.timestamp).toLocaleString()}</p>
            <div className="chart-container">
              <Bar data={getChartData(session.quizData)} options={options} />
            </div>
            <table>
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Dominant Emotion</th>
                  <th>Action</th>
                  <th>Theme (Background)</th>
                </tr>
              </thead>
              <tbody>
                {session.quizData.map((entry, index) => {
                  const dominantEmotion =
                    entry.emotions && Object.keys(entry.emotions).length > 0
                      ? Object.keys(entry.emotions).reduce((a, b) =>
                          entry.emotions[a] > entry.emotions[b] ? a : b
                        )
                      : "N/A";
                  return (
                    <tr key={index}>
                      <td>{entry.question}</td>
                      <td>{dominantEmotion}</td>
                      <td>{entry.action || "N/A"}</td>
                      <td>{entry.theme || "N/A"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))
      )}
      <button onClick={() => window.location.href = "/"}>Back to Quiz</button>
    </div>
  );
}

export default Admin;