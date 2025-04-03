

require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const cors = require('cors');
const bodyParser = require('body-parser');
const { Server } = require('socket.io');
const http = require('http');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

app.use(cors());
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('✅ MongoDB connected'))
    .catch(err => console.log('❌ MongoDB connection error:', err));

// Define Schema & Model
const ChildSchema = new mongoose.Schema({
    name: String,
    phone: String,
    userId: String,
    password: String,
    registeredAt: { type: Date, default: Date.now }
});
const Child = mongoose.model('Child', ChildSchema);

// 📌 **Register Child (Admin Panel)**
app.post('/register', async (req, res) => {
    try {
        console.log("📩 Received Registration Data:", req.body);

        const { name, phone, userId, password } = req.body;
        if (!name || !phone || !userId || !password) {
            console.log("⚠️ Missing fields");
            return res.status(400).json({ message: 'All fields are required' });
        }

        // Check if the phone number is already registered
        const existingChild = await Child.findOne({ phone });
        if (existingChild) {
            return res.status(400).json({ message: '❌ Child already registered' });
        }

        const hashedPassword = await bcrypt.hash(password, 10);
        const newChild = new Child({ name, phone, userId, password: hashedPassword });

        await newChild.save();

        console.log("✅ Child registered successfully");

        // Emit event for real-time update
        io.emit("newChild", newChild);

        res.json({ message: '✅ Child registered successfully' });

    } catch (error) {
        console.error("❌ Error in /register:", error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

// 📌 **Authenticate Child (Login)**
app.post('/login', async (req, res) => {
    const { name, userId, password } = req.body;
    console.log("📩 Received Login Data:", req.body); // 🔥 Log input

    if (!name || !userId || !password) {
        console.log("⚠️ Missing fields");
        return res.status(400).json({ message: '⚠️ All fields are required' });
    }

    try {
        console.log(`🔍 Searching for: Name = ${name}, User ID = ${userId}`);

        // Find child in database (case-insensitive)
        const child = await Child.findOne({ 
            name: { $regex: new RegExp(`^${name}$`, "i") }, 
            userId: userId.toString()
        });

        if (!child) {
            console.log("❌ No matching child found!");
            return res.status(401).json({ message: 'Invalid name or user ID' });
        }

        console.log("✅ Found Child:", child);

        // Compare password
        const isMatch = await bcrypt.compare(password, child.password);
        console.log("🔑 Password Match:", isMatch);

        if (!isMatch) {
            return res.status(401).json({ message: 'Invalid password' });
        }

        // Generate JWT token
        const token = jwt.sign(
            { userId: child.userId, name: child.name },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );

        console.log("✅ Login successful!");
        res.json({ message: '✅ Login successful', token });

    } catch (error) {
        console.error('❌ Login error:', error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});


// 📌 **Get Recently Registered Children**
app.get('/children', async (req, res) => {
    try {
        const children = await Child.find().sort({ registeredAt: -1 }).limit(10);
        res.json(children);
    } catch (error) {
        console.error("❌ Error fetching children:", error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

// 📌 **Real-Time Socket.io Connection**
io.on("connection", (socket) => {
    console.log("🟢 Client connected");

    socket.on("fetchChildren", async () => {
        const children = await Child.find().sort({ registeredAt: -1 }).limit(10);
        io.emit("updateChildren", children);
    });

    socket.on("disconnect", () => console.log("🔴 Client disconnected"));
});
const path = require('path');

// Serve Static Files
app.use(express.static('public'));

// Global Error Handler - Redirect to Error Page
app.use((err, req, res, next) => {
    console.error('Server Error:', err);
    res.status(500).sendFile(path.join(__dirname, 'public', 'error.html'));
});


// 📌 **Start Server**
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
