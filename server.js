require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('MongoDB connected'))
    .catch(err => console.log(err));

// Define Schema & Model
const ChildSchema = new mongoose.Schema({
    name: String,
    phone: String,
    userId: String,
    password: String
});
const Child = mongoose.model('Child', ChildSchema);

// Register Child (Admin Panel)
app.post('/register', async (req, res) => {
    try {
        console.log("Received Data:", req.body); // Debugging Log

        const { name, phone, userId, password } = req.body;
        if (!name || !phone || !userId || !password) {
            console.log("Missing fields");
            return res.status(400).json({ message: 'All fields required' });
        }

        // Check if user already exists
        const existingChild = await Child.findOne({ phone });
        if (existingChild) return res.status(400).json({ message: 'Child already registered' });

        const hashedPassword = await bcrypt.hash(password, 10);
        const child = new Child({ name, phone, userId, password: hashedPassword });
        
        await child.save();
        console.log("Child registered successfully");
        res.json({ message: 'Child registered successfully' });

    } catch (error) {
        console.error("Error in /register:", error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

// Authenticate Child (Login)
app.post('/login', async (req, res) => {
    const { name, userId } = req.body;

    if (!name || !userId) {
        return res.status(400).json({ message: 'All fields required' });
    }

    try {
        console.log(`Searching for: Name = ${name}, User ID = ${userId}`);

        // Use case-insensitive search and convert userId to string
        const child = await Child.findOne({ 
            name: { $regex: new RegExp(`^${name}$`, "i") }, 
            userId: userId.toString() 
        });

        if (!child) {
            console.log("No matching child found in DB!");
            return res.status(401).json({ message: 'Invalid name or user ID' });
        }

        // Generate JWT token
        const token = jwt.sign(
            { userId: child.userId, name: child.name },
            process.env.JWT_SECRET,
            { expiresIn: '1h' }
        );

        console.log("Login successful!");
        res.json({ message: 'Login successful', token });

    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});



// Start Server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
