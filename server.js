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
    const { name, phone, userId, password } = req.body;
    if (!name || !phone || !userId || !password) return res.status(400).json({ message: 'All fields required' });

    const hashedPassword = await bcrypt.hash(password, 10);
    const child = new Child({ name, phone, userId, password: hashedPassword });
    await child.save();

    res.json({ message: 'Child registered successfully' });
});

// Authenticate Child (Login)
app.post('/login', async (req, res) => {
    const { name, userId, password } = req.body;
    if (!name || !userId || !password) return res.status(400).json({ message: 'All fields required' });

    const child = await Child.findOne({ name, userId });
    if (!child || !(await bcrypt.compare(password, child.password))) {
        return res.status(401).json({ message: 'Invalid credentials' });
    }

    const token = jwt.sign({ userId: child.userId, name: child.name }, process.env.JWT_SECRET, { expiresIn: '1h' });
    res.json({ message: 'Login successful', token });
});

// Start Server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
