const express = require('express');
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.post('/predict', (req, res) => {
    const { age, gender, heartRate, systolicBloodPressure, diastolicBloodPressure, bloodSugar, ckMb, troponin, country } = req.body;

    // Validate inputs
    if (!age || !gender || !heartRate || !systolicBloodPressure || !diastolicBloodPressure || !bloodSugar || !ckMb || !troponin) {
        return res.status(400).json({ error: 'Missing required fields' });
    }

    const inputData = JSON.stringify({
        age,
        gender,
        heartRate,
        systolicBloodPressure,
        diastolicBloodPressure,
        bloodSugar,
        ckMb,
        troponin,
        country: country || 'India'
    });

    const pythonProcess = spawn('python', ['predict.py', inputData]);

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
    });

    pythonProcess.on('close', (code) => {
        if (res.headersSent) return; // Prevent sending multiple responses

        if (code === 0 && output) {
            try {
                const result = JSON.parse(output);
                res.json(result);
            } catch (err) {
                res.status(500).json({ error: 'Invalid response from Python script', details: err.message });
            }
        } else {
            res.status(500).json({ error: 'Prediction failed', details: errorOutput || 'Unknown error' });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);    //node server.js
});