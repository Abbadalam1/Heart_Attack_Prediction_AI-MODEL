const express = require('express');
const { spawn } = require('child_process');
const app = express();
const port = 3000; // CHANGE HERE: Update port if needed (e.g., for deployment)

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post('/predict', (req, res) => {
    const inputData = {
        age: req.body.age,
        country: req.body.country,
        troponin: req.body.troponin,
        ecgHeartRate: req.body.ecgHeartRate
        // CHANGE HERE: Add more fields if your AI model requires additional inputs
    };

    const pythonProcess = spawn('python', ['predict.py', JSON.stringify(inputData)]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            const result = JSON.parse(output);
            const prediction = result.prediction;
            const plans = getTailoredPlans(prediction, inputData.age);
            res.json({ prediction, plans });
        } else {
            res.status(500).json({ error: 'Prediction failed' });
        }
    });
});

function getTailoredPlans(prediction, age) {
    // Age-specific contextualization
    let diet, exercise, yoga, stressManagement;
    if (parseInt(age) >= 80) {
        diet = "Low-sodium, heart-healthy diet with soft fruits, vegetables, and lean proteins.";
        exercise = "10-15 minutes of gentle walking daily, avoid strenuous activity.";
        yoga = "Chair yoga for mobility, 10 minutes daily.";
        stressManagement = "Guided meditation for 5 minutes daily.";
    } else {
        diet = prediction === 'High Risk' 
            ? "Low-sodium, heart-healthy diet with fruits, vegetables, and lean proteins."
            : "Balanced diet with whole grains and healthy fats.";
        exercise = prediction === 'High Risk' 
            ? "30 minutes of moderate walking 5 days a week."
            : "20 minutes of light exercise daily.";
        yoga = prediction === 'High Risk' 
            ? "Beginner yoga for stress relief, 15 minutes daily."
            : "Gentle stretching for flexibility.";
        stressManagement = prediction === 'High Risk' 
            ? "Meditation and deep breathing for 10 minutes daily."
            : "Mindfulness exercises for 5 minutes daily.";
    }

    return { diet, exercise, yoga, stressManagement };
}

app.listen(port, () => {
    console.log(`Server running on port ${port}`);  //node server.js
});

//some updation and enhancements are required