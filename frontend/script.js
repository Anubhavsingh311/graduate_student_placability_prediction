/*
  Change this URL if the frontend is hosted separately from the Flask backend.
  For local Flask testing use: http://127.0.0.1:5000
  For Render, put your deployed backend URL here.
*/
const API_BASE_URL = "https://graduate-student-placability-prediction.onrender.com";

const form        = document.getElementById("predictionForm");
const button      = document.getElementById("predictBtn");
const statusEl    = document.getElementById("status");
const resultEl    = document.getElementById("result");
const resultTitle = document.getElementById("resultTitle");
const resultText  = document.getElementById("resultText");
const resultIcon  = document.getElementById("resultIcon");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  // Reset state
  statusEl.textContent = "";
  statusEl.style.display = "none";
  resultEl.style.display = "none";
  resultEl.classList.remove("result--placed", "result--not-placed", "animate-reveal");
  resultIcon.classList.remove("result-icon--placed", "result-icon--not-placed", "animate-pop");

  const data = {
    gender:           document.getElementById("gender").value,
    "10th_percent":   Number(document.getElementById("10th_percent").value),
    "10th_board":     document.getElementById("10th_board").value.trim(),
    "12th_percent":   Number(document.getElementById("12th_percent").value),
    "12th_board":     document.getElementById("12th_board").value.trim(),
    "12th_stream":    document.getElementById("12th_stream").value,
    degree_percent:   Number(document.getElementById("degree_percent").value),
    degree:           document.getElementById("degree").value.trim(),
    experience:       document.getElementById("experience").value,
    emp_test_percent: Number(document.getElementById("emp_test_percent").value),
    mba_stream:       document.getElementById("mba_stream").value.trim(),
    mba_percent:      Number(document.getElementById("mba_percent").value),
  };

  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span>Predicting…';

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const body = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(body.error || `Server returned ${response.status}`);
    }

    // Expected backend response:
    // { "prediction": 1, "result": "Placed", "probability": 0.87 }
    const prediction = Number(body.prediction);
    const isPlaced =
      prediction === 1 ||
      String(body.result || "").toLowerCase().includes("placed");

    resultTitle.textContent = body.result || (isPlaced ? "Placed" : "Not Placed");

    if (body.probability !== undefined && body.probability !== null) {
      const probability = Number(body.probability);
      const percentage  = probability <= 1 ? probability * 100 : probability;
      resultText.textContent =
        `The model predicts that the student is likely to be ${isPlaced ? "placed" : "not placed"} ` +
        `with an estimated probability of ${percentage.toFixed(2)}%.`;
    } else {
      resultText.textContent =
        `The model predicts that the student is likely to be ${isPlaced ? "placed" : "not placed"}.`;
    }

    resultIcon.textContent = isPlaced ? "✓" : "×";
    resultIcon.classList.add(isPlaced ? "result-icon--placed" : "result-icon--not-placed");
    resultIcon.classList.add("animate-pop");

    resultEl.classList.add(isPlaced ? "result--placed" : "result--not-placed");
    resultEl.classList.add("animate-reveal");
    resultEl.style.display = "flex";

  } catch (err) {
    console.error(err);
    statusEl.textContent =
      `Unable to connect to the prediction server. ${err instanceof Error ? err.message : "Please try again."}`;
    statusEl.style.display = "block";
  } finally {
    button.disabled = false;
    button.textContent = "Predict Placement";
  }
});
