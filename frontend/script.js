document.addEventListener("DOMContentLoaded", () => {
    
    const dropZoneElement = document.getElementById("drop-zone");
    const inputElement = document.getElementById("file-input");
    const promptElement = dropZoneElement.querySelector(".drop-zone__prompt");
    const form = document.getElementById("upload-form");
    const targetColumnSelect = document.getElementById("target-column");
    const taskTypeSelect = document.getElementById("task-type");

    // Views
    const uploadView = document.getElementById("upload-view");
    const loadingView = document.getElementById("loading-view");
    const dashboardView = document.getElementById("dashboard-view");
    const dataExplorer = document.getElementById("data-explorer");
    const loadingText = document.getElementById("loading-text");

    // Dashboard Elements
    const bestModelEl = document.getElementById("best-model");
    const testMetricTitleEl = document.getElementById("test-metric-title");
    const testMetricValueEl = document.getElementById("test-metric-value");
    const leaderboardMetricTitleEl = document.getElementById("leaderboard-metric-title");
    const leaderboardTableBody = document.querySelector("#leaderboard-table tbody");
    
    // Batch Predict Elements
    const batchForm = document.getElementById("batch-predict-form");
    const batchFileInput = document.getElementById("batch-file-input");
    const batchResult = document.getElementById("batch-result");
    let currentHeaders = [];
    let currentTarget = "";
    let uploadedDataSample = null;

    // Data Explorer Elements
    const previewTableHead = document.querySelector("#data-preview-table thead");
    const previewTableBody = document.querySelector("#data-preview-table tbody");
    let chart1Instance = null;
    let chart2Instance = null;
    let predictionChartInstance = null;

    // Drag and Drop Logic
    dropZoneElement.addEventListener("click", (e) => {
        inputElement.click();
    });

    inputElement.addEventListener("change", (e) => {
        if (inputElement.files.length) {
            handleFileSelect(inputElement.files[0]);
        }
    });

    dropZoneElement.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZoneElement.classList.add("drop-zone--over");
    });

    ["dragleave", "dragend"].forEach((type) => {
        dropZoneElement.addEventListener(type, (e) => {
            dropZoneElement.classList.remove("drop-zone--over");
        });
    });

    dropZoneElement.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            handleFileSelect(e.dataTransfer.files[0]);
        }
        dropZoneElement.classList.remove("drop-zone--over");
    });

    function handleFileSelect(file) {
        promptElement.innerHTML = `<strong>Selected:</strong> ${file.name}`;
        
        // Use PapaParse to parse the CSV locally
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function(results) {
                const data = results.data;
                if (data.length > 0) {
                    const headers = Object.keys(data[0]);
                    currentHeaders = headers;
                    uploadedDataSample = data[0]; 
                    
                    // Populate Target Dropdown
                    targetColumnSelect.innerHTML = '';
                    headers.forEach(header => {
                        const option = document.createElement("option");
                        option.value = header;
                        option.textContent = header;
                        targetColumnSelect.appendChild(option);
                    });
                    
                    // Render Data Preview
                    renderDataPreview(headers, data.slice(0, 5));
                    
                    // Render Charts
                    renderCharts(headers, data);
                    
                    // Show Data Explorer and Form
                    dataExplorer.style.display = "block";
                    form.style.display = "block";
                }
            }
        });
    }

    function renderDataPreview(headers, rows) {
        previewTableHead.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
        previewTableBody.innerHTML = rows.map(row => {
            return `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`;
        }).join('');
    }

    function renderCharts(headers, data) {
        // Destroy existing charts if any
        if (chart1Instance) chart1Instance.destroy();
        if (chart2Instance) chart2Instance.destroy();

        // Chart.js defaults for Light Theme (AWS Style)
        Chart.defaults.color = '#545b64';
        Chart.defaults.borderColor = '#eaeded';

        let catCol = headers.find(h => typeof data[0][h] === 'string');
        if (!catCol) {
            for (let h of headers) {
                const uniqueValues = new Set(data.map(r => r[h]));
                if (uniqueValues.size > 0 && uniqueValues.size <= 10) {
                    catCol = h;
                    break;
                }
            }
        }
        
        let numCol = headers.find(h => {
            const uniqueValues = new Set(data.map(r => r[h]));
            return typeof data[0][h] === 'number' && uniqueValues.size > 10;
        });
        if (!numCol && headers.length > 0) numCol = headers[0];

        if (catCol) {
            const counts = {};
            data.forEach(row => {
                const val = row[catCol] !== null ? row[catCol] : 'Unknown';
                counts[val] = (counts[val] || 0) + 1;
            });

            const ctx1 = document.getElementById('chart-1').getContext('2d');
            chart1Instance = new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: Object.keys(counts),
                    datasets: [{
                        label: `Count by ${catCol}`,
                        data: Object.values(counts),
                        backgroundColor: 'rgba(0, 115, 187, 0.8)',
                        borderColor: '#005a9e',
                        borderWidth: 1
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        }

        if (numCol) {
            const values = data.map(r => r[numCol]).filter(v => v !== null && !isNaN(v));
            if(values.length > 0) {
                const min = Math.min(...values);
                const max = Math.max(...values);
                const bins = 10;
                const binSize = (max - min) / bins;
                
                const histogram = Array(bins).fill(0);
                values.forEach(v => {
                    let binIndex = Math.floor((v - min) / binSize);
                    if (binIndex >= bins) binIndex = bins - 1;
                    histogram[binIndex]++;
                });

                const labels = Array(bins).fill(0).map((_, i) => `${(min + i*binSize).toFixed(1)} - ${(min + (i+1)*binSize).toFixed(1)}`);

                const ctx2 = document.getElementById('chart-2').getContext('2d');
                chart2Instance = new Chart(ctx2, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: `Distribution of ${numCol}`,
                            data: histogram,
                            backgroundColor: 'rgba(255, 153, 0, 0.8)',
                            borderColor: '#ec7211',
                            borderWidth: 1
                        }]
                    },
                    options: { 
                        responsive: true, 
                        maintainAspectRatio: false,
                        scales: { x: { display: false } }
                    }
                });
            }
        }
    }

    // Form Submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const file = inputElement.files[0];
        const targetColumn = targetColumnSelect.value;
        const taskType = taskTypeSelect.value;
        currentTarget = targetColumn;

        if (!file || !targetColumn) {
            alert("Please provide both a file and a target column.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("target_column", targetColumn);
        formData.append("task_type", taskType);

        // Switch to loading view
        switchView(uploadView, loadingView);
        animateLoadingText();

        try {
            const response = await fetch("/upload-and-train", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Pipeline failed");
            }

            const data = await response.json();
            populateDashboard(data.data, taskType);
            
            // Fetch predictions to draw output dashboard
            await renderOutputDashboard(targetColumn, taskType);

            // Switch to dashboard view
            switchView(loadingView, dashboardView);

        } catch (error) {
            alert("Error: " + error.message);
            switchView(loadingView, uploadView);
        }
    });

    // Batch Prediction Submission
    batchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = batchFileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        batchResult.textContent = "Processing batch predictions...";
        batchResult.style.color = "var(--text-secondary)";

        try {
            const response = await fetch("/batch-predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Batch prediction failed");
            }

            // Trigger file download
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "untested_predictions.csv";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);

            batchResult.textContent = "✅ Predictions generated and downloaded successfully!";
            batchResult.style.color = "var(--aws-blue)";
        } catch (error) {
            batchResult.textContent = `Error: ${error.message}`;
            batchResult.style.color = "red";
        }
    });

    function switchView(hideView, showView) {
        hideView.classList.remove("active");
        setTimeout(() => {
            showView.classList.add("active");
        }, 400); // Wait for fade out
    }

    function animateLoadingText() {
        const steps = [
            "Cleaning data...",
            "Engineering features...",
            "Training AutoML models...",
            "Evaluating models...",
            "Generating predictions...",
            "Generating reports..."
        ];
        let stepIndex = 0;
        
        const interval = setInterval(() => {
            if (!loadingView.classList.contains("active")) {
                clearInterval(interval);
                return;
            }
            stepIndex = (stepIndex + 1) % steps.length;
            loadingText.textContent = steps[stepIndex];
        }, 3000);
    }

    function populateDashboard(resultData, taskType) {
        const bestModelName = resultData.automl_report.best_model;
        const metricName = resultData.automl_report.metric;
        
        let testMetricValue;
        if (metricName === "rmse") {
            testMetricTitleEl.textContent = "Test RMSE";
            leaderboardMetricTitleEl.textContent = "RMSE Score";
            testMetricValue = resultData.evaluation_report.mean_rmse.toFixed(3);
        } else {
            testMetricTitleEl.textContent = "Test Accuracy";
            leaderboardMetricTitleEl.textContent = "Accuracy Score";
            testMetricValue = (resultData.evaluation_report.mean_accuracy * 100).toFixed(1) + "%";
        }

        bestModelEl.textContent = bestModelName;
        testMetricValueEl.textContent = testMetricValue;
        
        const results = resultData.automl_report.results;
        const modelsArray = [];
        for (const [name, metrics] of Object.entries(results)) {
            modelsArray.push({ name: name, score: metrics[metricName] });
        }
        
        if (metricName === "rmse") {
            modelsArray.sort((a, b) => a.score - b.score);
        } else {
            modelsArray.sort((a, b) => b.score - a.score);
        }
        
        leaderboardTableBody.innerHTML = "";
        
        modelsArray.forEach((model, index) => {
            const isBest = model.name === bestModelName;
            const tr = document.createElement("tr");
            if (isBest) tr.classList.add("best-model-row");
            
            let displayScore = metricName === "accuracy" ? (model.score * 100).toFixed(1) + "%" : model.score.toFixed(3);

            tr.innerHTML = `
                <td>${model.name}</td>
                <td>${displayScore}</td>
                <td>${isBest ? '<span class="badge badge-winner">Selected Winner</span>' : ''}</td>
            `;
            leaderboardTableBody.appendChild(tr);
        });
    }

    async function renderOutputDashboard(targetColumn, taskType) {
        if (predictionChartInstance) predictionChartInstance.destroy();
        
        try {
            const response = await fetch("/download-predictions");
            const csvText = await response.text();
            
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    const data = results.data;
                    const predColName = `Predicted_${targetColumn}`;
                    
                    const ctx = document.getElementById('prediction-chart').getContext('2d');
                    
                    if (taskType === "classification") {
                        // Calculate Accuracy (Correct vs Incorrect)
                        let correct = 0;
                        let incorrect = 0;
                        
                        data.forEach(row => {
                            if (row[targetColumn] !== undefined && row[predColName] !== undefined) {
                                // String conversion for safety against type mismatches (1 vs "1")
                                if (String(row[targetColumn]) === String(row[predColName])) {
                                    correct++;
                                } else {
                                    incorrect++;
                                }
                            }
                        });
                        
                        document.getElementById("prediction-chart-title").textContent = "Model Accuracy (Actual vs Predicted)";
                        
                        predictionChartInstance = new Chart(ctx, {
                            type: 'bar',
                            data: {
                                labels: ['Correct Predictions', 'Errors (Incorrect)'],
                                datasets: [{
                                    label: 'Number of Rows',
                                    data: [correct, incorrect],
                                    backgroundColor: ['rgba(0, 115, 187, 0.8)', 'rgba(236, 114, 17, 0.8)'], // AWS Blue for Correct, Orange for Error
                                    borderColor: ['#005a9e', '#d13212'],
                                    borderWidth: 1
                                }]
                            },
                            options: { 
                                responsive: true, 
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: { display: false }
                                }
                            }
                        });
                    } else {
                        // Scatter Plot or Histogram for Regression
                        document.getElementById("prediction-chart-title").textContent = "Predicted Values Distribution";
                        
                        const predictions = data.map(r => r[predColName]).filter(v => v !== undefined && v !== null);
                        const min = Math.min(...predictions);
                        const max = Math.max(...predictions);
                        const bins = 10;
                        const binSize = ((max - min) / bins) || 1;
                        
                        const histogram = Array(bins).fill(0);
                        predictions.forEach(v => {
                            let binIndex = Math.floor((v - min) / binSize);
                            if (binIndex >= bins) binIndex = bins - 1;
                            histogram[binIndex]++;
                        });

                        const labels = Array(bins).fill(0).map((_, i) => `${(min + i*binSize).toFixed(1)}`);

                        predictionChartInstance = new Chart(ctx, {
                            type: 'bar',
                            data: {
                                labels: labels,
                                datasets: [{
                                    label: 'Distribution of Predictions',
                                    data: histogram,
                                    backgroundColor: 'rgba(0, 115, 187, 0.8)',
                                    borderColor: '#005a9e',
                                    borderWidth: 1
                                }]
                            },
                            options: { 
                                responsive: true, 
                                maintainAspectRatio: false,
                                scales: { x: { display: false } }
                            }
                        });
                    }
                }
            });
        } catch (err) {
            console.error("Failed to load predictions for chart:", err);
        }
    }
});
