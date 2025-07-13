/**
 * NightScan Async Prediction Client Example
 * 
 * This example demonstrates how to use the asynchronous prediction API
 * which returns immediately and allows checking status/results later.
 */

class NightScanAsyncClient {
    constructor(baseUrl = '/api/v1') {
        this.baseUrl = baseUrl;
        this.tasks = new Map(); // Track active tasks
    }

    /**
     * Submit a file for prediction
     * @param {File} file - The audio file to analyze
     * @returns {Promise<Object>} Task information including task_id
     */
    async submitPrediction(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                body: formData,
                credentials: 'include' // Include cookies for auth
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Upload failed');
            }

            const data = await response.json();
            
            // Track this task
            this.tasks.set(data.task_id, {
                taskId: data.task_id,
                filename: file.name,
                status: 'processing',
                createdAt: new Date()
            });

            return data;
        } catch (error) {
            console.error('Failed to submit prediction:', error);
            throw error;
        }
    }

    /**
     * Check the status of a prediction task
     * @param {string} taskId - The task ID to check
     * @returns {Promise<Object>} Current task status
     */
    async checkStatus(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/status/${taskId}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Status check failed');
            }

            const status = await response.json();
            
            // Update tracked task
            if (this.tasks.has(taskId)) {
                this.tasks.get(taskId).status = status.status;
                this.tasks.get(taskId).progress = status.progress;
            }

            return status;
        } catch (error) {
            console.error('Failed to check status:', error);
            throw error;
        }
    }

    /**
     * Get the results of a completed prediction
     * @param {string} taskId - The task ID to get results for
     * @returns {Promise<Object>} Prediction results
     */
    async getResults(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/result/${taskId}`, {
                credentials: 'include'
            });

            if (!response.ok) {
                const error = await response.json();
                
                // Task still processing
                if (response.status === 202) {
                    return {
                        ready: false,
                        status: error.status
                    };
                }
                
                throw new Error(error.error || 'Failed to get results');
            }

            const data = await response.json();
            
            // Remove from tracked tasks
            this.tasks.delete(taskId);
            
            return {
                ready: true,
                ...data
            };
        } catch (error) {
            console.error('Failed to get results:', error);
            throw error;
        }
    }

    /**
     * Poll for results with exponential backoff
     * @param {string} taskId - The task ID to poll
     * @param {Object} options - Polling options
     * @returns {Promise<Object>} Final results
     */
    async pollForResults(taskId, options = {}) {
        const {
            maxAttempts = 30,
            initialDelay = 1000,
            maxDelay = 10000,
            onProgress = null
        } = options;

        let attempts = 0;
        let delay = initialDelay;

        while (attempts < maxAttempts) {
            attempts++;

            // Check status
            const status = await this.checkStatus(taskId);
            
            // Call progress callback if provided
            if (onProgress) {
                onProgress(status);
            }

            // Check if completed
            if (status.status === 'completed') {
                return await this.getResults(taskId);
            }

            // Check if failed
            if (status.status === 'failed') {
                throw new Error(status.error || 'Prediction failed');
            }

            // Check if cancelled
            if (status.status === 'cancelled') {
                throw new Error('Prediction was cancelled');
            }

            // Wait before next attempt
            await new Promise(resolve => setTimeout(resolve, delay));
            
            // Exponential backoff
            delay = Math.min(delay * 1.5, maxDelay);
        }

        throw new Error('Polling timeout - prediction took too long');
    }

    /**
     * Cancel a pending prediction
     * @param {string} taskId - The task ID to cancel
     * @returns {Promise<Object>} Cancellation result
     */
    async cancelPrediction(taskId) {
        try {
            const response = await fetch(`${this.baseUrl}/cancel/${taskId}`, {
                method: 'POST',
                credentials: 'include'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Cancellation failed');
            }

            // Remove from tracked tasks
            this.tasks.delete(taskId);

            return await response.json();
        } catch (error) {
            console.error('Failed to cancel prediction:', error);
            throw error;
        }
    }

    /**
     * Get all tasks for the current user
     * @param {boolean} includeCompleted - Include completed tasks
     * @returns {Promise<Array>} List of user's tasks
     */
    async getUserTasks(includeCompleted = true) {
        try {
            const response = await fetch(
                `${this.baseUrl}/tasks?include_completed=${includeCompleted}`,
                { credentials: 'include' }
            );

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to get tasks');
            }

            const data = await response.json();
            return data.tasks;
        } catch (error) {
            console.error('Failed to get user tasks:', error);
            throw error;
        }
    }
}

// WebSocket integration for real-time updates
class NightScanWebSocketClient {
    constructor(wsUrl = 'ws://localhost:8000/ws') {
        this.wsUrl = wsUrl;
        this.ws = null;
        this.handlers = new Map();
    }

    /**
     * Connect to WebSocket server
     */
    connect() {
        this.ws = new WebSocket(this.wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Handle prediction complete events
                if (data.type === 'prediction_complete') {
                    this.emit('prediction_complete', data);
                }
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            // Reconnect after delay
            setTimeout(() => this.connect(), 5000);
        };
    }

    /**
     * Register event handler
     */
    on(event, handler) {
        if (!this.handlers.has(event)) {
            this.handlers.set(event, []);
        }
        this.handlers.get(event).push(handler);
    }

    /**
     * Emit event to handlers
     */
    emit(event, data) {
        const handlers = this.handlers.get(event) || [];
        handlers.forEach(handler => handler(data));
    }
}

// Example usage
async function exampleUsage() {
    const client = new NightScanAsyncClient();
    const wsClient = new NightScanWebSocketClient();

    // Connect WebSocket for real-time updates
    wsClient.connect();

    // File input handler
    const fileInput = document.getElementById('audioFile');
    const statusDiv = document.getElementById('status');
    const resultsDiv = document.getElementById('results');

    fileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        try {
            // Show uploading status
            statusDiv.textContent = 'Uploading file...';

            // Submit prediction
            const task = await client.submitPrediction(file);
            console.log('Task created:', task);

            statusDiv.textContent = `Processing... Task ID: ${task.task_id}`;

            // Option 1: Poll for results
            const results = await client.pollForResults(task.task_id, {
                onProgress: (status) => {
                    statusDiv.textContent = `Status: ${status.status} (${status.progress}%)`;
                }
            });

            // Display results
            console.log('Results:', results);
            resultsDiv.innerHTML = `
                <h3>Prediction Results</h3>
                <pre>${JSON.stringify(results.results, null, 2)}</pre>
                <p>Processing time: ${results.results.processing_time}s</p>
            `;

        } catch (error) {
            statusDiv.textContent = `Error: ${error.message}`;
            console.error(error);
        }
    });

    // Option 2: Use WebSocket for real-time updates
    wsClient.on('prediction_complete', async (data) => {
        console.log('Prediction completed via WebSocket:', data);
        
        // Get full results
        const results = await client.getResults(data.task_id);
        if (results.ready) {
            // Update UI with results
            resultsDiv.innerHTML = `
                <h3>Results for ${data.filename}</h3>
                <pre>${JSON.stringify(results.results, null, 2)}</pre>
            `;
        }
    });

    // Show user's tasks
    const showTasksBtn = document.getElementById('showTasks');
    showTasksBtn.addEventListener('click', async () => {
        try {
            const tasks = await client.getUserTasks();
            console.log('User tasks:', tasks);
            
            const tasksList = tasks.map(task => `
                <li>
                    ${task.filename} - ${task.status} 
                    (Created: ${new Date(task.created_at).toLocaleString()})
                    ${task.status === 'processing' ? 
                        `<button onclick="cancelTask('${task.task_id}')">Cancel</button>` : 
                        ''
                    }
                </li>
            `).join('');
            
            resultsDiv.innerHTML = `
                <h3>Your Tasks</h3>
                <ul>${tasksList}</ul>
            `;
        } catch (error) {
            console.error('Failed to get tasks:', error);
        }
    });

    // Make cancel function global for button onclick
    window.cancelTask = async (taskId) => {
        try {
            await client.cancelPrediction(taskId);
            alert('Task cancelled');
            // Refresh tasks list
            showTasksBtn.click();
        } catch (error) {
            alert(`Failed to cancel: ${error.message}`);
        }
    };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', exampleUsage);