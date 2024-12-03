document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const webcamSection = document.getElementById('webcam-section');
    const webcamFeed = document.getElementById('webcam-feed');
    const startButton = document.getElementById('start-webcam');
    const stopButton = document.getElementById('stop-webcam');

    // Function to start webcam
    window.startWebcam = function() {
        try {
            // Set the video feed source to the Flask route
            webcamFeed.src = '/video_feed';
            
            // Show stop button and hide start button
            startButton.classList.add('hidden');
            stopButton.classList.remove('hidden');
            
            console.log('Webcam started');
        } catch (error) {
            console.error('Error starting webcam:', error);
        }
    };

    // Function to stop webcam
    window.stopWebcam = function() {
        try {
            // Clear the video feed source
            webcamFeed.src = '';
            
            // Show start button and hide stop button
            startButton.classList.remove('hidden');
            stopButton.classList.add('hidden');
            
            console.log('Webcam stopped');
        } catch (error) {
            console.error('Error stopping webcam:', error);
        }
    };

    // Handle errors in video feed
    webcamFeed.onerror = function(error) {
        console.error('Error in video feed:', error);
        stopWebcam();
    };
});