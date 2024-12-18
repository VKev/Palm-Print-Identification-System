import { log } from "@tensorflow/tfjs";
import React, { useState, useRef, useEffect } from "react";

export default function CamTest() {
  const [isCapturing, setIsCapturing] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const vectorList = useRef([]);

  // Function to send the frame to the API
  const sendFrameToAPI = async (imageData) => {
    try {
      // Make an async fetch request to your API
      const response = await fetch("http://localhost:5000/ai/vectorize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ images: [imageData] }),

      }).catch((err) => console.error("Error sending frame:", err));
      if (response.ok) {
        const result = await response.json();
        vectorList.current.push(result.feature_vector[0])
      }

      if (vectorList.current.length > 30) {
        console.log("Logged Responses:", vectorList.current);
        vectorList.current = []; // Clear the list
      }

      

    } catch (err) {
      console.error("Failed to send frame to API:", err);
    }
  };

  // Capture Frame and send to API
  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL("image/png");

    // Send the frame to the API without awaiting
    sendFrameToAPI(imageData);
  };

  // Start capturing frames
  const startCapturing = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      setIsCapturing(true);

      // Capture every 100ms
      intervalRef.current = setInterval(captureFrame, 100);
    } catch (err) {
      console.error("Error accessing media devices.", err);
    }
  };

  // Stop capturing frames
  const stopCapturing = () => {
    clearInterval(intervalRef.current);
    setIsCapturing(false);
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <div>
      <h1>Frame Capture</h1>
      <video ref={videoRef} autoPlay muted style={{ width: "600px" }}></video>
      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
      <div style={{ marginTop: "20px" }}>
        {!isCapturing ? (
          <button onClick={startCapturing}>Start Capturing Frames</button>
        ) : (
          <button onClick={stopCapturing}>Stop Capturing Frames</button>
        )}
      </div>
    </div>
  );
}
