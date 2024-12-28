import { useState, useRef, useEffect } from "react";
import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";

export default function CamTest() {
  const [isCapturing, setIsCapturing] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const modelRef = useRef<any>(null);
  const vectorList = useRef<any[]>([]);

  // Load the handpose model
  const loadModel = async () => {
    try {
      modelRef.current = await handpose.load();
      console.log("Handpose model loaded");
    } catch (error) {
      console.error("Error loading Handpose model", error);
    }
  };

  useEffect(() => {
    tf.ready().then(loadModel);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Function to detect hands in the video frame
  const detectHands = async () => {
    const video = videoRef.current;
    if (modelRef.current && video && video.readyState === 4) {
      const predictions = await modelRef.current.estimateHands(video);
      return predictions.length > 0; // Return true if a hand is detected
    }
    return false;
  };

  // Function to send the frame to the API
  const sendFrameToAPI = async (imageData: string) => {
    try {
      const response = await fetch("http://localhost:5000/ai/vectorize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ images: [imageData] }),
      });

      if (response.ok) {
        const result = await response.json();
        vectorList.current.push(result.feature_vector[0]);
      }

      if (vectorList.current.length > 30) {
        console.log("Logged Responses:", vectorList.current);
        vectorList.current = []; // Clear the list
      }
    } catch (err) {
      console.error("Failed to send frame to API:", err);
    }
  };

  // Capture frame if a hand is detected
  const captureFrameIfHandDetected = async () => {
    const isHandDetected = await detectHands();

    if (isHandDetected) {
      const video = videoRef.current!;
      const canvas = canvasRef.current!;
      const context = canvas.getContext("2d")!;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL("image/png");

      // Send the frame to the API
      console.log(imageData.substring(0, 20));
      //sendFrameToAPI(imageData);
    }
  };

  // Start capturing frames
  const startCapturing = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsCapturing(true);

        // Capture every 30ms
        intervalRef.current = setInterval(captureFrameIfHandDetected, 30);
      }
    } catch (err) {
      console.error("Error accessing media devices.", err);
    }
  };

  // Stop capturing frames
  const stopCapturing = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    setIsCapturing(false);
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach((track) => track.stop());
  };

  return (
    <div>
      <h1>Hand Detection and Frame Capture</h1>
      <video ref={videoRef} autoPlay muted style={{ width: "600px" }}></video>
      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
      <div style={{ marginTop: "20px" }}>
        {!isCapturing ? (
          <button onClick={startCapturing}>Start Capturing</button>
        ) : (
          <button onClick={stopCapturing}>Stop Capturing</button>
        )}
      </div>
    </div>
  );
}
