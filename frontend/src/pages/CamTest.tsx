import React, { useState, useRef, useEffect } from "react";

export default function CamTest() {
    const [isCapturing, setIsCapturing] = useState(false);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const intervalRef = useRef(null);
  
    const captureFrame = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");
  
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
  
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const imageData = canvas.toDataURL("image/png");
      console.log("Captured Frame Data:", imageData); // Này là base64 nè bạn, cắt đủ 30 frames rồi mới gửi
    };
  
    const startCapturing = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
        setIsCapturing(true);

        intervalRef.current = setInterval(captureFrame, 100); // Cắt frame mỗi 100 mili giây 
      } catch (err) {
        console.error("Error accessing media devices.", err);
      }
    };
  
    const stopCapturing = () => {
      clearInterval(intervalRef.current); 
      setIsCapturing(false);
    };
  
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
