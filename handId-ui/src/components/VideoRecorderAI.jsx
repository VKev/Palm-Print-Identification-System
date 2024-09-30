import { useState, useRef, useCallback, useEffect } from 'react';
import * as handpose from '@tensorflow-models/handpose';
import VideoAPI from '../service/VideoAPI';
import * as tf from '@tensorflow/tfjs';
import { ACCESS_TOKEN_KEY, DEFAULT_MP4_NAME } from '../config/Constant';

function VideoRecorderAI({ isOpen, roleNumber }) {

  const [recording, setRecording] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [handDetected, setHandDetected] = useState(false);
  const [handDetectionTime, setHandDetectionTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const videoRef = useRef(null);
  const handposeModelRef = useRef(null);

  //console.log("Rolenumber: ", roleNumber);
  

  useEffect(() => {
    async function initTF() {
      await tf.ready();
      console.log('TensorFlow.js backend:', tf.getBackend());
    }
    initTF();
  }, []);

  useEffect(() => {
    const loadHandposeModel = async () => {
      const model = await handpose.load();
      handposeModelRef.current = model;
    };
    loadHandposeModel();
  }, []);

  const detectHand = useCallback(async () => {
    if (videoRef.current && handposeModelRef.current) {
      const predictions = await handposeModelRef.current.estimateHands(videoRef.current);
      const isHandDetected = predictions.length > 0;
      setHandDetected(isHandDetected);

      if (isHandDetected) {
        setHandDetectionTime(prevTime => prevTime + 100);  // ms
      }
      else {
        // Reset if hand is not detected
        setHandDetectionTime(0);
      }

      if (handDetectionTime >= 3000) { // 3 seconds
        mediaRecorderRef.current.stop();
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
        setRecording(false);

      }
    }
  }, [handDetectionTime]);

  useEffect(() => {
    let detectionInterval;
    if (recording) {
      // Check every 100ms
      detectionInterval = setInterval(detectHand, 100);
    }
    return () => {
      if (detectionInterval) clearInterval(detectionInterval);
    };
  }, [recording, detectHand]);

  const startRecording = useCallback(async (roleNumber) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      videoRef.current.srcObject = stream;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      const chunks = [];
      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/mp4' });
        const url = URL.createObjectURL(blob);
        setVideoUrl(url);

        // Automatically download video
        const a = document.createElement('a');
        a.href = url;
        a.download = DEFAULT_MP4_NAME;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        VideoAPI.sendRequestToExtract(DEFAULT_MP4_NAME, roleNumber, localStorage.getItem(ACCESS_TOKEN_KEY)).then(
          response => {
            console.log(response.data);
          }
        )

      };

      mediaRecorder.start();
      setRecording(true);
      setHandDetectionTime(0);
    }
    catch (error) {
      console.error('Error accessing media devices:', error);
    }
  }, []);

  const downloadVideo = () => {
    if (videoUrl) {
      const a = document.createElement('a');
      a.href = videoUrl;
      a.download = DEFAULT_MP4_NAME;
      a.click();
    }
  };

  return (
    <div>
      {
        isOpen && 
        <div>
          <video ref={videoRef} autoPlay muted style={{ width: '100%', maxWidth: '700px' }} />
          <div>
            <button className='btn btn-warning mt-3' onClick={() => startRecording(roleNumber)} disabled={recording}>
              <i className="bi bi-camera-video-fill"></i>&nbsp;&nbsp;&nbsp;
              {recording ? 'Recording...' : 'Start Recording'}
            </button>
            {videoUrl && (
              <button className='btn btn-success mx-3 mt-3' onClick={downloadVideo}>Download Video</button>
            )}
          </div>
          <p className='text-danger mt-3' style={{ fontSize: 24 }}>
            {recording
              ? handDetected
                ? `Hand detected for ${(handDetectionTime / 1000).toFixed(1)} seconds...`
                : "Waiting for hand detection..."
              : videoUrl
                ? "Recording complete. You can now download the video."
                : "Press 'Start Recording' to begin."}
          </p>
        </div > 
      }
    </div>




  );
};

export default VideoRecorderAI;