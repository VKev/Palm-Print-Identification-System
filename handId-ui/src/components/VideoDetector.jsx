/* eslint-disable react/prop-types */
import { useState, useRef, useCallback, useEffect } from 'react';
import * as handpose from '@tensorflow-models/handpose';
import VideoAPI from '../service/VideoAPI';
import * as tf from '@tensorflow/tfjs';
import handImage from '../assets/hand-frame.jpg'
import { DEFAULT_MP4_NAME } from '../config/Constant';

function VideoDetector() {

    const [recording, setRecording] = useState(true);
    const [videoUrl, setVideoUrl] = useState(null);
    const [handDetected, setHandDetected] = useState(false);
    const [handDetectionTime, setHandDetectionTime] = useState(0);
    const mediaRecorderRef = useRef(null);
    const videoRef = useRef(null);
    const handposeModelRef = useRef(null);


    useEffect(() => {
        async function initTF() {
            await tf.ready();
            //console.log('TensorFlow.js backend:', tf.getBackend());
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

            if (handDetectionTime >= 3000) { // 3s
                mediaRecorderRef.current.stop();
                downloadVideo(); 
                setHandDetectionTime(0); 
            }
        }
    }, [handDetectionTime]);

    useEffect(() => {
        // Start recording immediately
        startRecording(); // Start recording when the component mounts
    }, []);

    useEffect(() => {
        let detectionInterval = setInterval(detectHand, 100);
       return () => {
        if (detectionInterval) clearInterval(detectionInterval);
       }
    }, [detectHand])

    const startRecording = useCallback(async () => {
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
                downloadVideo(); 
            };

            mediaRecorder.start();
            //setRecording(true);
            setHandDetectionTime(0);
        }
        catch (error) {
            alert('Error accessing media devices:', error);
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
        <div  >
            <div>
                <div>
                    <button style={{ fontSize: '22px' }} className='btn btn-warning mt-2' onClick={() => startRecording()} disabled={recording}>
                        <i className="bi bi-camera-video-fill"></i>&nbsp;&nbsp;&nbsp;
                        {recording ? 'Recording...' : 'Start Recording'}
                    </button>
                </div>

                <p className='text-danger mt-3' style={{ fontSize: 34 }}>
                    {recording
                        ? handDetected
                            ? `Hand detected for ${(handDetectionTime / 1000).toFixed(1)} seconds...`
                            : "Waiting for hand detection..."
                        : videoUrl
                            ? "Recording complete. Register successfully."
                            : "Press 'Start Recording' to begin."}
                </p>

                <div style={{
                    position: "relative",
                    width: '640px',
                    height: '480px',
                    margin: '0 auto'
                }}>


                    <video ref={videoRef} autoPlay muted style={{ width: '100%', maxWidth: '700px' }} />
                    <img src={handImage} style={{ position: 'absolute', height: '100%', width: '100%', top: 0, left: 0, pointerEvents: 'none' }} />

                </div >
            </div>


        </div>




    );
};

export default VideoDetector;