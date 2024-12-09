import { useCallback, useEffect, useRef, useState } from "react";
import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import { toast } from "react-toastify";
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import LoadEffect from './LoadEffect'; 
import DeviceNameIdentifier from "./DeviceNameIdentifier";

// const DEFAULT_MP4_NAME = "recorded-video.mp4";

export default function HandRecognizer() {

    const [recording] = useState<boolean>(true);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [handDetected, setHandDetected] = useState<boolean>(false);
    const [handDetectionTime, setHandDetectionTime] = useState<number>(0);
    const [isLoading, setIsLoading] = useState(true);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const handposeModelRef = useRef<handpose.HandPose | null>(null);


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
                // reset
                setHandDetectionTime(0);
                setHandDetected(false)
            }

            if (handDetectionTime >= 3000) { // 3s
                mediaRecorderRef.current?.stop();
                // downloadVideo();
                setHandDetectionTime(0);
            }
        }

    }, [handDetectionTime]);

    useEffect(() => {
        startRecording(); 
    }, []);

    useEffect(() => {
        const detectionInterval: NodeJS.Timeout = setInterval(detectHand, 100);
        return () => {
            if (detectionInterval) clearInterval(detectionInterval);
        }
    }, [detectHand])

    const startRecording = useCallback(async () => {

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }

            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;

            const chunks: BlobPart[] | undefined = [];
            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/mp4' });
                const url = URL.createObjectURL(blob);
                setVideoUrl(url);
                // downloadVideo();
            };

            mediaRecorder.start();
            //setRecording(true);
            setHandDetectionTime(0);
        }
        catch (error) {
            toast.error(`Error accessing media devices: ${error as string}`)
        }

    }, []);

    // const downloadVideo = () => {
    //     if (videoUrl) {
    //         const a = document.createElement('a');
    //         a.href = videoUrl;
    //         //a.download = DEFAULT_MP4_NAME;
    //         a.click();
    //     }
    // };

    useEffect(() => {
        const videoElement = videoRef.current;

        const handleLoadedData = () => {
            setIsLoading(false);
        };

        if (videoElement) {
            videoElement.addEventListener('loadeddata', handleLoadedData);
        }

        return () => {
            if (videoElement) {
                videoElement.removeEventListener('loadeddata', handleLoadedData);
            }
        };
    }, []);

    return (
        <div  >
            <div>
                <div>
                    <button className='text-black bg-gradient-to-r from-yellow-400 via-yellow-500 to-yellow-500 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-yellow-300 dark:focus:ring-yellow-800 font-medium rounded-lg text-sm px-4 py-2 text-center'
                        onClick={() => startRecording()} disabled={recording}>
                        &nbsp;&nbsp;&nbsp;
                        <RadioButtonCheckedIcon color="error" /> {recording ? 'Recording...' : 'Start Recording'}
                        &nbsp;&nbsp;&nbsp;
                    </button>

                </div>

                <p className='text-2xl mb-5 mt-4'>
                    {recording
                        ? handDetected
                            ? `Hand detected for ${(handDetectionTime / 1000).toFixed(1)} seconds...`
                            : "Waiting for hand detection..."
                        : videoUrl
                            ? "Recording complete. Register successfully."
                            : "Press 'Start Recording' to begin."}
                </p>

                <div style={{ position: 'relative', width: '100%', maxWidth: '600px' }}>
                    {isLoading && (
                        <div style={{
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            backgroundColor: 'rgba(255, 255, 255, 0.8)'
                        }}>
                            <LoadEffect />
                        </div>
                    )}

                    <video
                        className="border-dashed border-2 border-gray-300 rounded-lg"
                        ref={videoRef}
                        autoPlay
                        muted
                        style={{ width: '100%', maxWidth: '700px' }}
                        // transform: 'scaleX(-1)'
                    />
                    <div>
                        <DeviceNameIdentifier />
                    </div>
                </div>
            </div>
        </div>
    );
}
