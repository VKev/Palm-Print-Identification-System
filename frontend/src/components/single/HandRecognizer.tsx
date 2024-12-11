/* eslint-disable @typescript-eslint/no-unused-vars */
import { useCallback, useEffect, useRef, useState } from "react";
import * as handpose from '@tensorflow-models/handpose';
import * as tf from '@tensorflow/tfjs';
import { toast } from "react-toastify";
import RadioButtonCheckedIcon from '@mui/icons-material/RadioButtonChecked';
import LoadEffect from './LoadEffect';
import DeviceNameIdentifier from "./DeviceNameIdentifier";
import useAxios from "../../utils/useAxios";
import API from "../../config/API";
import { CameraMode, RecognitionResult, VideoUploadedResponse } from "../../models/PalmPrint";
import HttpStatus from "../../config/HttpStatus";
import { v4 as uuidv4 } from 'uuid';
import { FileType, ImageFile } from "../../models/Student";
import { UserProfile } from "../../models/User";

type Props = {
    width: string;
    maxWidth: string;
    cameraMode: string;
    studentCode?: string | null;
    userProfile?: UserProfile | null;
    setSelectedImages: (imagesFiles: ImageFile[]) => void;
    setRecognitionResult: (recognitionResult: RecognitionResult | null) => void;
}


export default function HandRecognizer(cameraProps: Props) {

    const api = useAxios();
    const [recording] = useState<boolean>(true);
    const [videoUrl, setVideoUrl] = useState<string | null>(null);
    const [handDetected, setHandDetected] = useState<boolean>(false);
    const [handDetectionTime, setHandDetectionTime] = useState<number>(0);
    const [isLoading, setIsLoading] = useState(true);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const videoRef = useRef<HTMLVideoElement | null>(null);
    const handposeModelRef = useRef<handpose.HandPose | null>(null);
    const [isHandlingVideo, setIsHandlingVideo] = useState<boolean>(false);
    const [videoUploadedResponse, setVideoUploadedResponse] = useState<VideoUploadedResponse | null>(null);


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

    const sendVideoToServer = async (videoBlob: Blob, studentCode: string) => {
        switch(cameraProps.cameraMode) {
            case CameraMode.REGISTRATION:
                uploadVideoForRegistration(videoBlob, studentCode);
                break;
            case CameraMode.RECOGNITION:
                recognizePalmPrint(videoBlob);
                break;
            default:
                toast.error('Invalid camera mode');
                break;
        }
    };

    const recognizePalmPrint = async (videoBlob: Blob) => {
        setIsHandlingVideo(true);
        const formData = new FormData();
        formData.append('video', videoBlob, uuidv4()+'.mp4');
        try {
            const response = await api.post(API.Staff.RECOGNIZE_PALM_PRINT_BY_VIDEO + cameraProps.userProfile?.id , formData);
            if (response.status === HttpStatus.OK) {
                //console.log(response.data);
                //setVideoUploadedResponse(response.data);
                // cameraProps.setSelectedImages(response.data.base64Images.map(
                //     (image: string) => ({ file: null, base64: image, isSelected: false, type: FileType.BASE64 })
                // ));
                cameraProps.setRecognitionResult(response.data);
                toast.success(response.data.student_info.studentCode + " - "+ response.data.student_info.studentName);
            }
            else {
                toast.error('Error uploading video');
            }
        } 
        catch (error: any) {
            toast.error('Error uploading video:', error);
        }
        finally {
            setIsHandlingVideo(false);
        }
    }

    const uploadVideoForRegistration = async (videoBlob: Blob, studentCode: string) => {
        setIsHandlingVideo(true);
        const formData = new FormData();
        formData.append('video', videoBlob, uuidv4()+'.mp4');
        try {
            const response = await api.post(API.Staff.UPLOAD_PALM_PRINT_VIDEO_REGISTRATION + studentCode, formData);
            if (response.status === HttpStatus.OK) {
                console.log(response.data);
                setVideoUploadedResponse(response.data);
                cameraProps.setSelectedImages(response.data.base64Images.map(
                    (image: string) => ({ file: null, base64: image, isSelected: false, type: FileType.BASE64 })
                ));
                toast.success(response.data.message);
            }
            else {
                toast.error('Error uploading video');
            }
        } 
        catch (error: any) {
            toast.error('Error uploading video:', error);
        }
        finally {
            setIsHandlingVideo(false);
        }
    }

    const detectHand = useCallback(async () => {
        if (videoRef.current && handposeModelRef.current) {
            const predictions = await handposeModelRef.current.estimateHands(videoRef.current);
            const isHandDetected = predictions.length > 0;
            setHandDetected(isHandDetected);

            if (isHandDetected) {
                setHandDetectionTime(prevTime => prevTime + 100);  // ms
            }
            else { // reset
                setHandDetectionTime(0);
                setHandDetected(false)
            }

            if (handDetectionTime >= 3000) { // 3s
                if (mediaRecorderRef.current) {
                    mediaRecorderRef.current.stop();
                    mediaRecorderRef.current.ondataavailable = (event) => {
                        if (event.data.size > 0) {
                            sendVideoToServer(event.data, cameraProps.studentCode || '');
                        }
                    };
                    setHandDetectionTime(0);
                }
            }
        }
    }, [handDetectionTime, cameraProps.studentCode]);

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
            };
            mediaRecorder.start();
            //setRecording(true);
            setHandDetectionTime(0);
        }
        catch (error) {
            toast.error(`Error accessing media devices: ${error as string}`)
        }

    }, []);

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
        <div className="container mx-auto px-4 max-w-4xl">
            <div className="space-y-8">
                {/* Recording Button Section */}
                <div className="flex justify-center">
                    <button 
                        className={`
                            flex items-center space-x-2 
                            px-6 py-3 
                            text-black font-semibold rounded-lg
                            transition-all duration-300
                            ${recording 
                                ? 'bg-red-400 hover:bg-red-500' 
                                : 'bg-gradient-to-r from-yellow-300 to-yellow-400 hover:from-yellow-400 hover:to-yellow-500'
                            }
                            focus:ring-4 focus:outline-none focus:ring-yellow-300
                            shadow-lg hover:shadow-xl
                        `}
                        onClick={() => startRecording()} 
                        disabled={recording}
                    >
                        <RadioButtonCheckedIcon 
                            className={recording ? 'animate-pulse' : ''} 
                            color="error" 
                        />
                        <span>{recording ? 'Recording...' : 'Start Recording'}</span>
                    </button>
                </div>
    
                {/* Status Message */}
                <p className="text-2xl text-center font-medium text-gray-700 
                             animate-fade-in">
                    {recording
                        ? handDetected
                            ? <span className="text-green-600">
                                Hand detected for {(handDetectionTime / 1000).toFixed(1)} seconds...
                              </span>
                            : <span className="text-yellow-600">
                                Waiting for hand detection...
                              </span>
                        : videoUrl
                            ? <span className="text-blue-600">
                                Recording complete. Register successfully.
                              </span>
                            : <span className="text-gray-500">
                                Press 'Start Recording' to begin.
                              </span>
                    }
                </p>
    
                {/* Video Section */}
                <div className="relative">
                    {(isLoading || isHandlingVideo) && (
                        <div className="absolute inset-0 bg-white/80 
                                      backdrop-blur-sm flex items-center justify-center
                                      z-10 rounded-lg transition-all duration-300">
                            <LoadEffect />
                        </div>
                    )}
    
                    <div className="flex flex-col items-center space-y-4">
                        <video
                            className="rounded-lg border-2 border-dashed border-gray-300
                                     shadow-lg hover:shadow-xl transition-shadow duration-300
                                     bg-gray-50"
                            ref={videoRef}
                            autoPlay
                            muted
                            style={{
                                width: cameraProps.width,
                                maxWidth: cameraProps.maxWidth
                            }}
                        />
                        <div>
                            <DeviceNameIdentifier />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
