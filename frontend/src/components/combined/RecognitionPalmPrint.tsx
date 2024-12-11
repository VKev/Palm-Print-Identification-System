import HandRecognizer from "../single/HandRecognizer";
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import { IconButton } from '@mui/material';
import { useState } from "react";
import { ImageFile } from "../../models/Student";
import { CameraMode, RecognitionResult } from "../../models/PalmPrint";
import { UserProfile } from "../../models/User";

export default function RecognitionPalmPrint({user} : {user: UserProfile | null}) {

    const [selectedImages, setSelectedImages] = useState<ImageFile[]>([]);
    const [recognitionResult, setRecognitionResult] = useState<RecognitionResult | null>(null);
    const [cameraOn, setCameraOn] = useState(true);

    const toggleCamera = () => {
        setCameraOn(!cameraOn);
    };

    return (
        <div>
            <div className="mt-3 text-4xl text-center font-medium">Recognize Palm Print</div>
            <hr className="mt-5 mb-10" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                <div>
                    {
                        cameraOn && <HandRecognizer width={"100%"} maxWidth={"700px"}
                            cameraMode={CameraMode.RECOGNITION}
                            setSelectedImages={setSelectedImages}
                            setRecognitionResult={setRecognitionResult}
                            userProfile={user}
                            />
                    }

                    {/* <div>
                        <IconButton onClick={toggleCamera} style={{ color: cameraOn ? 'inherit' : 'red', fontSize: '3rem' }} title="Open/close camera">
                            {cameraOn ? <VideocamIcon style={{ fontSize: 'inherit' }} /> : <VideocamOffIcon style={{ fontSize: 'inherit' }} />}
                        </IconButton>
                    </div> */}
                    <div className="flex items-center gap-4">
                        <IconButton
                            onClick={toggleCamera}
                            className={`p-4 transition-colors duration-200 rounded-full hover:bg-gray-100 scale-125
        ${cameraOn ? 'text-blue-600 hover:text-blue-700' : 'text-red-600 hover:text-red-700'}`}
                            title="Open/close camera"
                        >
                            {cameraOn ? (
                                <VideocamIcon className="w-12 h-12" />
                            ) : (
                                <VideocamOffIcon className="w-12 h-12" />
                            )}
                        </IconButton>
                        <h2 className="text-xl font-semibold text-gray-700">
                            {cameraOn ? 'Camera Active' : 'Camera Inactive'}
                        </h2>
                    </div>
                </div>

                <div className="border-dashed border-2 border-gray-300 rounded-lg">


                    <div className="bg-white rounded-lg shadow-md p-6 max-w-md mx-auto my-4">
                        <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
                            Recognition Result
                        </h2>

                        <div className="space-y-4">
                            <div className="flex justify-between items-center border-b pb-2">
                                <span className="text-gray-600 font-medium">Accept</span>
                                <span className={`px-3 py-1 rounded-full ${recognitionResult?.accept
                                    ? 'bg-green-100 text-green-800'
                                    : 'bg-red-100 text-red-800'
                                    }`}>
                                    {recognitionResult?.accept ? "True" : "False"}
                                </span>
                            </div>

                            <div className="flex justify-between items-center border-b pb-2">
                                <span className="text-gray-600 font-medium">Average Occurrence Score</span>
                                <span className="text-gray-800">{recognitionResult?.average_occurrence_score}</span>
                            </div>

                            <div className="flex justify-between items-center border-b pb-2">
                                <span className="text-gray-600 font-medium">Average Similarity Score</span>
                                <span className="text-gray-800">{recognitionResult?.average_similarity_score}</span>
                            </div>

                            <div className="flex justify-between items-center border-b pb-2">
                                <span className="text-gray-600 font-medium">Most Common ID</span>
                                <span className="text-gray-800">{recognitionResult?.most_common_id}</span>
                            </div>

                            <div className="flex justify-between items-center border-b pb-2">
                                <span className="text-gray-600 font-medium">Occurrence Count</span>
                                <span className="text-gray-800">{recognitionResult?.occurrence_count}</span>
                            </div>

                            <div className="flex justify-between items-center border-b pb-2">
                                <span className="text-gray-600 font-medium">Score</span>
                                <span className="text-gray-800">{recognitionResult?.score}</span>
                            </div>

                            <div className="flex justify-between items-center">
                                <span className="text-gray-600 font-medium">Student Name</span>
                                <span className="text-gray-800">{recognitionResult?.student_info?.studentName}</span>
                            </div>
                        </div>
                    </div>

                </div>

            </div>
        </div>
    )
}
