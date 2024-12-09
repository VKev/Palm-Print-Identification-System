import { IconButton } from "@mui/material";
import HandRecognizer from "../single/HandRecognizer";
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import { useState } from "react";

export default function RegisterPalmPrint() {

    const [cameraOn, setCameraOn] = useState(false);

    const toggleCamera = () => {
        setCameraOn(!cameraOn);
    };

    return (
        <div>
            <div className="mt-3 text-4xl text-center font-medium">Register Palm Print</div>
            <hr className="mt-5 mb-10" />
            <div className="grid grid-cols-2 gap-4">

                <div>
                    <div className="mb-5">
                        <div className="text-lg mb-2">Enter student code for register<span className="text-red-500">*</span></div>
                        <div className="flex">
                            <input type="text" placeholder="Enter student code ...."
                                className="w-4/6 p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                            <button className="w-1/6 ml-3 text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2">
                                Check
                            </button>
                        </div>
                    </div>
                    {
                        cameraOn && <HandRecognizer />
                    }

                    <div>
                        <IconButton onClick={toggleCamera} style={{ color: cameraOn ? 'inherit' : 'red', fontSize: '3rem' }} title="Open/close camera">
                            {cameraOn ? <VideocamIcon style={{ fontSize: 'inherit' }} /> : <VideocamOffIcon style={{ fontSize: 'inherit' }} />}
                        </IconButton>
                    </div>
                </div>

                <div className="border-dashed border-2 border-gray-300 rounded-lg">
                    <div >
                        <div className="text-2xl text-center">ROI images here</div>
                    </div>
                </div>

            </div>
        </div>
    )
}
