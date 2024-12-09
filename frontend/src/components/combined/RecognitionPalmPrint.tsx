import HandRecognizer from "../single/HandRecognizer";
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import { IconButton } from '@mui/material';
import { useState } from "react";

export default function RecognitionPalmPrint() {

    const [cameraOn, setCameraOn] = useState(true);

    const toggleCamera = () => {
        setCameraOn(!cameraOn);
    };

    return (
        <div>
            <div className="mt-3 text-4xl text-center font-medium">Recognition Palm Print</div>
            <hr className="mt-5 mb-10" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">

                <div>
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
                    <div>
                        <div className="text-2xl text-center">ROI images here</div>
                    </div>
                </div>

            </div>
        </div>
    )
}
