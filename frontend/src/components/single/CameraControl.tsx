import { useState } from 'react';
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import { IconButton } from '@mui/material';

const CameraControl = () => {
    const [cameraOn, setCameraOn] = useState(false);

    const toggleCamera = () => {
        setCameraOn(!cameraOn);
    };


    return (
        <IconButton onClick={toggleCamera} style={{ color: cameraOn ? 'inherit' : 'red' }}>
            {cameraOn ? <VideocamIcon /> : <VideocamOffIcon />}
        </IconButton>
    );
};

export default CameraControl;
