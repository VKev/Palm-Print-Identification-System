import { useState, useEffect } from 'react';

export default function DeviceNameIdentifier() {

    const [deviceName, setDeviceName] = useState('');

    useEffect(() => {
        const getDeviceName = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const videoTrack = stream.getVideoTracks()[0];
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevice = devices.find(device => device.deviceId === videoTrack.getSettings().deviceId);
                if (videoDevice) {
                    setDeviceName(videoDevice.label);
                }
            } catch (error) {
                console.error('Error accessing media devices.', error);
            }
        };

        getDeviceName();
    }, []);

    return (
        <div className='mt-2 text-gray-500'>
            {deviceName ? `Device: ${deviceName}` : 'Loading device name...'}
        </div>
    );
};
