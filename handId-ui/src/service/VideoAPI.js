import axios from "axios";

const REST_API_BASE = "http://localhost:8090/HandIdDetector";
const EXTRACT_VIDEO_API = REST_API_BASE + "/api/video/process"

const VideoAPI = {

    sendRequestToExtract(filename, roleNumber, access_token) {
        console.log("Role number api: ",roleNumber)
        return axios.post(EXTRACT_VIDEO_API, null, {
            headers: {
                "Authorization": `Bearer ${access_token}`
            },
            params: {
                filename,
                roleNumber
            }
        });
    }

}

export default VideoAPI;