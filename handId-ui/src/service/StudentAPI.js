import axios from "axios";


const REST_API_BASE = "http://localhost:8090/HandIdDetector";
const STUDENT_CHECKING_API = REST_API_BASE + "/api/students/check/"
const GET_STUDENT_DATA_API = REST_API_BASE + "/admin/students/list";


const StudentAPI = {

    checkRoleNumber(roleNumber, access_token) {
        return axios.get(STUDENT_CHECKING_API + roleNumber, {
            headers: {
                "Authorization": `Bearer ${access_token}`,
                // "Access-Control-Allow-Origin": "*",
                // "Accept": "application/json"   
            }
        })
    },

    getStudentData(access_token) {
        return axios.get(GET_STUDENT_DATA_API, {
            headers: {
                Authorization: `Bearer ${access_token}`,
            },
        });
    },

}

export default StudentAPI;
