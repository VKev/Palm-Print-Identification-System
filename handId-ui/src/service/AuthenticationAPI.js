import axios from "axios";
const REST_API_BASE = "http://localhost:8090/HandIdDetector"
const AUTHEN_URL = REST_API_BASE + "/api/auth/authenticate"
const GET_USER_INFO_URL = REST_API_BASE + "/api/info/get"
const REFRESH_TOKEN_URL = REST_API_BASE + "/api/auth/refresh-token"
const LOGOUT_URL = REST_API_BASE + "/api/auth/logout"


const AuthenticationAPI = {

    authenticate(authenticationRequest) {
        return axios.post(AUTHEN_URL, authenticationRequest);
    },


    refresh(refresh_token) {
        return axios.post(REFRESH_TOKEN_URL, {} ,{
            headers: {
                "Authorization": `Bearer ${refresh_token}`
            }
        });
    },


    logout(access_token) {
        return axios.post(LOGOUT_URL,{} , {
            headers: {
                "Authorization": `Bearer ${access_token}`
            }
        })
    },



    getInfo(access_token) {
        return axios.get(GET_USER_INFO_URL ,{
            headers: {
                "Authorization": `Bearer ${access_token}`,
                "Access-Control-Allow-Origin": "*",
                "Accept": "application/json"   // cach chua chay
            }
        });
    }

}

export default AuthenticationAPI;