import axios from "axios";

const REST_API_BASE = "http://localhost:8090/HandIdDetector";
const GET_STAFF_ACCOUNTS =  REST_API_BASE + "/api/admin/get/staff-accounts"
const REGISTER_STAFF_ACCOUNT = REST_API_BASE + "/api/admin/register/staff-accounts"
const UPDATE_ACCOUNT_STAFF = REST_API_BASE + "/api/admin/update/staff-acccount/"
const DISABLE_ACCOUNT_STAFF = REST_API_BASE+ "/api/admin/disable/staff-acccount/"


const UserApi = {

    getAllStaffAccounts(access_token) {
        return axios.get(GET_STAFF_ACCOUNTS, {
            headers: {
                "Authorization": `Bearer ${access_token}` 
            }
        })
    },

    registerAccount(registerRequest ,access_token) {
        return axios.post(REGISTER_STAFF_ACCOUNT, registerRequest, {
            headers: {
                "Authorization": `Bearer ${access_token}`
            }
        } )
    },

    updateAccount(username, updateStaffRequest, access_token) {
        return axios.put(UPDATE_ACCOUNT_STAFF+username, updateStaffRequest, {
            headers: {
                "Authorization": `Bearer ${access_token}`
            }
        })
    },

    disableEnableAccount(username, access_token) {
        return axios.put(DISABLE_ACCOUNT_STAFF+username, {} , {
            headers: {
                "Authorization": `Bearer ${access_token}`
            }
        } )
    }

}


export default UserApi;



