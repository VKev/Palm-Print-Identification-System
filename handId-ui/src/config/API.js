
const API = {

    BASE_API : 'http://localhost:8090/HandIdDetector',

    Authenticaion: {
        LOGIN : '/api/auth/authenticate',
        REFRESH_TOKEN : '/api/auth/refresh-token',
        GET_INFO : '/api/info/get'
    },

    Student: {
        STUDENT_CHECKING : "/api/students/check/",
        GET_STUDENT_DATA : "/api/students/list"
    },

    User : {
        GET_STAFF_ACCOUNTS : "/api/admin/get/staff-accounts",
        REGISTER_STAFF_ACCOUNT : "/api/admin/register/staff-accounts",
        UPDATE_ACCOUNT_STAFF : "/api/admin/update/staff-acccount/",
        DISABLE_ACCOUNT_STAFF : "/api/admin/disable/staff-acccount/"
    },
    
    Video : {
        EXTRACT_VIDEO_API : "/api/video/process"
    }

}

export default API