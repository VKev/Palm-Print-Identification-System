const API = {

    BASE_API: "http://localhost:8090",
    
    Authenticaion: {
        LOGIN : '/api/auth/authenticate',
        REFRESH_TOKEN : '/api/auth/refresh-token',
        GET_INFO : '/api/auth/user/info',
    },

    Staff: {
        VALIDATE_STUDENT_CODE: '/api/staff/validate-student-code/',
        UPLOAD_PALM_PRINT_FRAME: '/api/staff/upload-palm-print-images/',
        TEST: '/api/staff/test-ai',
    }

}

export default API;