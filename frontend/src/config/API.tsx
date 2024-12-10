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
        UPLOAD_PALM_PRINT_VIDEO_REGISTRATION: '/api/staff/upload-palm-print-video/registration/',
        UPLOAD_PALM_PRINT_VIDEO_RECOGNITION: '/api/staff/upload-palm-print-video/recognition/',
        UPLOAD_BACKGROUND_CUT_FRAME: '/api/staff/upload-filter-background-cut-images',
        REGISTER_INFERENCE: '/api/staff/register-palm-print/',
        TEST: '/api/staff/test-ai',
    },

    Admin: {
        GET_STAFF_ACCOUNTS: '/api/admin/staff-accounts/get-all',
        GET_STUDENT_DATA: '/api/admin/student-data/get-all',
        REGISTER_STAFF_ACCOUNT: '/api/admin/staff-accounts/register',
    }

}

export default API;