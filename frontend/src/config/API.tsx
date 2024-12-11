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
        UPLOAD_BACKGROUND_CUT_FRAME: '/api/staff/upload-filter-background-cut-images',
        REGISTER_INFERENCE: '/api/staff/register-palm-print/',
        TEST: '/api/staff/test-ai',
        RECOGNIZE_PALM_PRINT_BY_VIDEO: '/api/staff/recognize-palm-print/',
        RECOGNIZE_PALM_PRINT_BY_IMAGE: '/api/staff/recognize-palm-print-image',
        GET_HISTORY_LOGS_BY_STAFF: '/api/staff/history-logs/'
    },

    Admin: {
        GET_STAFF_ACCOUNTS: '/api/admin/staff-accounts/get-all',
        GET_STUDENT_DATA: '/api/admin/student-data/get-all',
        REGISTER_STAFF_ACCOUNT: '/api/admin/staff-accounts/register',
        CREATE_STUDENT: '/api/admin/student-data/create',
    }

}

export default API;