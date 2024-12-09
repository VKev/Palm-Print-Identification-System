const API = {

    BASE_API: "http://localhost:8080",
    
    Authenticaion: {
        LOGIN : '/api/auth/authenticate',
        REFRESH_TOKEN : '/api/auth/refresh-token',
        GET_INFO : '/api/auth/user/info',
    }

}

export default API;