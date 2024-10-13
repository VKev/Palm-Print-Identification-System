import axios from 'axios'
import { jwtDecode } from 'jwt-decode'
import dayjs from 'dayjs';
import { useContext } from 'react';
import API from '../config/API';
import AuthContext from '../context/AuthContext';
import { AUTH_TOKENS_KEY } from '../config/Constant';


const BASE_API = API.BASE_API;
const REFRESH_TOKEN_API = BASE_API + "/api/users/refresh-token"


const useAxios = () => {
    const { authTokens, setUser, setAuthTokens } = useContext(AuthContext)

    const AxiosInstance = axios.create({
        baseURL: BASE_API,
        headers: {
            Authorization: `Bearer ${authTokens?.access_token}`
        }
    })

    AxiosInstance.interceptors.request.use(async req => {
        const user = jwtDecode(authTokens?.access_token || '{}');
        const isExpired = dayjs.unix(user.exp).diff(dayjs()) < 1;
    
        if (!isExpired) return req;
    
        const response = await axios.post(REFRESH_TOKEN_API, {}, {
            params: {
                refreshToken: authTokens.refresh_token
            }
        });
        setAuthTokens(response.data.object)
        setUser(jwtDecode(response.data.object))
    
        localStorage.setItem(AUTH_TOKENS_KEY, JSON.stringify(response.data.object));
        req.headers.Authorization = `Bearer ${response.data.object.access_token}`;
    
        return req
    });

    return AxiosInstance;
}

export default useAxios;