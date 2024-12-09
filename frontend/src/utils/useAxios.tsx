import axios from 'axios'
import { jwtDecode } from 'jwt-decode'
import dayjs from 'dayjs';
import API from '../config/API';
import { useAuth } from '../context/AuthContext';
import { AUTH_TOKENS_KEY } from '../config/Constant';


const useAxios = () => {
    const { authTokens, setUser, setAuthTokens } = useAuth();

    const AxiosInstance = axios.create({
        baseURL: API.BASE_API,
        headers: {
            Authorization: `Bearer ${authTokens?.access_token}`
        }
    })

    AxiosInstance.interceptors.request.use(async req => {
        const user = jwtDecode(authTokens?.access_token || '{}');
        const isExpired = user.exp ? dayjs.unix(user.exp).diff(dayjs()) < 1 : true;
    
        if (!isExpired) return req;
    
        const response = await axios.post(API.BASE_API + API.Authenticaion.REFRESH_TOKEN, {}, {
            params: {
                refreshToken: authTokens?.refresh_token
            }
        });
        setAuthTokens(response.data)
        setUser(jwtDecode(response.data))
    
        localStorage.setItem(AUTH_TOKENS_KEY, JSON.stringify(response.data));
        req.headers.Authorization = `Bearer ${response.data.access_token}`;
    
        return req
    });

    return AxiosInstance;
}

export default useAxios;