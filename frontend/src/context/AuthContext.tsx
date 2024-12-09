import { createContext, useContext, useEffect, useState } from 'react';
import { AuthTokens, UserProfile } from '../models/User';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import API from '../config/API';
import HttpStatus from '../config/HttpStatus';
import { toast } from 'react-toastify';

const AUTH_TOKENS_KEY = 'authTokens';
const USER_KEY = 'user';

type UserContextType = {
    user: UserProfile | null,
    authTokens: AuthTokens | null,
    //registerUser: (username: string, password: string) => void,
    loginUser: (username: string, password: string) => void,
    logout: () => void,
    isLoggedIn: () => boolean,
    setUser: (user: UserProfile) => void,
    setAuthTokens: (authTokens: AuthTokens) => void
}

const UserContext = createContext<UserContextType>({} as UserContextType);


export const UserProvider = ({ children }: { children: React.ReactNode }) => {

    const navigate = useNavigate();
    const [authTokens, setAuthTokens] = useState<AuthTokens | null>(null);
    const [user, setUser] = useState<UserProfile | null>(null);
    const [isReady, setIsReady] = useState<boolean>(false);

    useEffect(() => {
        const user = localStorage.getItem(USER_KEY);
        const authTokens = localStorage.getItem(AUTH_TOKENS_KEY);
        if (user && authTokens) {
            setUser(JSON.parse(user));
            setAuthTokens(JSON.parse(authTokens));
        }
        setIsReady(true);
    }, []);


    const loginUser = async (username: string, password: string) => {
        const authenticationRequest = {
            username: username,
            password: password
        }
        try {
            const response = await axios.post(API.BASE_API + API.Authenticaion.LOGIN, authenticationRequest);
            if (response.status === HttpStatus.OK) {
                setAuthTokens(response.data);
                localStorage.setItem(AUTH_TOKENS_KEY, JSON.stringify(response.data));

            }
            else if (response.status === HttpStatus.UNAUTHORIZED) {
                toast.error(response.data.message);
            }
        }
        catch (error) {
            console.log(error);
            toast.error('Invalid username or password');
        }
    }


    const logout = () => {
        localStorage.removeItem(AUTH_TOKENS_KEY);
        localStorage.removeItem(USER_KEY);
        setUser(null);
        setAuthTokens(null);
        navigate('/login');
    }


    const isLoggedIn = (): boolean => {
        return !!user;
    }

    return (
        <UserContext.Provider value={{
            loginUser, user, authTokens, logout, isLoggedIn, setUser, setAuthTokens
        }}
        >
            {isReady ? children : null}
        </UserContext.Provider>
    )
}

export const useAuth = () => useContext(UserContext);
