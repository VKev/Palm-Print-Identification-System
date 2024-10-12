/* eslint-disable react/prop-types */
import { createContext, useState, useEffect } from 'react'
import { jwtDecode } from 'jwt-decode';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { AUTH_TOKENS_KEY, LOGIN_PAGE, ROLE_HANDLING_PAGE } from '../config/Constant';
import API from '../config/API';

const BASE_API = API.BASE_API;
const AUTHENTICATION_API = BASE_API + API.Authenticaion.LOGIN;
const AuthContext = createContext();

export default AuthContext;


export const AuthProvider = ({children}) => {

    const navigator = useNavigate()

    let [authTokens, setAuthTokens] = useState(
        ()=> localStorage.getItem(AUTH_TOKENS_KEY) ? JSON.parse(localStorage.getItem(AUTH_TOKENS_KEY)) : null
    )

    let [user, setUser] = useState(
        ()=> localStorage.getItem(AUTH_TOKENS_KEY) ? jwtDecode(localStorage.getItem(AUTH_TOKENS_KEY)) : null
    )

    let [loading, setLoading] = useState(true)

    

    let loginUser = async (e, username, password)=> {
        e.preventDefault()

        const authenticationRequest = {
            username: username,
            password: password
        }

        const response = await axios.post(AUTHENTICATION_API, authenticationRequest)
        
        if(response.status === 200){
            setAuthTokens(response.data)
            localStorage.setItem(AUTH_TOKENS_KEY, JSON.stringify(response.data))
            navigator(ROLE_HANDLING_PAGE)
        }
        else {
            alert('Something went wrong!')
        }
    }


    let logoutUser = () => {
        setAuthTokens(null)
        setUser(null)
        localStorage.removeItem(AUTH_TOKENS_KEY)
        navigator(LOGIN_PAGE)
        window.location.reload();
    }

    let contextData = {
        user:user,
        authTokens:authTokens,
        setAuthTokens:setAuthTokens,
        setUser:setUser,
        loginUser:loginUser,
        logoutUser:logoutUser,
    }


    useEffect(()=> {

        if(authTokens){
            setUser(jwtDecode(authTokens.access_token))
        }
        setLoading(false)


    }, [authTokens, loading])

    return(
        <AuthContext.Provider value={contextData} >
            {loading ? null : children}
        </AuthContext.Provider>
    )
}