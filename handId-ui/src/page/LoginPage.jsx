import { useNavigate } from "react-router-dom";
import { ACCESS_TOKEN_KEY, HOME_PAGE, REFRESH_TOKEN_KEY } from "../config/Constant";
import AuthenticationAPI from "../service/AuthenticationAPI";
import { useState } from 'react'

export default function LoginPage() {

    const navigator = useNavigate();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [errorResponse, setErrorResponse] = useState({
        httpStatus: '',
        message: ''
    });

    // Function handler
    const handleUsername = (e) => setUsername(e.target.value);

    const handlePassword = (e) => setPassword(e.target.value);

    const submit = (e) => {
        e.preventDefault();

        const authRequest = {
            username, password
        }

        //console.log(authRequest);

        if (authRequest.username && authRequest.password) {
            AuthenticationAPI.authenticate(authRequest).then(
                (response) => {
                    localStorage.setItem(ACCESS_TOKEN_KEY, response.data.access_token);
                    localStorage.setItem(REFRESH_TOKEN_KEY, response.data.refresh_token);
                    navigator(HOME_PAGE);
                }
            )
            .catch(
                (error) => {
                    setErrorResponse(error.response.data);
                    console.log(error.response.data);
                }
            )
        }
    }


    return (
        <div className="container mt-5">
            <div className="row justify-content-center">
                <div className="col-md-4">
                    <h3 className="text-center mb-4">Login</h3>
                    <form onSubmit={submit}>
                        <div className="mb-3">
                            <label className="form-label">Username</label>
                            <input type="text" className="form-control" id="username"
                                onChange={handleUsername}
                                placeholder="Enter your username" required
                            />
                        </div>
                        <div className="mb-3">
                            <label className="form-label">Password</label>
                            <input type="password" className="form-control" id="password"
                                onChange={handlePassword}
                                placeholder="Enter your password" required />
                        </div>
                        {
                            errorResponse.message && <div className='text-danger mb-3'>{errorResponse.message}</div>
                        }
                        <div className='d-flex'>
                            <button style={{width: '100%'}} type="submit" className="btn btn-primary">Login</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    )
}
