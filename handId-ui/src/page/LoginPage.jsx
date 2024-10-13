import { useContext, useState } from 'react'
import AuthContext from "../context/AuthContext";

export default function LoginPage() {

    const { loginUser } = useContext(AuthContext)
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

        if (authRequest.username && authRequest.password) {
            handleLogin(e)
        }
    }
    
    async function handleLogin(e) {
        e.preventDefault(); 
        try {
            await loginUser(e, username, password); 
        } 
        catch (error) {
            setErrorResponse(error);
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
