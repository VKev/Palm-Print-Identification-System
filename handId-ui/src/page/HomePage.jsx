import { useNavigate } from "react-router-dom";
import { ACCESS_TOKEN_KEY, ADMIN_PAGE, LOGIN_PAGE, REFRESH_TOKEN_KEY } from "../config/Constant";
import AuthenticationAPI from "../service/AuthenticationAPI";
import { useEffect, useState } from "react";
import VideoRecorderAI from "../components/VideoRecorderAI";
import StudentAPI from "../service/StudentAPI";

export default function HomePage() {

    const navigator = useNavigate();

    const [user, setUser] = useState({
        username: '',
        fullname: '',
        role: '',
        phone: ''
    });

    const access_token = localStorage.getItem(ACCESS_TOKEN_KEY);
    const refresh_token = localStorage.getItem(REFRESH_TOKEN_KEY);

    useEffect(() => {

        // console.log(access_token);
        if (access_token) {

            AuthenticationAPI.getInfo(access_token)
                .then((response) => {
                    setUser(response.data);
                    if(response.data.role === "ADMIN") navigator(ADMIN_PAGE);
                })
                .catch((error) => {
                    console.error("Access token invalid:", error);
                    if (refresh_token) {
                        AuthenticationAPI.refresh(refresh_token).then((response) => {
                            localStorage.setItem(ACCESS_TOKEN_KEY, response.data.access_token);
                            // console.log("New Access token: ", response.data.access_token);
                            return AuthenticationAPI.getInfo(response.data.access_token);
                        })
                        .then((response) => {
                            //console.log(response.data);
                            setUser(response.data);
                        })
                        .catch((refreshError) => {
                            console.error("Error refreshing token:", refreshError);
                            window.location.reload();
                        });
                    }
                    else {
                        navigator(LOGIN_PAGE)
                    }
                });
        }
        else {
            navigator(LOGIN_PAGE)
        }
    }, []);

    const logout = () => {
        const access_token = localStorage.getItem(ACCESS_TOKEN_KEY);
        AuthenticationAPI.logout(access_token).then(
            (response) => {
                console.log(response);
            }
        )
            .catch(
                (error) => console.log(error)
            );
        navigator(LOGIN_PAGE);
    }

    const options = ['Detect', 'Register']
    const [option, setOption] = useState(options[0]);

    const handleChooseOption = (opt) => {
        setOption(opt)
    }

    const [studentChecking, setStudentChecking] = useState({isExist: false, message: null});
    const [roleNumber, setRoleNumber] = useState('');

    const handleInputRoleNumber = (e) => {
        setRoleNumber(e.target.value)
    }

    const handleCheckRoleNumber = () => {
        StudentAPI.checkRoleNumber(roleNumber, localStorage.getItem(ACCESS_TOKEN_KEY)).then(
            response => {
                setStudentChecking(response.data.object)
            }
        )
    }

    return (
        <div>

            {
                user.fullname &&
                <nav className="navbar navbar-expand-lg navbar-light bg-light">
                    <div className="container-fluid">
                        <span className="navbar-brand mb-0 h1">
                            Welcome {user.fullname}
                        </span>
                        <button onClick={logout} className="btn btn-danger" type="button">Logout</button>
                    </div>
                </nav>
            }


            <div className="container-fluid">
                <div className="row mt-5">
                    <div className="col-md-2">

                        <h2 className="text-center mb-4">Dashboard</h2>

                        <hr />

                        <div onClick={() => handleChooseOption(options[0])} 
                                className= { option === options[0] ?
                                    "text-center bg-primary mt-4" : "text-center border border-primary mt-4 text-dark"
                                } 
                                style={{color: 'white', padding: 15, fontSize: 18, cursor: 'pointer'}}>
                            DETECT HANDID
                        </div>

                        <div onClick={() => handleChooseOption(options[1])} 
                                className= { option === options[1] ?
                                    "text-center bg-primary mt-4" : "text-center border border-primary mt-4 text-dark"
                                } 
                               style={{color: 'white', padding: 15, fontSize: 18, cursor: 'pointer'}}>
                                REGISTER HANDID
                        </div>

                    </div>

                    <div className="col-md-1"></div>

                    <div className="col-md-8">

                        {
                            option === options[0] ?
                                <div>

                                    <h2 className="text-center">Detect HandID</h2>

                                    <hr />

                                    <VideoRecorderAI isOpen={true} roleNumber={null} />

                                </div>
                                :
                                <div>

                                    <h2 className="text-center">Register HandID</h2>

                                    <hr />

                                    <div className="mb-3">
                                        <label htmlFor="student-role-number" className="form-label">Student Role Number</label>
                                        <div className="d-flex">
                                            <input style={{ width: '80%' }} type="text" className="form-control" id="student-role-number" 
                                                    onChange={handleInputRoleNumber}
                                                    aria-describedby="emailHelp" />
                                            <button onClick={handleCheckRoleNumber} type="button" className="btn btn-outline-primary mx-5">Check</button>
                                        </div>
                                        <div id="emailHelp" className="form-text">Input correct student role number to register handID!</div>
                                    </div>

                                    <div className="mt-3 mb-3">
                                        {
                                            studentChecking.message != null &&
                                            <div className={ studentChecking.isExist ? "text-success" : "text-danger" }>
                                                {studentChecking.message}
                                            </div>
                                        }
                                    </div>

                                    <VideoRecorderAI isOpen={studentChecking.isExist} roleNumber={roleNumber}  />

                                </div>
                        }

                    </div>

                    <div className="col-md-1"></div>

                </div>
            </div>

        </div>
    )
}
