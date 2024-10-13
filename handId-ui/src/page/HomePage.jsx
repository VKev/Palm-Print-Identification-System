
import { useContext, useEffect, useState } from "react";
import VideoRecorderAI from "../components/VideoRecorderAI";
import useAxios from "../utils/useAxios";
import API from "../config/API";
import AuthContext from "../context/AuthContext";
import VideoDetector from "../components/VideoDetector";

export default function HomePage() {

    const api = useAxios()
    const  { logoutUser } = useContext(AuthContext)
    const [user, setUser] = useState({
        username: '',
        fullname: '',
        role: '',
        phone: ''
    });

    useEffect(() => {
        const fetchData = async () => {
            const response = await api.get(API.Authenticaion.GET_INFO)
            if (response.status == 200) {
                setUser(response.data)
            }
            else {
                alert('Something went wrong!')
            }
        }
        fetchData().catch(console.error)
    }, [])


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
        const fetchData = async () => {
            const response = await api.get(API.Student.STUDENT_CHECKING + roleNumber)
            if (response.status == 200) {
                setStudentChecking(response.data.object)
            }
        }
        fetchData().catch(console.error)
    }

    return (
        <div>

            {
                user.fullname &&
                <nav className="navbar navbar-expand-lg navbar-light bg-light">
                    <div className="container-fluid">
                        <span style={{ fontSize: '24px' }} className="navbar-brand mb-0 h1">
                            Welcome, {user.fullname}
                        </span>
                        <button style={{ fontSize: '18px' }} onClick={logoutUser} className="btn btn-danger" type="button">Logout</button>
                    </div>
                </nav>
            }


            <div className="container-fluid">
                <div className="row mt-5">
                    <div className="col-md-2">

                        <h1 className="text-center mb-4">Dashboard</h1>

                        <hr />

                        <div onClick={() => handleChooseOption(options[0])} 
                                className= { option === options[0] ?
                                    "text-center bg-primary mt-4" : "text-center border border-primary mt-4 text-dark"
                                } 
                                style={{color: 'white', padding: 15, fontSize: 18, cursor: 'pointer'}}>
                            DETECT PALMID
                        </div>

                        <div onClick={() => handleChooseOption(options[1])} 
                                className= { option === options[1] ?
                                    "text-center bg-primary mt-4" : "text-center border border-primary mt-4 text-dark"
                                } 
                               style={{color: 'white', padding: 15, fontSize: 18, cursor: 'pointer'}}>
                                REGISTER PALMID
                        </div>

                    </div>

                    <div className="col-md-1"></div>

                    <div className="col-md-8">

                        {
                            option === options[0] ?
                                <div>

                                    <h1 className="text-center">Detect PalmID</h1>

                                    <hr />

                                    <VideoDetector/>

                                </div>
                                :
                                <div>

                                    <h1 className="text-center">Register PalmID</h1>

                                    <hr />

                                    <div className="mb-3">
                                        <label style={{ fontSize: '34px' }} htmlFor="student-role-number" className="form-label">Student Role Number</label>
                                        <div className="d-flex">
                                            <input style={{ width: '80%', fontSize: '26px' }} type="text" className="form-control" id="student-role-number" 
                                                    onChange={handleInputRoleNumber}
                                                    aria-describedby="emailHelp" />
                                            <button style={{ fontSize: '22px' }} onClick={handleCheckRoleNumber} type="button" className="btn btn-outline-primary mx-5">Check</button>
                                        </div>
                                        <div style={{ fontSize: '34px' }}  className="form-text">Input correct student role number to register palmID!</div>
                                    </div>

                                    <div className="mt-3 mb-3">
                                        {
                                            studentChecking.message != null &&
                                            <div style={{ fontSize: '32px' }} className={ studentChecking.isExist ? "text-success" : "text-danger" }>
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
