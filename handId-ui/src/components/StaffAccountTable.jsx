import { useEffect, useState } from "react"
import UserApi from "../service/UserApi";
import { ACCESS_TOKEN_KEY } from "../config/Constant";
import UpdateStaffBox from "./UpdateStaffBox";


export default function StaffAccountTable() {

    const [staffAccounts, setStaffAccounts] = useState([])

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [fullname, setFullname] = useState('');
    const [phone, setPhone] = useState('');
    const [registerMessage, setRegisterMessage] = useState({ message: null });


    useEffect(() => {
        UserApi.getAllStaffAccounts(localStorage.getItem(ACCESS_TOKEN_KEY)).then(
            response => {
                setStaffAccounts(response.data.object);
            }
        )
            .catch(
                error => {
                    console.log(error);
                }
            )
    }, [])


    const handleUsernameChange = (e) => setUsername(e.target.value)

    const handlePasswordChange = (e) => setPassword(e.target.value)

    const handleFullnameChange = (e) => setFullname(e.target.value)

    const handlePhoneChange = (e) => setPhone(e.target.value)


    const addStaffAccount = () => {
        if (username.trim().length == 0
            || password.trim().length == 0
            || fullname.trim().length == 0
            || phone.trim().length == 0) {
            setRegisterMessage({ message: "Fiels are not empty!" })
        }
        else {
            let registerRequest = { username, password, fullname, phone, isEnable: true };
            setStaffAccounts([...staffAccounts, { username, password, fullname, phone, isEnable: true }])
            UserApi.registerAccount(registerRequest, localStorage.getItem(ACCESS_TOKEN_KEY)).then(
                response => {
                    setRegisterMessage(response.data)
                }
            )
                .then(
                    error => {
                        console.log(error);
                    }
                )
        }
    }

    const disableEnableStaffAccount = (username) => {
        setStaffAccounts(prevAccounts =>
            prevAccounts.map(account =>
                account.username === username ? { ...account, isEnable: !account.isEnable } : account
            )
        );
        UserApi.disableEnableAccount(username, localStorage.getItem(ACCESS_TOKEN_KEY)).then()
        .catch(
            error => {
                console.log(error);
                
            }
        )
    };


    return (
        <div>

            <h2 className="text-center mb-4">Manage Staff Accounts</h2>

            <hr />

            <a data-bs-toggle="collapse" href="#add-account" className="btn btn-success mt-3 mb-4">
                <i className="bi bi-file-earmark-plus">
                </i>&nbsp;&nbsp;Add Account
            </a>

            <div className="collapse mb-5" id="add-account">

                <div className="card card-body">
                    <div className="mb-3 mx-5">
                        <label className="form-label">Username</label>
                        <input onChange={handleUsernameChange} type="text" className="form-control" aria-describedby="username" />
                    </div>
                    <div className="mb-3 mx-5">
                        <label className="form-label">Password</label>
                        <input onChange={handlePasswordChange} type="password" className="form-control" aria-describedby="password" />
                    </div>
                    <div className="mb-3 mx-5">
                        <label className="form-label">Fullname</label>
                        <input onChange={handleFullnameChange} type="text" className="form-control" aria-describedby="fullname" />
                    </div>
                    <div className="mb-3 mx-5">
                        <label className="form-label">Phone</label>
                        <input onChange={handlePhoneChange} type="text" className="form-control" aria-describedby="phone" />
                    </div>
                    {
                        registerMessage.message && <div className="mt-3 mx-5 mb-3">{registerMessage.message}</div>
                    }
                    <div className="mb-3 mx-5">
                        <button onClick={addStaffAccount} className="btn btn-success">Add</button>
                    </div>
                </div>

            </div>

            <table className="table">
                <thead>
                    <tr>
                        <th scope="col">#</th>
                        <th scope="col">Username</th>
                        <th scope="col">Fullname</th>
                        <th scope="col">Phone</th>
                        <th scope="col">Active</th>
                        <th scope="col">Action</th>
                    </tr>
                </thead>
                <tbody>

                    {
                        staffAccounts.map((user, index) => (
                            <tr key={user.username}>
                                <td>{index + 1}</td>
                                <td>{user.username}</td>
                                <td>{user.fullname}</td>
                                <td>{user.phone}</td>
                                <td>
                                    {user.isEnable ? (
                                        <span className="badge rounded-pill bg-success">Active</span>
                                    ) : (
                                        <span className="badge rounded-pill bg-danger">Disabled</span>
                                    )}
                                </td>
                                <td>
                                    <button data-bs-toggle="modal" data-bs-target={"#" + user.username} className="btn btn-link">
                                        <i className="bi bi-pencil-square text-primary"></i>
                                    </button>
                                    <button onClick={() => { disableEnableStaffAccount(user.username) }} className="btn btn-link">
                                        Disable/Enable
                                    </button>
                                </td>
                                <UpdateStaffBox user={user} />
                            </tr>
                        ))
                    }

                </tbody>
            </table>

        </div>
    )
}
