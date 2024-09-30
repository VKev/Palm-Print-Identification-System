
import { useState } from "react";
import UserApi from "../service/UserApi";
import { ACCESS_TOKEN_KEY } from "../config/Constant";

export default function UpdateStaffBox({ user }) {

    const [fullname, setFullname] = useState(user.fullname);
    const [phone, setPhone] = useState(user.phone);
    const [updateMessage, setUpdateMessage]  = useState(null)

    const handleFullnameChange = (e) => setFullname(e.target.value)

    const handlePhoneChange = (e) => setPhone(e.target.value)

    const update = () => {
        console.log(fullname, phone);
        if (fullname.trim().length == 0 || phone.trim().length == 0) {
            setUpdateMessage("Fiels are not empty!")
        }
        else {
            let updateStaffRequest = {fullname: fullname, phone: phone}
            UserApi.updateAccount(user.username, updateStaffRequest, localStorage.getItem(ACCESS_TOKEN_KEY)).then(
                response => {
                    setUpdateMessage(response.data.object)
                }
            )
        }
    }

    const discard = () => {
        setFullname(user.fullname)
        setPhone(user.phone)
    }

    return (
        <div className="modal fade" id={user.username} tabIndex="-1" aria-hidden="true">
            <div className="modal-dialog">
                <div className="modal-content">
                    <div className="modal-header">
                        <h1 className="modal-title fs-5" id="exampleModalLabel">Update {user.username}</h1>
                        <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div className="modal-body">
                        <div className="mb-3 mx-5">
                            <label className="form-label">Fullname</label>
                            <input onChange={handleFullnameChange} value={fullname} type="text" className="form-control" aria-describedby="fullname" />
                        </div>
                        <div className="mb-3 mx-5">
                            <label className="form-label">Phone</label>
                            <input onChange={handlePhoneChange} value={phone} type="text" className="form-control" aria-describedby="phone" />
                        </div>
                        <div className="mb-3 mx-5">
                            {
                                updateMessage && <div>{updateMessage}</div>
                            }
                        </div>
                    </div>
                    <div className="modal-footer">
                        <button onClick={discard} type="button" className="btn btn-secondary" data-bs-dismiss="modal">Discard</button>
                        <button onClick={update} type="button" className="btn btn-success">Save changes</button>
                    </div>
                </div>
            </div>
        </div>
    )
}
