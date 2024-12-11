import { Modal } from "@mui/material";
import { useState } from "react";
import { toast } from "react-toastify";
import useAxios from "../../utils/useAxios";
import API from "../../config/API";
import HttpStatus from "../../config/HttpStatus";
import { Account } from "../../models/User";

type Props = {
    open: boolean;
    handleClose: () => void;
    onAccountCreated: (account: Account) => void; 
}

export default function AccountCreationModal({ open, handleClose, onAccountCreated }: Props) { 

    const api = useAxios();
    const [username, setUsername] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [repeatPassword, setRepeatPassword] = useState<string>('');   
    const [fullname, setFullname] = useState<string>('');

    const createAccount = async () => {
        const registerRequest = {
            username: username,
            password: password,
            fullname: fullname,
        }
        if (!username || !password || !fullname) {
            toast.error('All fields are required!');
            return;
        }
        else if (repeatPassword !== password) {
            toast.error('Repeat password does not match!');
            return;
        }
        const response = await api.post(API.Admin.REGISTER_STAFF_ACCOUNT, registerRequest);
        if (response.status === HttpStatus.OK) {
            toast.success('Account created successfully!');
            onAccountCreated(response.data.createdUser); 
            handleClose();
        }
    }
    
    return (
        <Modal open={open} onClose={handleClose} aria-labelledby="modal-title" aria-describedby="modal-description">
            <div className="fixed inset-0 flex items-center justify-center z-50">
                <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md mx-auto">
                    <div id="modal-title" className="text-2xl font-semibold mb-4">
                        Create new staff
                    </div>
                    <hr className='mb-3' />

                    <div id="modal-description" className="mb-4">
                        <div className="py-4">
                            <span className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Username</span>
                            <input onChange={(e) => setUsername(e.target.value)}
                                type="text"
                                className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                        </div>

                        <div className="py-4">
                            <span className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Password</span>
                            <input onChange={(e) => setPassword(e.target.value)}
                                type="password"
                                className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                        </div>

                        <div className="py-4">
                            <span className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Repeat password</span>
                            <input onChange={(e) => setRepeatPassword(e.target.value)}
                                type="password"
                                className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                        </div>

                        <div className="py-4">
                            <span className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Fullname</span>
                            <input onChange={(e) => setFullname(e.target.value)}
                                type="text"
                                className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                        </div>

                        <div className="py-4">
                            <label htmlFor="countries" className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Select role</label>
                            <select id="countries" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500">
                                <option selected value="STAFF">Staff</option>
                            </select>
                        </div>
                    </div>

                    <hr className='mb-3' />
                    <button onClick={createAccount} className="rounded-md bg-green-500 py-2 px-4 border border-transparent text-center text-sm text-white transition-all shadow-md hover:shadow-lg focus:bg-green-600 focus:shadow-none active:bg-green-600 hover:bg-green-600 active:shadow-none disabled:pointer-events-none disabled:opacity-50 disabled:shadow-none mr-3">
                        Save
                    </button>
                    <button onClick={handleClose} className='text-white bg-gradient-to-r from-red-500 via-red-600 to-red-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-red-300 dark:focus:ring-red-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2'>
                        Close
                    </button>
                </div>
            </div>
        </Modal>
    );
}
