import { Modal } from "@mui/material";
import { Student } from "../../models/Student";
import { useState } from "react";
import { toast } from "react-toastify";
import useAxios from "../../utils/useAxios";
import API from "../../config/API";
import HttpStatus from "../../config/HttpStatus";
import { checkStudentCodeFormatDetailed, StudentCodeStatus } from "../../utils/validator";

type Props = {
    open: boolean;
    handleClose: () => void;
    onStudentCreated: (account: Student) => void;
}

export default function StudentCreationModal({ open, handleClose, onStudentCreated }: Props) {

    const api = useAxios();
    const [studentCode, setStudentCode] = useState<string>('');
    const [studentName, setStudentName] = useState<string>('');

    const createStudent = async () => {
        const studentCreationRequest = {
            studentCode: studentCode,
            studentName: studentName,
        }
        if (!studentCode || !studentName) {
            toast.error('All fields are required!');
            return;
        }
        const studentCodeStatus: StudentCodeStatus = checkStudentCodeFormatDetailed(studentCode);
        if (studentCodeStatus.isValid) {
            try {
                const response = await api.post(API.Admin.CREATE_STUDENT, studentCreationRequest);
                if (response.status === HttpStatus.OK) {
                    toast.success('Student created successfully!');
                    //console.log(response.data);
                    onStudentCreated(response.data);
                    handleClose();
                }
            }
            catch (error: any) {
                toast.error(error.response.data.message);
            }
        }
        else {
            toast.error(studentCodeStatus.error);
        }
    }

    return (
        <Modal open={open} onClose={handleClose} aria-labelledby="modal-title" aria-describedby="modal-description">
            <div className="fixed inset-0 flex items-center justify-center z-50">
                <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md mx-auto">
                    <div id="modal-title" className="text-2xl font-semibold mb-4">
                        Add student
                    </div>
                    <hr className='mb-3' />

                    <div id="modal-description" className="mb-4">
                        <div className="py-4">
                            <span className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Student code</span>
                            <input onChange={(e) => setStudentCode(e.target.value)}
                                type="text"
                                className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                        </div>

                        <div className="py-4">
                            <span className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">Student name</span>
                            <input onChange={(e) => setStudentName(e.target.value)}
                                type="text"
                                className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                            />
                        </div>
                    </div>

                    <hr className='mb-3' />
                    <button onClick={createStudent} className="rounded-md bg-green-500 py-2 px-4 border border-transparent text-center text-sm text-white transition-all shadow-md hover:shadow-lg focus:bg-green-600 focus:shadow-none active:bg-green-600 hover:bg-green-600 active:shadow-none disabled:pointer-events-none disabled:opacity-50 disabled:shadow-none mr-3">
                        Save
                    </button>
                    <button onClick={handleClose} className='text-white bg-gradient-to-r from-red-500 via-red-600 to-red-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-red-300 dark:focus:ring-red-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2'>
                        Close
                    </button>
                </div>
            </div>
        </Modal>
    )
}
