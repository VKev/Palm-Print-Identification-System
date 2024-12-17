import { Modal } from "@mui/material";
import { Student } from "../../models/Student";
import useAxios from "../../utils/useAxios";
import API from "../../config/API";
import HttpStatus from "../../config/HttpStatus";
import { toast } from "react-toastify";

type Props = {
    open: boolean;
    handleClose: () => void;
    student: Student | null;
    handleDeleteStudent: (student: Student) => void;
}

export default function StudentDeletionModal({ open, handleClose, student, handleDeleteStudent }: Props) {

    const api = useAxios();

    const deleteStudent = async () => {
        try {
            const response = await api.delete(API.Admin.DELETE_STUDENT + student?.studentCode);
            if (response.status === HttpStatus.OK) {
                toast.success("Student deleted successfully");
                handleClose();
                handleDeleteStudent(student as Student);
            }
        }
        catch(error: any) {
            toast.error(error.response.data.message);
        }
    }

    return (
        <Modal open={open} onClose={handleClose} aria-labelledby="modal-title" aria-describedby="modal-description">
            <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/70">
                <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-3xl mx-auto transform transition-all duration-300 ease-in-out max-h-[90vh] overflow-y-auto">
                    {/* Header */}
                    <div id="modal-title" className="text-3xl font-bold mb-6 text-center text-gray-800 border-b border-gray-200 pb-4">
                        Delete Student {student?.studentCode}
                    </div>

                    {/* Content Container */}
                    <div className="space-y-6">
                        {/* Student Details */}
                        <div className="grid grid-cols-1 gap-4 bg-gray-50 p-6 rounded-xl">
                            <div className="flex items-center text-xl text-gray-700">
                                <span className="font-semibold w-32">Code:</span>
                                <span className="bg-blue-50 px-4 py-2 rounded-lg flex-1">{student?.studentCode}</span>
                            </div>
                            <div className="flex items-center text-xl text-gray-700">
                                <span className="font-semibold w-32">Name:</span>
                                <span className="bg-blue-50 px-4 py-2 rounded-lg flex-1">{student?.studentName}</span>
                            </div>
                            <div className="flex items-center text-xl text-gray-700">
                                <span className="font-semibold w-32">Status:</span>
                                <span className={`px-4 py-2 rounded-lg flex-1 ${student?.isRegistered
                                    ? "bg-green-100 text-green-800"
                                    : "bg-red-100 text-red-800"
                                    }`}>
                                    {student?.isRegistered ? "✅ Registered" : "❌ Not Registered"}
                                </span>
                            </div>
                        </div>

                        <hr className='mb-3' />
                        <button onClick={deleteStudent} className="rounded-md bg-green-500 py-2 px-4 border border-transparent text-center text-sm text-white transition-all shadow-md hover:shadow-lg focus:bg-green-600 focus:shadow-none active:bg-green-600 hover:bg-green-600 active:shadow-none disabled:pointer-events-none disabled:opacity-50 disabled:shadow-none mr-3">
                            Delete
                        </button>
                        <button onClick={handleClose} className='text-white bg-gradient-to-r from-gray-500 via-gray-600 to-gray-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-gray-300 dark:focus:ring-gray-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2'>
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </Modal>
    )
}
