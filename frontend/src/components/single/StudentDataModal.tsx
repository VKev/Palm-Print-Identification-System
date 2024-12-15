import { Modal } from "@mui/material";
import { Student } from "../../models/Student";


type Props = {
    open: boolean;
    handleClose: () => void;
    student: Student;
}

export default function StudentDataModal({ open, handleClose, student }: Props) {
    return (
        <Modal open={open} onClose={handleClose} aria-labelledby="modal-title" aria-describedby="modal-description">
            <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/70">
                <div className="bg-white rounded-2xl shadow-2xl p-8 w-full max-w-3xl mx-auto transform transition-all duration-300 ease-in-out max-h-[90vh] overflow-y-auto">
                    {/* Header */}
                    <div id="modal-title" className="text-3xl font-bold mb-6 text-center text-gray-800 border-b border-gray-200 pb-4">
                        Student Information
                    </div>

                    {/* Content Container */}
                    <div className="space-y-6">
                        {/* Student Details */}
                        <div className="grid grid-cols-1 gap-4 bg-gray-50 p-6 rounded-xl">
                            <div className="flex items-center text-xl text-gray-700">
                                <span className="font-semibold w-32">Code:</span>
                                <span className="bg-blue-50 px-4 py-2 rounded-lg flex-1">{student.studentCode}</span>
                            </div>
                            <div className="flex items-center text-xl text-gray-700">
                                <span className="font-semibold w-32">Name:</span>
                                <span className="bg-blue-50 px-4 py-2 rounded-lg flex-1">{student.studentName}</span>
                            </div>
                            <div className="flex items-center text-xl text-gray-700">
                                <span className="font-semibold w-32">Status:</span>
                                <span className={`px-4 py-2 rounded-lg flex-1 ${
                                    student.isRegistered 
                                    ? "bg-green-100 text-green-800" 
                                    : "bg-red-100 text-red-800"
                                }`}>
                                    {student.isRegistered ? "✅ Registered" : "❌ Not Registered"}
                                </span>
                            </div>
                        </div>

                        {/* Images Section */}
                        <div className="border-t border-gray-200 pt-6">
                            <h3 className="text-xl font-semibold mb-4 text-gray-800">Palm Print Images</h3>
                            <div className="grid grid-cols-3 gap-4">
                                {student.imagePaths.map((path, index) => (
                                    <div key={index} className="aspect-square relative group">
                                        <img 
                                            src={path} 
                                            alt={`Student Image ${index + 1}`} 
                                            className="w-full h-full object-cover rounded-xl shadow-md transition-transform duration-300 hover:scale-105 cursor-pointer" 
                                        />
                                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-xl flex items-center justify-center">
                                            <span className="text-white text-sm">View Full</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Close Button */}
                    <div className="mt-8 border-t border-gray-200 pt-4">
                        <button
                            onClick={handleClose}
                            className="w-full bg-gray-800 text-white py-3 rounded-xl hover:bg-gray-700 transition-colors duration-300"
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </Modal>
    );
}
