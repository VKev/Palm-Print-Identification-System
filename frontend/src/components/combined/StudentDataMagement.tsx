import { useEffect, useState } from "react";
import SearchIcon from '@mui/icons-material/Search';
import SourceIcon from '@mui/icons-material/Source';
// import ImportExportIcon from '@mui/icons-material/ImportExport';
import { Student } from "../../models/Student";
import useAxios from "../../utils/useAxios";
import API from "../../config/API";
import HttpStatus from "../../config/HttpStatus";
import StudentCreationModal from "../single/StudentCreationModal";
import PersonAddAlt1Icon from '@mui/icons-material/PersonAddAlt1';
import StudentDataModal from "../single/StudentDataModal";
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import StudentDeletionModal from "../single/StudentDeletionModal";

export default function StudentDataMagement() {

  const api = useAxios();
  const [currentPage, setCurrentPage] = useState(1);
  const [studentData, setStudentData] = useState<Student[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [isStudentDataModalOpen, setIsStudentDataModalOpen] = useState(false);
  const [selectedStudentForDeletion, setSelectedStudentForDeletion] = useState<Student | null>(null);
  const [isStudentDeletionModalOpen, setIsStudentDeletionModalOpen] = useState(false);

  const handleOpenStudentDeletionModal = (student: Student) => {
    setSelectedStudentForDeletion(student);
    setIsStudentDeletionModalOpen(true);
  }

  const handleCloseStudentDeletionModal = () => {
    setIsStudentDeletionModalOpen(false);
    setSelectedStudentForDeletion(null);
  }

  const handleOpenStudentDataModal = (student: Student) => {
    setSelectedStudent(student);
    setIsStudentDataModalOpen(true);
  };

  const handleCloseStudentDataModal = () => {
    setIsStudentDataModalOpen(false);
    setSelectedStudent(null);
  };

  useEffect(() => {
    const fetchData = async () => {
      const response = await api.get(API.Admin.GET_STUDENT_DATA);
      if (response.status === HttpStatus.OK) {
        setStudentData(response.data);
      }
    }
    fetchData().catch(console.error);
  }, [])

  const handleDeleteStudent = (deletedStudent: Student) => {
    const updatedStudentData = studentData.filter(
      (student) => student.id !== deletedStudent.id
    );
    setStudentData(updatedStudentData);
  };

  const handleCreateStudent = (newStudent: Student) => {
    setStudentData((prevStudents) => [...prevStudents, newStudent]);
  }
  const handleOpen = () => setIsModalOpen(true);
  const handleClose = () => setIsModalOpen(false);


  const studentsPerPage = 7;
  const indexOfLastStudent = currentPage * studentsPerPage;
  const indexOfFirstAccount = indexOfLastStudent - studentsPerPage;
  const currentStudents = studentData.slice(indexOfFirstAccount, indexOfLastStudent);
  const totalPages = Math.ceil(studentData.length / studentsPerPage);

  return (
    <div>
      <div className="mt-3 text-4xl text-center font-medium">Manage Student Data</div>

      <hr className="mt-5 mb-10" />

      <div className='flex justify-between'>

        <div>
          {/* <button
            className='text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2'
          >
            <ImportExportIcon />&nbsp;<span className='align-middle'>Import new data</span>
          </button> */}
          <button onClick={handleOpen}
            className='text-white bg-gradient-to-r from-lime-600 via-lime-600 to-lime-600 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-lime-300 dark:focus:ring-lime-500 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2'
          >
            <PersonAddAlt1Icon />&nbsp;<span className='align-middle'>Create student</span>
          </button>
        </div>

        <div className="flex items-center max-w-md">
          <label htmlFor="simple-search" className="sr-only">Search</label>
          <div className="relative w-full">
            <div className="absolute inset-y-0 start-0 flex items-center ps-3 pointer-events-none">
              <SearchIcon />
            </div>
            <input type="text" id="simple-search" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full ps-10 p-2.5  dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Search student code..." required />
          </div>
          <button type="submit" className="p-2.5 ms-2 text-sm font-medium text-white bg-blue-700 rounded-lg border border-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">
            <svg className="w-4 h-4" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
              <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z" />
            </svg>
            <span className="sr-only">Search</span>
          </button>
        </div>

      </div>


      <div className="mx-20 mt-5">
        <div className="overflow-x-auto shadow-md sm:rounded-lg">
          <table className="min-w-full bg-white">
            <thead className="bg-gray-50">
              <tr>
                <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">ID</th>
                <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Student code</th>
                <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Fullname</th>
                <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Status</th>
                <th className="py-3 px-6 text-xs font-medium tracking-wider text-center text-gray-700 uppercase">Action</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {
                currentStudents.map((student) => (
                  <tr key={student?.id} className="hover:bg-gray-100">
                    <td className="py-4 px-6 text-sm font-medium text-gray-900 text-center">{student.id}</td>
                    <td className="py-4 px-6 text-sm text-gray-500 text-center">{student.studentCode}</td>
                    <td className="py-4 px-6 text-sm text-gray-500 text-center">{student.studentName}</td>
                    <td className={`py-4 px-6 text-sm text-center ${student.isRegistered ? 'text-green-500' : 'text-red-500'}`}>
                      {student.isRegistered ? "✅ Registered" : "❌ Not Registered"}
                    </td>
                    <td className="py-4 px-6 text-sm text-center">

                      <button title="View palm print images"
                        className="text-sm text-white bg-gradient-to-r from-green-400 via-green-500 to-green-500 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-green-300 dark:focus:ring-green-500 font-medium rounded-lg px-3 py-1.5 text-center me-2"
                        onClick={() => handleOpenStudentDataModal(student)}
                      >
                        <span>
                          <SourceIcon fontSize="inherit" />
                        </span>
                      </button>

                      <button title="View palm print images"
                        className="text-sm text-white bg-gradient-to-r from-red-400 via-red-500 to-red-500 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-red-300 dark:focus:ring-red-500 font-medium rounded-lg px-3 py-1.5 text-center me-2"
                        onClick={() => {handleOpenStudentDeletionModal(student)}}
                      >
                        <span>
                          <DeleteForeverIcon fontSize="inherit" />
                        </span>
                      </button>
                    </td>
                  </tr>
                ))
              }
            </tbody>
          </table>
        </div>
        {selectedStudent && (
          <StudentDataModal
            open={isStudentDataModalOpen}
            handleClose={handleCloseStudentDataModal}
            student={selectedStudent}
          />
        )}
      </div>

      <div className="flex justify-center mt-4">
        <button
          className='text-sm text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg px-3 py-1.5 text-center me-2'
          onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        <span className="px-4 py-2 mx-1">{currentPage} / {totalPages}</span>
        <button
          className='text-sm text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg px-3 py-1.5 text-center me-2'
          onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
          disabled={currentPage === totalPages}
        >
          Next
        </button>
      </div>

      <StudentDeletionModal open={isStudentDeletionModalOpen} 
              handleClose={handleCloseStudentDeletionModal} 
              student={selectedStudentForDeletion} 
              handleDeleteStudent={handleDeleteStudent}
      />
      <StudentCreationModal open={isModalOpen} handleClose={handleClose} onStudentCreated={handleCreateStudent} />
    </div>
  );
}
