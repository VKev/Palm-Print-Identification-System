import { DataGrid } from "@mui/x-data-grid";
import Paper from "@mui/material/Paper";
import  { useEffect, useState } from "react";
import useAxios from "../utils/useAxios";
import API from "../config/API";

function StudentDataTable() {
  
  const api = useAxios();
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);


  useEffect(() => {
    const fetchData = async () => {
      const response = await api.get(API.Student.GET_STUDENT_DATA)
      if (response.status == 200) {
        setStudents(response.data.object)
        setLoading(false)
      }
    }
    fetchData().catch(console.error)
  }, [])
  

  const columns = [
    { field: "id", headerName: "ID", width: 70 },
    { field: "studentId", headerName: "Student ID", width: 130 },
    { field: "fullName", headerName: "Full Name", width: 390 },
  ];

  const paginationModel = { page: 0, pageSize: 5 };

  return (
    <Paper style={{ height: 400, width: "100%" }}>
      {loading ? (
        <div>Loading...</div> 
      ) : (
        <DataGrid
          rows={students} 
          columns={columns}
          initialState={{ pagination: { paginationModel } }}
          pageSizeOptions={[5, 10]}
          style={{ border: 0 }}
        />
      )}
    </Paper>
  );
}

export default StudentDataTable;
