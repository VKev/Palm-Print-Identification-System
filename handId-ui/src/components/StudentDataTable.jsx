import { DataGrid } from "@mui/x-data-grid";
import Paper from "@mui/material/Paper";
import  { useEffect, useState } from "react";
import { ACCESS_TOKEN_KEY, LOGIN_PAGE } from "../config/Constant";
import StudentAPI from "../service/StudentAPI";

function StudentDataTable() {
  // Tạo state để lưu danh sách sinh viên
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true); // Thêm state loading
  const access_token = localStorage.getItem(ACCESS_TOKEN_KEY);
  useEffect(() => {
    if (access_token) {
      StudentAPI.getStudentData(access_token)
        .then((response) => {
          console.log("API response:", response);
          if (response && Array.isArray(response.data.object)) {
            const formattedData = response.data.object.map((item, index) => ({
              id: index + 1,
              studentId: item.roleNumber,
              fullName: item.fullname,
            }));
            //console.log(formattedData);

            setStudents(formattedData);
          }
          setLoading(false);
        })
        .catch((error) => {
          console.error("Error fetching data:", error); 
          setLoading(false); 
        });
    } else {
      navigator(LOGIN_PAGE);
    }
  }, []);

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
