import { useContext, useEffect, useState } from "react";
import StaffAccountTable from "../components/StaffAccountTable";
import StudentDataTable from "../components/StudentDataTable";
import useAxios from "../utils/useAxios";
import AuthContext from "../context/AuthContext";
import API from "../config/API";


export default function AdminPage() {

  const api = useAxios()
  const { logoutUser } = useContext(AuthContext)
  const options = ['staffAcc', 'stuData']
  const [option, setOption] = useState(options[0]);
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


  const handleChooseOption = (opt) => setOption(opt)
  

  return (
    <div>

      <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container-fluid">
          <span className="navbar-brand mb-0 h1">
            Welcome {user.fullname}
          </span>
          <button onClick={logoutUser} className="btn btn-danger" type="button">Logout</button>
          {/* onClick={logout} */}
        </div>
      </nav>

      <div className="container-fluid">

        <div className="row mt-5">

          <div className="col-md-2">
            <h1 className="text-center mb-4">Dashboard</h1>

            <hr />

            <div onClick={() => handleChooseOption(options[0])}
              className={option === options[0] ?
                "text-center bg-primary mt-4" : "text-center border border-primary mt-4 text-dark"
              }
              style={{ color: 'white', padding: 15, fontSize: 18 }}>
              Manage Staff Accounts
            </div>

            <div onClick={() => handleChooseOption(options[1])}
              className={option === options[1] ?
                "text-center bg-primary mt-4" : "text-center border border-primary mt-4 text-dark"
              } style={{ color: 'white', padding: 15, fontSize: 18 }}>
              Manage Student Data
            </div>


          </div>

          <div className="col-md-1"></div>

          <div className="col-md-8">

            {
              option === options[0] ? <StaffAccountTable /> : <StudentDataTable />
            }

          </div>

          <div className="col-md-1"></div>

        </div>

      </div>

    </div>
  )
}
