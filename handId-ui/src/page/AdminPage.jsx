import { useEffect, useState } from "react";
import { ACCESS_TOKEN_KEY, LOGIN_PAGE, REFRESH_TOKEN_KEY } from "../config/Constant";
import AuthenticationAPI from "../service/AuthenticationAPI";
import StaffAccountTable from "../components/StaffAccountTable";
import { useNavigate } from "react-router-dom";
import StudentDataTable from "../components/StudentDataTable";


export default function AdminPage() {

  const [user, setUser] = useState({
    username: '',
    fullname: '',
    role: '',
    phone: ''
  });

  const options = ['staffAcc', 'stuData']
  const [option, setOption] = useState(options[0]);
  const navigator = useNavigate();

  const handleChooseOption = (opt) => {
    setOption(opt)
  }

  const access_token = localStorage.getItem(ACCESS_TOKEN_KEY);
  const refresh_token = localStorage.getItem(REFRESH_TOKEN_KEY);

  useEffect(() => {

    // console.log(access_token);
    if (access_token) {

      AuthenticationAPI.getInfo(access_token)
        .then((response) => {
          setUser(response.data);
        })
        .catch((error) => {
          console.error("Access token invalid:", error);
          if (refresh_token) {
            AuthenticationAPI.refresh(refresh_token).then((response) => {
              localStorage.setItem(ACCESS_TOKEN_KEY, response.data.access_token);
              return AuthenticationAPI.getInfo(response.data.access_token);
            })
              .then((response) => {
                // console.log(response.data);
                setUser(response.data);
              })
              .catch((refreshError) => {
                console.error("Error refreshing token:", refreshError);
                window.location.reload();
              });
          }
          else {
            navigator(LOGIN_PAGE)
          }
        });
    }
    else {
      navigator(LOGIN_PAGE)
    }
  }, []);


  const logout = () => {
    const access_token = localStorage.getItem(ACCESS_TOKEN_KEY);
    AuthenticationAPI.logout(access_token).then(
      (response) => {
        localStorage.removeItem(ACCESS_TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        
      }
    )
      .catch(
        (error) => console.log(error)
      );
    navigator(LOGIN_PAGE);
  }

  return (
    <div>

      <nav className="navbar navbar-expand-lg navbar-light bg-light">
        <div className="container-fluid">
          <span className="navbar-brand mb-0 h1">
            Welcome {user.fullname}
          </span>
          <button onClick={logout} className="btn btn-danger" type="button">Logout</button>
          {/* onClick={logout} */}
        </div>
      </nav>

      <div className="container-fluid">

        <div className="row mt-5">

          <div className="col-md-2">
            <h2 className="text-center mb-4">Dashboard</h2>

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
