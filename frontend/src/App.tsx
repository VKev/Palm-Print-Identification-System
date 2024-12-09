import { BrowserRouter, Route, Routes } from "react-router-dom"
import LoginPage from "./pages/LoginPage"
import { ADMIN_DASHBOARD_PAGE, DEFAULT_PAGE, LOGIN_PAGE, NOTFOUND_PAGE, STAFF_DASHBOARD_PAGE, UNAUTHORIZATION_PAGE } from "./config/Constant"
import StaffDashboardPage from "./pages/StaffDashboardPage"
import AdminDashboardPage from "./pages/AdminDashboardPage"
import UnAuthorizationPage from "./pages/UnAuthorizationPage"
// import { UserProvider } from "./context/AuthContext"


function App() {
  return (
    <>
      <BrowserRouter>

        {/* <UserProvider> */}

          <Routes>

            <Route path={DEFAULT_PAGE} element={<LoginPage />} ></Route>

            <Route path={LOGIN_PAGE} element={<LoginPage />}> </Route>

            <Route path={STAFF_DASHBOARD_PAGE} element={<StaffDashboardPage/>}> </Route>

            <Route path={ADMIN_DASHBOARD_PAGE} element={<AdminDashboardPage/>}></Route>

            <Route path={UNAUTHORIZATION_PAGE} element={<UnAuthorizationPage/>}></Route>

            <Route path={NOTFOUND_PAGE}></Route>
            
          </Routes>

        {/* </UserProvider> */}

      </BrowserRouter>

    </>
  )
}

export default App
