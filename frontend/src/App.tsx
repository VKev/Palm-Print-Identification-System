import { BrowserRouter, Route, Routes } from "react-router-dom"
import LoginPage from "./pages/LoginPage"
import { ADMIN_DASHBOARD_PAGE, DEFAULT_PAGE, LOGIN_PAGE, STAFF_DASHBOARD_PAGE, UNAUTHORIZATION_PAGE, VERYFYING_PAGE, TEST_VIDEO_WHILE_FRAM_CUT } from "./config/Constant"
import StaffDashboardPage from "./pages/StaffDashboardPage"
import AdminDashboardPage from "./pages/AdminDashboardPage"
import UnAuthorizationPage from "./pages/UnAuthorizationPage"
import { UserProvider } from "./context/AuthContext"
import RoleRedirector from "./helpers/RoleRedirector"
import CamTest from "./pages/CamTest"


function App() {
  return (
    <>
      <BrowserRouter>

        <UserProvider>

          <Routes>

            <Route path={DEFAULT_PAGE} element={<LoginPage />} ></Route>

            <Route path={LOGIN_PAGE} element={<LoginPage />}> </Route>

            <Route path={VERYFYING_PAGE} element={<RoleRedirector/>}></Route>

            <Route path={STAFF_DASHBOARD_PAGE} element={<StaffDashboardPage/>}> </Route>

            <Route path={ADMIN_DASHBOARD_PAGE} element={<AdminDashboardPage/>}></Route>

            <Route path={UNAUTHORIZATION_PAGE} element={<UnAuthorizationPage/>}></Route>

            <Route path={TEST_VIDEO_WHILE_FRAM_CUT} element={<CamTest/>}></Route>
            
          </Routes>

        </UserProvider>

      </BrowserRouter>

    </>
  )
}

export default App
