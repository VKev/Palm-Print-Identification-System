import { useNavigate } from "react-router-dom";
import LoadEffect from "../components/single/LoadEffect";
import { Role } from "../models/User";
import { useEffect, useState } from "react";
import API from "../config/API";
import useAxios from "../utils/useAxios";
import { UserProfile } from "../models/User";
import { ADMIN_DASHBOARD_PAGE, STAFF_DASHBOARD_PAGE } from "../config/Constant";


export default function RoleRedirector() {

    const [user, setUser] = useState<UserProfile | null>(null)
    const api = useAxios()
    const navigator = useNavigate()

    useEffect(() => {
        const fetchData = async () => {
            const response = await api.get(API.Authenticaion.GET_INFO)
            if (response.status === 200) {
                //console.log(response.data);
                setUser(response.data)
            }
            else {
                alert('Something went wrong!')
            }
        }
        fetchData().catch(console.error)
    }, [])

    if (user) {
        if (user.role === Role.ADMIN) {
            navigator(ADMIN_DASHBOARD_PAGE)
        }
        else if (user.role === Role.STAFF) {
            navigator(STAFF_DASHBOARD_PAGE)
        }
    }


    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <LoadEffect />
        </div>
    )
}
