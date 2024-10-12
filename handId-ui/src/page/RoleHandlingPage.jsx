
import { useEffect, useState } from 'react'
import LoadEffect from '../components/LoadEffect'
import useAxios from '../utils/useAxios';
import API from '../config/API';
import { useNavigate } from 'react-router-dom';
import { ADMIN_PAGE, HOME_PAGE } from '../config/Constant';

export default function RoleHandlingPage() {

    const api = useAxios()
    const navigator = useNavigate()
    const [user, setUser] = useState();

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

    if (user) {
        if (user?.role == 'ADMIN') {
            navigator(ADMIN_PAGE)
        }
        else if (user?.role == 'USER') {
            navigator(HOME_PAGE)
        }
    }

    return (
        <div>
            <LoadEffect />
        </div>
    )
}
