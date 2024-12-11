import  { useEffect, useState } from 'react';
import PanToolIcon from '@mui/icons-material/PanTool';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import LogoutIcon from '@mui/icons-material/Logout';
import RecognitionPalmPrint from '../components/combined/RecognitionPalmPrint';
import RegisterPalmPrint from '../components/combined/RegisterPalmPrint';
import HistoryIcon from '@mui/icons-material/History';
import AccountInfoBlockSidebar from '../components/single/AccountInfoBlockSidebar';
import { UserProfile } from '../models/User';
import useAxios from '../utils/useAxios';
import { useAuth } from '../context/AuthContext';
import API from '../config/API';
import HistoryLogging from '../components/combined/HistoryLogging';

type Tab = { id: number; name: string; };

const tabs: Tab[] = [
    { id: 1, name: 'Recognize Palm Print' },
    { id: 2, name: 'Register Palm Print' },
    { id: 3, name: 'History Logs' }
];


const StaffDashboardPage = () => {
    const api = useAxios()
    const { logout } = useAuth();
    const [activeTab, setActiveTab] = useState<Tab>(tabs[0]);
    const [user, setUser] = useState<UserProfile | null>(null);

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

    const liItemStyle = 'mt-5 flex items-center p-3 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group';
    return (
        <div>

            <button data-drawer-target="default-sidebar" data-drawer-toggle="default-sidebar" aria-controls="default-sidebar" type="button" 
                    className="inline-flex items-center p-2 mt-2 ms-3 text-sm text-gray-500 rounded-lg sm:hidden hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-200 dark:text-gray-400 dark:hover:bg-gray-700 dark:focus:ring-gray-600">
                <span className="sr-only">Open sidebar</span>
                <svg className="w-6 h-6" aria-hidden="true" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                    <path clip-rule="evenodd" fill-rule="evenodd" 
                        d="M2 4.75A.75.75 0 012.75 4h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 4.75zm0 10.5a.75.75 0 01.75-.75h7.5a.75.75 0 010 1.5h-7.5a.75.75 0 01-.75-.75zM2 10a.75.75 0 01.75-.75h14.5a.75.75 0 010 1.5H2.75A.75.75 0 012 10z">
                    </path>
                </svg>
            </button>

            <aside id="default-sidebar" className="fixed top-0 left-0 z-40 w-64 h-screen transition-transform -translate-x-full sm:translate-x-0" aria-label="Sidebar">
                <div className="h-full px-3 py-4 overflow-y-auto bg-gray-50 dark:bg-gray-800">
                    <AccountInfoBlockSidebar userProfile={user}></AccountInfoBlockSidebar>
                    <hr />
                    <ul className="space-y-2 font-medium">
                        <li
                            onClick={() => setActiveTab(tabs[0])}
                            className={`cursor-pointer ${activeTab?.id === tabs[0].id ? 'bg-gray-200 rounded-lg' : ''}`}
                        >
                            <a href="" className={liItemStyle}>
                                <PanToolIcon />
                                <span className="ms-3">Recognize Palm Print</span>
                            </a>
                        </li>
                        <li
                            onClick={() => setActiveTab(tabs[1])}
                            className={`cursor-pointer ${activeTab?.id === tabs[1].id ? 'bg-gray-200 rounded-lg' : ''}`}
                        >
                            <a className={liItemStyle}>
                                <PhotoCameraIcon />
                                <span className="ms-3">Register Palm Print</span>
                            </a>
                        </li>
                        <li
                            onClick={() => setActiveTab(tabs[2])}
                            className={`cursor-pointer ${activeTab?.id === tabs[2].id ? 'bg-gray-200 rounded-lg' : ''}`}
                        >
                            <a className={liItemStyle}>
                                <HistoryIcon />
                                <span className="ms-3">History Logs</span>
                            </a>
                        </li>
                        <hr />
                        <li className='cursor-pointer' onClick={logout}>
                            <a className="mt-5 flex items-center p-2 text-red-500 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700 group">
                                <LogoutIcon />
                                <span className="ms-3">Sign out</span>
                            </a>
                        </li>
                    </ul>
                </div>

                <div style={{position: 'absolute', bottom: 0, width: '100%'}}
                    className='text-sm text-center p-3 text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800'
                >
                    Palm Print Recognition System <span className='italic'>v2.0</span>
                </div>
            </aside>


            <div className="p-4 px-10 sm:ml-64">
                
                {
                    activeTab.id === tabs[0].id ? <RecognitionPalmPrint /> : null
                }

                {
                    activeTab.id === tabs[1].id ? <RegisterPalmPrint /> : null
                }

                {
                    activeTab.id === tabs[2].id ? <HistoryLogging user={user} /> : null
                }

            </div>



        </div>
    );
};

export default StaffDashboardPage;
