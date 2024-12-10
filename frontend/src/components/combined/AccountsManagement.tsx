import { useEffect, useState } from 'react';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import SearchIcon from '@mui/icons-material/Search';
import EditIcon from '@mui/icons-material/Edit';
import BlockIcon from '@mui/icons-material/Block';
import { Account } from '../../models/User';
import useAxios from '../../utils/useAxios';
import API from '../../config/API';
import { toast } from 'react-toastify';
import AccountCreationModal from '../single/AccountCreationModal';

export default function AccountsManagement() {

  const api = useAxios();
  const [currentPage, setCurrentPage] = useState(1);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [accounts, setAccounts] = useState<Account[]>([]); 

  useEffect(() => {
    const fetchData = async () => {
      const response = await api.get(API.Admin.GET_STAFF_ACCOUNTS);
      if (response.status === 200) {
        setAccounts(response.data);
      } 
      else {
        toast.error('Something went wrong during fetch accounts!');
      }
    }
    fetchData().catch(console.error);
  }, [])

  const handleOpen = () => setIsModalOpen(true);
  const handleClose = () => setIsModalOpen(false);

  const handleAccountCreated = (newAccount: Account) => {
    setAccounts((prevAccounts) => [...prevAccounts, newAccount]);
  };

  const accountsPerPage = 15;
  const indexOfLastAccount = currentPage * accountsPerPage;
  const indexOfFirstAccount = indexOfLastAccount - accountsPerPage;
  const currentAccounts = accounts.slice(indexOfFirstAccount, indexOfLastAccount);
  const totalPages = Math.ceil(accounts.length / accountsPerPage);

  
  return (
    <div>
      <div className="mt-3 text-4xl text-center font-medium">Manage Staff Accounts</div>

      <hr className="mt-5 mb-10" />

      <div className='flex justify-between'>
        <button onClick={handleOpen}
          className='text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2'
        >
          <PersonAddIcon />&nbsp; <span className='align-middle'>Add New Staff</span>
        </button>

        <div className="flex items-center max-w-md">
          <label htmlFor="simple-search" className="sr-only">Search</label>
          <div className="relative w-full">
            <div className="absolute inset-y-0 start-0 flex items-center ps-3 pointer-events-none">
              <SearchIcon />
            </div>
            <input type="text" id="simple-search" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full ps-10 p-2.5  dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Search username..." required />
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
        <table className="min-w-full bg-white">
          <thead>
            <tr>
              <th className="py-2 px-4 border-b">ID</th>
              <th className="py-2 px-4 border-b">Username</th>
              <th className="py-2 px-4 border-b">Fullname</th>
              <th className="py-2 px-4 border-b">Role</th>
              <th className="py-2 px-4 border-b">Status</th>
              <th className="py-2 px-4 border-b">Action</th>
            </tr>
          </thead>
          <tbody>
            {
            currentAccounts.map((account) => (
              <tr key={account.id}>
                <td className="py-2 px-4 border-b text-center">{account.id}</td>
                <td className="py-2 px-4 border-b text-center">{account.username}</td>
                <td className="py-2 px-4 border-b text-center">{account.fullname}</td>
                <td className="py-2 px-4 border-b text-center">{account.role}</td>
                <td className="py-2 px-4 border-b text-center">
                  {
                    account.isEnable ?
                    <span className="bg-green-100 text-green-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded-full dark:bg-green-900 dark:text-green-300">
                      Active
                    </span> :
                    <span className="bg-red-100 text-red-800 text-xs font-medium me-2 px-2.5 py-0.5 rounded-full dark:bg-red-900 dark:text-red-300">
                      Disable
                    </span>
                  }
                </td>
                <td className="py-2 px-4 border-b text-center">
                  <button title='Edit' className='text-sm text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg px-3 py-1.5 text-center me-2'>
                    <EditIcon fontSize='inherit'/>
                  </button>
                  <button title='Disable' className='text-sm text-white bg-gradient-to-r from-red-500 via-red-600 to-red-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-red-300 dark:focus:ring-red-800 font-medium rounded-lg px-3 py-1.5 text-center me-2'>
                    <BlockIcon fontSize='inherit'/>
                  </button>
                </td>
              </tr>
            ))
            }
          </tbody>
        </table>
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

      <AccountCreationModal open={isModalOpen} handleClose={handleClose} onAccountCreated={handleAccountCreated} />

    </div>
  );
};
