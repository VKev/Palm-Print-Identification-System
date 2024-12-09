import { UserProfile } from "../../models/User";

type Props = {
    userProfile: UserProfile
}

export default function AccountInfoBlockSidebar({ userProfile }: Props) {
    return (
        <>
            <div className='text-2xl text-center mt-3 mb-3 text-gray-900 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700'>
                {userProfile.fullname}
            </div>
            <div className='text-sm text-center mt-3 mb-3 text-gray-500 rounded-lg dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700'>
                Account: {userProfile.username}
            </div>
        </>
    )
}
