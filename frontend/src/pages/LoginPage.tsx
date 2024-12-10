import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { toast } from "react-toastify";

export default function LoginPage() {

    const { loginUser } = useAuth();
    const [username, setUsername] = useState<string>('');
    const [password, setPassword] = useState<string>('');

    const handleLogin = () => {
        if (!username) {
            toast.error("Username is required");    
        }
        if (!password) {
            toast.error("Password is required");
        }
        if (username && password) {
            loginUser(username, password);
        }
    }


    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div
                className="relative flex flex-col m-6 space-y-8 bg-white shadow-2xl rounded-2xl md:flex-row md:space-y-0"
            >
                {/* left side */}
                <div className="flex flex-col justify-center p-8 md:p-14">
                    <span className="mb-3 text-4xl font-bold text-center">Sign In</span>
                    <span className="font-light text-center text-gray-700 mb-8">
                        Palm Print Recognition System 
                    </span>
                    <div className="py-4">
                        <span className="mb-2 text-md">Username</span>
                        <input onChange={(e) => setUsername(e.target.value)}
                            type="text"
                            className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                        />
                    </div>
                    <div className="py-4">
                        <span className="mb-2 text-md">Password</span>
                        <input onChange={(e) => setPassword(e.target.value)}
                            type="password"
                            name="pass"
                            className="w-full p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                        />
                    </div>
                    <div className="flex justify-between w-full py-4">
                        <div className="mr-24">
                            <input type="checkbox" name="ch" id="ch" className="mr-2" />
                            <span className="text-md">Remember me</span>
                        </div>
                        <a className="font-bold text-md">Forgot password?</a>
                    </div>
                    <button onClick={handleLogin}
                        className="w-full text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2"
                    >
                        Sign in
                    </button>
                    {/* <button
                        className="w-full border border-gray-300 text-md p-2 rounded-lg mb-6"
                    >
                        <img src="/src/assets/images/google.png" className="w-6 h-6 inline mr-2" />
                        Sign in with Google
                    </button> */}
                    
                </div>
                {/* right side */}
                <div className="relative">
                    <img
                        src="/src/assets/images/cyber-palm.jpg"
                        alt="img"
                        className="w-[400px] h-full hidden rounded-r-2xl md:block object-cover"
                    />
                    {/* text on image  */}
                    <div className="absolute hidden bottom-10 right-6 p-6 bg-white bg-opacity-30 backdrop-blur-sm rounded drop-shadow-lg md:block">
                        <span className="text-white text-xl">
                            - Palm Print Recognition System -
                        </span>
                    </div>
                </div>
            </div>
        </div>
    )
}
