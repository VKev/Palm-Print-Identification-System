import { IconButton } from "@mui/material";
import HandRecognizer from "../single/HandRecognizer";
import VideocamIcon from '@mui/icons-material/Videocam';
import VideocamOffIcon from '@mui/icons-material/VideocamOff';
import { useState } from "react";
import { toast } from "react-toastify";
import useAxios from "../../utils/useAxios";
import API from "../../config/API";
import HttpStatus from "../../config/HttpStatus";
import { FileType, ImageFile, StudentValidationResponse } from "../../models/Student";
import { CameraMode, RegistrationPhases } from "../../models/PalmPrint";
import { base64ToFile } from "../../utils/fileUtil";
import { v4 as uuidv4 } from 'uuid';


export default function RegisterPalmPrint() {

    const api = useAxios();
    const [studentCode, setStudentCode] = useState<string | null>(null);
    const [studentValidationResponse, setStudentCodeResponse] = useState<StudentValidationResponse | null>(null);
    const [cameraOn, setCameraOn] = useState(false);
    const [selectedImages, setSelectedImages] = useState<ImageFile[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [registerPhase, setRegisterPhase] = useState<number>(RegistrationPhases.BACKGROUND_CUT);

    const toggleCamera = () => setCameraOn(!cameraOn);
    

    const validateStudentCode = async () => {
        try {
            if (studentCode?.trim()) {
                const response = await api.get(API.Staff.VALIDATE_STUDENT_CODE + studentCode);
                if (response.status === HttpStatus.OK) {
                    setStudentCodeResponse(response.data);
                    toast.success(response.data.message);
                }
                //console.log(response.data);
            }
            else {
                setStudentCodeResponse(null);
                toast.error('Student code is required!')
            } 
        }
        catch (error: any) {
            if (error.response.status === HttpStatus.NOT_FOUND) {
                setStudentCodeResponse(null);
                toast.error(error.response.data.message);
                console.log(error.response.data);
            }
            else {
                setStudentCodeResponse(null);
                toast.error('Something went wrong! Try again later.');
            }
        }
        
    }

    // Send selected images to server
    const sendImagesToServer = async () => {
        switch (registerPhase) {
            case RegistrationPhases.BACKGROUND_CUT:
                sendFramesToCutBackground();
                break;
            case RegistrationPhases.ROI_CUT:
                sendBackgroundCutImagesToRoiCut();
                break;
            case RegistrationPhases.REGISTER_INFERENCE:
                registerInference();
                break;
            default:
                toast.error('Invalid registration phase');
                break;
        }
    }

    const sendFramesToCutBackground = async () => {
        setIsLoading(true);
        if (selectedImages.length === 0) return;
        const formData = new FormData();
        const files: File[] = [];
        selectedImages.forEach((image) => {
            if(!image.isSelected && image.file) {
                files.push(image.file);
            }
        });
        
        files.forEach(file => formData.append('images', file));
        try {
            const response = await api.post(API.Staff.UPLOAD_PALM_PRINT_FRAME + studentCode, formData);
            if (response.status === HttpStatus.OK) {
                toast.success(response.data.message);
                setSelectedImages(response.data.images.map(
                    (image: string) => ({ file: null, base64: image, isSelected: false, type: FileType.BASE64 }))
                );
                setRegisterPhase(RegistrationPhases.ROI_CUT);
            }
        }
        catch (error: any) {
            if (error.response.status === HttpStatus.BAD_REQUEST) {
                toast.error(error.response.data.message);
            }
            else {
                toast.error('Something went wrong while cutting background! Try again later.');
            }
        }
        finally {
            setIsLoading(false);
        }
    }

    const sendBackgroundCutImagesToRoiCut = async () => {
        setIsLoading(true);
        const formData = new FormData();
        const files: File[] = [];
        selectedImages.forEach((image) => {
            if(!image.isSelected && image.base64) {
                files.push(base64ToFile(image.base64, uuidv4()+'.png'));
            }
        });

        files.forEach(file => formData.append('images', file));
        try {
            const response = await api.post(API.Staff.UPLOAD_BACKGROUND_CUT_FRAME, formData);
            if (response.status === HttpStatus.OK) {
                toast.success(response.data.message);
                setSelectedImages(response.data.images.map(
                    (image: string) => ({ file: null, base64: image, isSelected: false, type: FileType.BASE64 }))
                );
                setRegisterPhase(RegistrationPhases.REGISTER_INFERENCE);
            }
        }
        catch (error: any) {
            if (error.response.status === HttpStatus.BAD_REQUEST) {
                toast.error(error.response.data.message);
            }
            else {
                toast.error('Something went wrong while cutting roi! Try again later.');
            }
        }
        finally {
            setIsLoading(false);
        }
    }

    const registerInference = async () => {
        setIsLoading(true);
        const formData = new FormData();
        const files: File[] = [];
        selectedImages.forEach((image) => {
            if(!image.isSelected && image.base64) {
                files.push(base64ToFile(image.base64, uuidv4()+'.png'));
            }
        });

        files.forEach(file => formData.append('images', file));
        try {
            const response = await api.post(API.Staff.REGISTER_INFERENCE + studentCode, formData);
            if (response.status === HttpStatus.OK) {
                //toast.success(response.data);
                console.log(response.data);
                setRegisterPhase(RegistrationPhases.BACKGROUND_CUT);
            }
        }
        catch (error: any) {
            if (error.response.status === HttpStatus.BAD_REQUEST) {
                toast.error(error.response.data.message);
            }
            else {
                toast.error('Something went wrong while register inferences! Try again later.');
            }
        }
        finally {
            setIsLoading(false);
        }
    }

    const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            const filesArray = Array.from(event.target.files);
            const imageFiles = filesArray.map(file => ({ file, base64: null, isSelected: false, type: FileType.FILE }));
            setSelectedImages(imageFiles);
        }
    };

    function handleImageClick(image: ImageFile): void {
        const newSelectedImages = selectedImages.map((img) => {
            if (img === image) {
                return { ...img, isSelected: !img.isSelected };
            }
            return img;
        });
        setSelectedImages(newSelectedImages);
    }

    return (
        <div>
            <div className="mt-3 text-4xl text-center font-medium">Register Palm Print</div>
            <hr className="mt-5 mb-10" />
            {/* <div className="grid grid-cols-2 gap-4"> */}

            <div>
                <div className="mb-5">
                    <div className="text-lg mb-2">Enter student code for register<span className="text-red-500">*</span></div>
                    <div className="flex">
                        <input onChange={(e) => setStudentCode(e.target.value)}
                            type="text" placeholder="Enter student code ...."
                            className="w-4/6 p-2 border border-gray-300 rounded-md placeholder:font-light placeholder:text-gray-500"
                        />
                        <button onClick={validateStudentCode}
                            className="w-1/6 ml-3 text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2">
                            Check
                        </button>
                    </div>
                </div>
                {
                    cameraOn && <HandRecognizer 
                        width={"200%"} maxWidth={"1000px"} 
                        cameraMode={CameraMode.REGISTRATION} 
                        studentCode={studentCode}
                    />
                }

                {
                    studentValidationResponse?.statusResult && !studentValidationResponse.isRegistered &&
                    <>
                        <div>
                            <IconButton onClick={toggleCamera} style={{ color: cameraOn ? 'inherit' : 'red', fontSize: '3rem' }} title="Open/close camera">
                                {cameraOn ? <VideocamIcon style={{ fontSize: 'inherit' }} /> : <VideocamOffIcon style={{ fontSize: 'inherit' }} />}
                            </IconButton>
                        </div>
                        <div className="mb-5">
                            <div className="w-full py-9 bg-gray-50 rounded-2xl border border-gray-300 gap-3 grid border-dashed">
                                <div className="grid gap-1">
                                    <svg className="mx-auto" width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <g id="File">
                                            <path id="icon" d="M31.6497 10.6056L32.2476 10.0741L31.6497 10.6056ZM28.6559 7.23757L28.058 7.76907L28.058 7.76907L28.6559 7.23757ZM26.5356 5.29253L26.2079 6.02233L26.2079 6.02233L26.5356 5.29253ZM33.1161 12.5827L32.3683 12.867V12.867L33.1161 12.5827ZM31.8692 33.5355L32.4349 34.1012L31.8692 33.5355ZM24.231 11.4836L25.0157 11.3276L24.231 11.4836ZM26.85 14.1026L26.694 14.8872L26.85 14.1026ZM11.667 20.8667C11.2252 20.8667 10.867 21.2248 10.867 21.6667C10.867 22.1085 11.2252 22.4667 11.667 22.4667V20.8667ZM25.0003 22.4667C25.4422 22.4667 25.8003 22.1085 25.8003 21.6667C25.8003 21.2248 25.4422 20.8667 25.0003 20.8667V22.4667ZM11.667 25.8667C11.2252 25.8667 10.867 26.2248 10.867 26.6667C10.867 27.1085 11.2252 27.4667 11.667 27.4667V25.8667ZM20.0003 27.4667C20.4422 27.4667 20.8003 27.1085 20.8003 26.6667C20.8003 26.2248 20.4422 25.8667 20.0003 25.8667V27.4667ZM23.3337 34.2H16.667V35.8H23.3337V34.2ZM7.46699 25V15H5.86699V25H7.46699ZM32.5337 15.0347V25H34.1337V15.0347H32.5337ZM16.667 5.8H23.6732V4.2H16.667V5.8ZM23.6732 5.8C25.2185 5.8 25.7493 5.81639 26.2079 6.02233L26.8633 4.56274C26.0191 4.18361 25.0759 4.2 23.6732 4.2V5.8ZM29.2539 6.70608C28.322 5.65771 27.7076 4.94187 26.8633 4.56274L26.2079 6.02233C26.6665 6.22826 27.0314 6.6141 28.058 7.76907L29.2539 6.70608ZM34.1337 15.0347C34.1337 13.8411 34.1458 13.0399 33.8638 12.2984L32.3683 12.867C32.5216 13.2702 32.5337 13.7221 32.5337 15.0347H34.1337ZM31.0518 11.1371C31.9238 12.1181 32.215 12.4639 32.3683 12.867L33.8638 12.2984C33.5819 11.5569 33.0406 10.9662 32.2476 10.0741L31.0518 11.1371ZM16.667 34.2C14.2874 34.2 12.5831 34.1983 11.2872 34.0241C10.0144 33.8529 9.25596 33.5287 8.69714 32.9698L7.56577 34.1012C8.47142 35.0069 9.62375 35.4148 11.074 35.6098C12.5013 35.8017 14.3326 35.8 16.667 35.8V34.2ZM5.86699 25C5.86699 27.3344 5.86529 29.1657 6.05718 30.593C6.25217 32.0432 6.66012 33.1956 7.56577 34.1012L8.69714 32.9698C8.13833 32.411 7.81405 31.6526 7.64292 30.3798C7.46869 29.0839 7.46699 27.3796 7.46699 25H5.86699ZM23.3337 35.8C25.6681 35.8 27.4993 35.8017 28.9266 35.6098C30.3769 35.4148 31.5292 35.0069 32.4349 34.1012L31.3035 32.9698C30.7447 33.5287 29.9863 33.8529 28.7134 34.0241C27.4175 34.1983 25.7133 34.2 23.3337 34.2V35.8ZM32.5337 25C32.5337 27.3796 32.532 29.0839 32.3577 30.3798C32.1866 31.6526 31.8623 32.411 31.3035 32.9698L32.4349 34.1012C33.3405 33.1956 33.7485 32.0432 33.9435 30.593C34.1354 29.1657 34.1337 27.3344 34.1337 25H32.5337ZM7.46699 15C7.46699 12.6204 7.46869 10.9161 7.64292 9.62024C7.81405 8.34738 8.13833 7.58897 8.69714 7.03015L7.56577 5.89878C6.66012 6.80443 6.25217 7.95676 6.05718 9.40704C5.86529 10.8343 5.86699 12.6656 5.86699 15H7.46699ZM16.667 4.2C14.3326 4.2 12.5013 4.1983 11.074 4.39019C9.62375 4.58518 8.47142 4.99313 7.56577 5.89878L8.69714 7.03015C9.25596 6.47133 10.0144 6.14706 11.2872 5.97592C12.5831 5.8017 14.2874 5.8 16.667 5.8V4.2ZM23.367 5V10H24.967V5H23.367ZM28.3337 14.9667H33.3337V13.3667H28.3337V14.9667ZM23.367 10C23.367 10.7361 23.3631 11.221 23.4464 11.6397L25.0157 11.3276C24.9709 11.1023 24.967 10.8128 24.967 10H23.367ZM28.3337 13.3667C27.5209 13.3667 27.2313 13.3628 27.0061 13.318L26.694 14.8872C27.1127 14.9705 27.5976 14.9667 28.3337 14.9667V13.3667ZM23.4464 11.6397C23.7726 13.2794 25.0543 14.5611 26.694 14.8872L27.0061 13.318C26.0011 13.1181 25.2156 12.3325 25.0157 11.3276L23.4464 11.6397ZM11.667 22.4667H25.0003V20.8667H11.667V22.4667ZM11.667 27.4667H20.0003V25.8667H11.667V27.4667ZM32.2476 10.0741L29.2539 6.70608L28.058 7.76907L31.0518 11.1371L32.2476 10.0741Z" fill="#4F46E5" />
                                        </g>
                                    </svg>
                                    <h2 className="text-center text-gray-400   text-xs leading-4">
                                        PNG, JPG, smaller than 20MB
                                        </h2>
                                </div>
                                <div className="grid gap-2">
                                    <h4 className="text-center text-gray-900 text-sm font-medium leading-snug">
                                        Choose images for register palm print
                                    </h4>
                                    <div className="flex items-center justify-center">
                                        <label>
                                            <input type="file" hidden multiple onChange={handleImageChange} />
                                            <div className="flex w-28 h-9 px-2 flex-col bg-blue-600 rounded-full shadow text-white text-xs font-semibold leading-4 items-center justify-center cursor-pointer focus:outline-none">
                                                Choose File
                                            </div>
                                        </label>
                                    </div>
                                </div>
                                {/* <div className="ml-3">Select frames to filter bad quality frame!</div> */}
                                <div className="flex flex-wrap gap-10 mt-4 ml-3">
                                    {
                                        selectedImages.map((image, index) => (
                                            <img
                                                key={index}
                                                src={
                                                    image.type === FileType.FILE && image.file ? 
                                                    URL.createObjectURL(image.file) : 
                                                    "data:image/png;base64, " + image.base64
                                                }
                                                alt={`Selected ${index}`}
                                                className={`w-40 h-auto rounded-lg ${image.isSelected ? 'border-4 border-red-500' : ''}`}
                                                onClick={() => handleImageClick(image)}
                                            />
                                        ))
                                    }
                                </div>

                                <div>
                                    <button 
                                        onClick={sendImagesToServer}
                                        className="ml-3 text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 font-medium rounded-lg text-sm px-5 py-2.5 text-center me-2"
                                        disabled={isLoading}>
                                        {isLoading ? 'Loading...' : 'Confirm to next step ('+(registerPhase-1)+'/3)'}
                                    </button>
                                </div>

                            </div>
                        </div>
                    </>
                }

            </div>

            <div className="border-dashed border-2 border-gray-300 rounded-lg">
                <div >
                    <div className="text-2xl text-center">ROI images here</div>
                </div>
            </div>

            {/* </div> */}
        </div>
    )
}
