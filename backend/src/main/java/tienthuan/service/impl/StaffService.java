package tienthuan.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.dto.response.ErrorResponse;
import tienthuan.dto.response.MessageResponse;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;
import tienthuan.repository.PalmPrintImageRepository;
import tienthuan.repository.StudentRepository;
import tienthuan.service.ai.api.PalmPrintRecognitionAiAPI;
import tienthuan.service.def.IStaffService;
import tienthuan.util.ImageUtil;

import java.util.ArrayList;
import java.util.Base64;
import java.util.Collection;

@Slf4j
@Service
@RequiredArgsConstructor
public class StaffService implements IStaffService {

    private final StudentRepository studentRepository;

    private final PalmPrintImageRepository palmPrintImageRepository;

    private final PalmPrintRecognitionAiAPI palmPrintRecognitionAiAPI;

    private final UploadFileCloudService uploadFileCloudService;


    @Override
    public ResponseEntity<?> uploadPalmPrintImages(String studentCode, MultipartFile[] files) {
        // Convert to files
        if (files == null)
            return new ResponseEntity<>(new ErrorResponse("No file uploaded"), HttpStatus.BAD_REQUEST);
        var student = studentRepository.findByStudentCode(studentCode);
        if (student.isEmpty()) {
            return new ResponseEntity<>(new ErrorResponse("Student code not found"), HttpStatus.NOT_FOUND);
        }
        else {
            // Compress and save base64 images
//            for (MultipartFile file : files) {
//                savePalmPrintImages(student.get(), file);
//            }
            return palmPrintRecognitionAiAPI.registerBackgroundCut(convertMultipartFilesToBase64(files));
        }
        //return new ResponseEntity<>(new MessageResponse("Upload images successfully"), HttpStatus.OK);
    }

    private void savePalmPrintImages(Student student, MultipartFile file) {
        boolean flag = true;
        try {
            String fileUrlCloud = uploadFileCloudService.uploadFile(file);
            PalmPrintImage palmPrintImage = PalmPrintImage.builder()
                    .student(student)
                    .imagePath(fileUrlCloud)
                    .image(ImageUtil.compressImage(file.getBytes()))
                    .build();
            palmPrintImageRepository.save(palmPrintImage);
        }
        catch (Exception exception) {
            log.info("Exception at save palm print images: " + exception.getMessage());
        }
    }

    public Collection<byte[]> convertMultipartFilesToBase64(MultipartFile[] files) {
        Collection<byte[]> base64Images = new ArrayList<>();
        for (MultipartFile file : files) {
            try {
                base64Images.add(file.getBytes());
            }
            catch (Exception exception) {
                log.error("Exception at convert to base64: " + exception.getMessage());
            }
        }
        return base64Images;
    }

    private Collection<byte[]> convertToBase64(MultipartFile[] files) {
        Collection<byte[]> base64Images = new ArrayList<>();
        for (MultipartFile file : files) {
            try {
                base64Images.add(ImageUtil.decompressImage(file.getBytes()));
            }
            catch (Exception exception) {
                log.info("Exception at convert to base64: " + exception.getMessage());
            }
        }
        return base64Images;
    }

    @Override
    public ResponseEntity<?> uploadPalmPrintVideo(MultipartFile videoFile) {
        return null;
    }

    @Override
    public ResponseEntity<?> uploadFilterBackgroundCutImages(MultipartFile[] files) {
        return null;
    }

    @Override
    public ResponseEntity<?> registerInference(String studentCode, MultipartFile[] files) {
        return null;
    }

    @Override
    public ResponseEntity<?> recognizePalmPrint(MultipartFile videoFile) {
        return null;
    }



    @Override
    public ResponseEntity<?> registerPalmPrint(String studentCode, MultipartFile videoFile) {
        return null;
    }

    @Override
    public ResponseEntity<?> validateStudentCode(String studentCode) {
        return null;
    }

    @Override
    public ResponseEntity<?> testAI() {
        return palmPrintRecognitionAiAPI.testRequestAiServer();
    }

}
