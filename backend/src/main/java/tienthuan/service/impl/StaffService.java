package tienthuan.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.dto.response.*;
import tienthuan.mapper.StudentMapper;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;
import tienthuan.repository.PalmPrintImageRepository;
import tienthuan.repository.StudentRepository;
import tienthuan.service.ai.api.PalmPrintRecognitionAiAPI;
import tienthuan.service.def.IStaffService;
import tienthuan.util.ImageUtil;
import tienthuan.util.VideoUtil;
import java.nio.file.Files;
import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class StaffService implements IStaffService {

    private final StudentRepository studentRepository;

    private final PalmPrintImageRepository palmPrintImageRepository;

    private final PalmPrintRecognitionAiAPI palmPrintRecognitionAiAPI;

    private final UploadFileCloudService uploadFileCloudService;

    private final VideoUtil videoUtil;

    private final StudentMapper studentMapper;


    @Override
    public ResponseEntity<?> uploadPalmPrintImages(String studentCode, MultipartFile[] files) {
        if (files == null)
            return new ResponseEntity<>(new ErrorResponse("No file uploaded"), HttpStatus.BAD_REQUEST);
        var student = studentRepository.findByStudentCode(studentCode);
        if (student.isEmpty()) {
            return new ResponseEntity<>(new ErrorResponse("Student code not found"), HttpStatus.NOT_FOUND);
        }
        else {
//            for (MultipartFile file : files) {
//                savePalmPrintImages(student.get(), file);
//            }
            return palmPrintRecognitionAiAPI.registerBackgroundCut(convertMultipartFilesToBase64(files));
        }
    }

    @Override
    public ResponseEntity<?> uploadFilterBackgroundCutImages(MultipartFile[] files) {
        if (files == null)
            return new ResponseEntity<>(new ErrorResponse("No file uploaded"), HttpStatus.BAD_REQUEST);
        else {
            return palmPrintRecognitionAiAPI.registerRoiCut(convertMultipartFilesToBase64(files));
        }
    }

    @Override
    public ResponseEntity<?> registerInference(String studentCode, MultipartFile[] files) {
        if (files == null)
            return new ResponseEntity<>(new ErrorResponse("No file uploaded"), HttpStatus.BAD_REQUEST);
        else {
            try {
                palmPrintRecognitionAiAPI.registerInference(studentCode, convertMultipartFilesToBase64(files));
                var student = studentRepository.findByStudentCode(studentCode).get();
                student.setIsRegistered(Boolean.TRUE);
                studentRepository.save(student);
                return new ResponseEntity<>(new MessageResponse("Register palm print successfully!"), HttpStatus.OK);
            }
            catch (Exception exception) {
                log.error("Exception at register inference: " + exception.getMessage());
                return new ResponseEntity<>(new ErrorResponse("Register palm print fail!"), HttpStatus.INTERNAL_SERVER_ERROR);
            }
        }
    }

    @Override
    public ResponseEntity<?> uploadPalmPrintVideoRegistration(String studentCode, MultipartFile videoFile) {
        try {
            Collection<File> extractedImages = videoUtil.extractImages(videoFile);
            List<byte[]> base64Images = new ArrayList<>();
            if (extractedImages == null)
                return new ResponseEntity<>(new ErrorResponse("No file extracted"), HttpStatus.BAD_REQUEST);

            var student = studentRepository.findByStudentCode(studentCode);
            if (student.isEmpty()) {
                return new ResponseEntity<>(new ErrorResponse("Student code not found"), HttpStatus.NOT_FOUND);
            }
            else {
                for (File file : extractedImages) {
                    base64Images.add(Files.readAllBytes(file.toPath()));
                    // savePalmPrintImages(student.get(), file);
                }
                List<byte[]> filterBase64Images = base64Images.stream().skip(Math.max(0, base64Images.size() - 30)).toList();
                return new ResponseEntity<>(
                        new VideoUploadingResponse("Upload and extract palm print video successfully!",
                                Boolean.TRUE, filterBase64Images), HttpStatus.OK
                );
            }
        }
        catch (Exception exception) {
            log.error("Exception at upload palm print video: " + exception.getMessage());
            return new ResponseEntity<>(new ErrorResponse("Upload and extract palm print video fail!"), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @Override
    public ResponseEntity<?> recognizePalmPrint(MultipartFile videoFile) {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            Collection<File> extractedImages = videoUtil.extractImages(videoFile);
            List<byte[]> base64Images = new ArrayList<>();
            if (extractedImages == null)
                return new ResponseEntity<>(new ErrorResponse("No file extracted"), HttpStatus.BAD_REQUEST);

            for (File file : extractedImages) {
                base64Images.add(Files.readAllBytes(file.toPath()));
            }
            List<byte[]> filterBase64Images = base64Images.stream().skip(Math.max(0, base64Images.size() - 30)).toList();
            var aiServeResponse = palmPrintRecognitionAiAPI.recognizePalmPrintCosine(filterBase64Images).getBody();
            AiRecognitionResponse aiRecognitionResponse = objectMapper.convertValue(aiServeResponse, AiRecognitionResponse.class);
            aiRecognitionResponse.setStudentResponse(
                    studentMapper.toResponse(studentRepository.findByStudentCode(aiRecognitionResponse.getMostCommonId()).get())
            );
            return new ResponseEntity<>(aiRecognitionResponse, HttpStatus.OK);
        }
        catch (Exception exception) {
            log.info("Exception at recognize palm print: " + exception.getMessage());
            return new ResponseEntity<>(new ErrorResponse("Recognize palm print fail!"), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    private void savePalmPrintImages(Student student, MultipartFile file) {
        try {
            byte[] compressedImage = ImageUtil.compressImage(file.getBytes());
            String fileUrlCloud = uploadFileCloudService.uploadFile(file);
            PalmPrintImage palmPrintImage = PalmPrintImage.builder()
                    .student(student)
                    .imagePath(fileUrlCloud)
                    .image(compressedImage)
                    .build();
            palmPrintImageRepository.save(palmPrintImage);
        }
        catch (Exception exception) {
            log.info("Exception at save palm print images: " + exception.getMessage());
        }
    }

    private void savePalmPrintImages(Student student, File file) {
        try {
            byte[] compressedImage = ImageUtil.compressImage(Files.readAllBytes(file.toPath()));
            String fileUrlCloud = uploadFileCloudService.uploadFile(file);
            log.info(fileUrlCloud);
            PalmPrintImage palmPrintImage = PalmPrintImage.builder()
                    .student(student)
                    .image(compressedImage)
                    .imagePath(fileUrlCloud)
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

    @Override
    public ResponseEntity<?> recognizePalmPrint(MultipartFile[] files) {
        if (files == null)
            return new ResponseEntity<>(new ErrorResponse("No file uploaded"), HttpStatus.BAD_REQUEST);

        else {
            return palmPrintRecognitionAiAPI.recognizePalmPrintCosine(convertMultipartFilesToBase64(files));
        }
    }


    @Override
    public ResponseEntity<?> validateStudentCode(String studentCode) {
        var student = studentRepository.findByStudentCode(studentCode);
        return student.map(
                value -> new ResponseEntity<>(
                         new StudentValidationResponse(Boolean.TRUE, value.getIsRegistered()), HttpStatus.OK
                ))
                .orElseGet(
                        () -> new ResponseEntity<>(
                              new StudentValidationResponse(Boolean.FALSE, Boolean.FALSE), HttpStatus.NOT_FOUND
                        )
                );
    }

    @Override
    public ResponseEntity<?> testAI() {
        return palmPrintRecognitionAiAPI.testRequestAiServer();
    }

}
