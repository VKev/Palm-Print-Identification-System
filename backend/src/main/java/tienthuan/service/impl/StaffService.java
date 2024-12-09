package tienthuan.service.impl;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.service.ai.api.PalmPrintRecognitionAiAPI;
import tienthuan.service.def.IStaffService;

@Slf4j
@Service
@RequiredArgsConstructor
public class StaffService implements IStaffService {

    private final PalmPrintRecognitionAiAPI palmPrintRecognitionAiAPI;

    @Override
    public ResponseEntity<?> uploadPalmPrintImages(MultipartFile[] files) {
        // Convert to files

        // Compress base64

        // Save to database

        // Save to cloud
        return null;
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
