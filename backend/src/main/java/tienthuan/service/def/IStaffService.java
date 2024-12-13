package tienthuan.service.def;

import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

public interface IStaffService {

    ResponseEntity<?> uploadPalmPrintImages(String studentCode, MultipartFile[] files);

    ResponseEntity<?> uploadPalmPrintVideoRegistration(String studentCode, MultipartFile videoFile);

    ResponseEntity<?> uploadFilterBackgroundCutImages(MultipartFile[] files);

    ResponseEntity<?> recognizePalmPrint(Long userId, MultipartFile videoFile);

    ResponseEntity<?> recognizePalmPrint(MultipartFile[] videoFile);

    ResponseEntity<?> registerInference(String studentCode, MultipartFile[] files);

    ResponseEntity<?> validateStudentCode(String studentCode);

    ResponseEntity<?> getHistoriesByUser(Long userId);

    ResponseEntity<?> testAI();

}
