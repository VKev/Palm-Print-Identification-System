package tienthuan.service.def;

import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

public interface IStaffService {

    ResponseEntity<?> uploadPalmPrintImages(MultipartFile[] files);

    ResponseEntity<?> uploadPalmPrintVideo(MultipartFile videoFile);

    ResponseEntity<?> uploadFilterBackgroundCutImages(MultipartFile[] files);

    ResponseEntity<?> recognizePalmPrint(MultipartFile videoFile);

    ResponseEntity<?> registerInference(String studentCode, MultipartFile[] files);

    ResponseEntity<?> registerPalmPrint(String studentCode, MultipartFile videoFile);

    ResponseEntity<?> validateStudentCode(String studentCode);

    ResponseEntity<?> testAI();

}
