package tienthuan.service.def;

import org.springframework.http.ResponseEntity;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.util.Collection;

public interface IStaffService {

    ResponseEntity<?> uploadPalmPrintImages(String studentCode, MultipartFile[] files);

    ResponseEntity<?> uploadPalmPrintVideo(String studentCode, MultipartFile videoFile);

    ResponseEntity<?> uploadFilterBackgroundCutImages(MultipartFile[] files);

    ResponseEntity<?> recognizePalmPrint(MultipartFile videoFile);

    ResponseEntity<?> recognizePalmPrint(MultipartFile[] videoFile);

    ResponseEntity<?> registerInference(String studentCode, MultipartFile[] files);

    ResponseEntity<?> registerPalmPrint(String studentCode, MultipartFile videoFile);

    ResponseEntity<?> validateStudentCode(String studentCode);

    ResponseEntity<?> testAI();

}
