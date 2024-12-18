package tienthuan.controller;

import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.scheduling.annotation.Async;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.dto.request.CapturedFrameRequest;
import tienthuan.dto.request.FrameRecognitionRequest;
import tienthuan.service.def.IStaffService;


@Slf4j
@RestController
@RequiredArgsConstructor
//@PreAuthorize("hasRole('STAFF')")
@RequestMapping("/api/staff")
public class StaffController {

    private final IStaffService staffService;

    @GetMapping("/validate-student-code/{studentCode}")
    public ResponseEntity<?> validateStudentCode(@PathVariable("studentCode") String studentCode) {
        return staffService.validateStudentCode(studentCode);
    }

    // Register by upload images - 3 phases

    /**
     * API upload palm print images - first phase
     * @param files Image files
     * @return Collections of cut background images
     */
    @PostMapping("/upload-palm-print-images/{studentCode}")
    public Object uploadPalmPrintImages(
            @PathVariable("studentCode") String studentCode,
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
        ResponseEntity<?> response = staffService.uploadPalmPrintImages(studentCode, files);
        return response.getBody();
    }

    /**
     * API upload filter background cut images - second phase
     * @param files filter background cut images
     * @return Collection of roi cut images
     */
    @PostMapping("/upload-filter-background-cut-images")
    public Object uploadFilterBackgroundCutImages(
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
        return staffService.uploadFilterBackgroundCutImages(files).getBody();
    }

    /**
     * API register palm print - final phase
     * @param studentCode Student code
     * @param files Image files
     * @return result of registration
     */
    @PostMapping("/register-palm-print/{studentCode}")
    public Object registerInference(
            @PathVariable("studentCode") String studentCode,
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
        return staffService.registerInference(studentCode, files).getBody();
    }

    // Register by upload video - 3 phases

    /**
     * API upload palm print video, extract frames - first phase
     * @param videoFile Video file
     * @return Collections of cut background images
     */
    @PostMapping(value = "/upload-palm-print-video/registration/{studentCode}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> uploadPalmPrintVideoRegistration(
            @PathVariable("studentCode") String studentCode,
            @Parameter(description = "Video file", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("video")
            MultipartFile videoFile
    ) {
        return staffService.uploadPalmPrintVideoRegistration(studentCode, videoFile);
    }


    @PostMapping("/recognize-palm-print/{userId}")
    public ResponseEntity<?> recognizePalmPrint(
            @PathVariable("userId") Long userId,
            @Parameter(description = "Video file", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("video")
            MultipartFile videoFile
    ) {
        return staffService.recognizePalmPrint(userId, videoFile);
    }

    @Async
    @PostMapping("/recognize-palm-print-by-frames")
    public ResponseEntity<?> recognizePalmPrint(
            @RequestBody CapturedFrameRequest capturedFrameRequest
    ) {
        return staffService.recognizePalmPrint(capturedFrameRequest.getUuid(), capturedFrameRequest.getBase64Image());
    }


    @PostMapping("/test-recognition-palm-print")
    public Object testRecognitionPalmPrint(
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
            return staffService.recognizePalmPrint(files).getBody();
    }

    @GetMapping(value = "/test-ai", produces = MediaType.APPLICATION_JSON_VALUE)
    public String testAI() {
        ResponseEntity<?> response = staffService.testAI();
        log.info("Response: {}", response);
        return (String) response.getBody();
    }

    @GetMapping("/history-logs/{userId}")
    public ResponseEntity<?> getHistoriesByUser(@PathVariable("userId") Long userId) {
        return staffService.getHistoriesByUser(userId);
    }
}
