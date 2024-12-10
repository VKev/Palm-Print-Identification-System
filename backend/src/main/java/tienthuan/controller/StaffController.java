package tienthuan.controller;

import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
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
    public ResponseEntity<?> uploadPalmPrintImages(
            @PathVariable("studentCode") String studentCode,
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
        return staffService.uploadPalmPrintImages(studentCode, files);
    }

    /**
     * API upload filter background cut images - second phase
     * @param files filter background cut images
     * @return Collection of roi cut images
     */
    @PostMapping("/upload-filter-background-cut-images")
    public ResponseEntity<?> uploadFilterBackgroundCutImages(
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
        return staffService.uploadFilterBackgroundCutImages(files);
    }

    /**
     * API register palm print - final phase
     * @param studentCode Student code
     * @param files Image files
     * @return result of registration
     */
    @PostMapping("/register-palm-print/{studentCode}")
    public ResponseEntity<?> registerInference(
            @PathVariable("studentCode") String studentCode,
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
        return staffService.registerInference(studentCode, files);
    }

    // Register by upload video - 3 phases

    /**
     * API upload palm print video, extract frames - first phase
     * @param videoFile Video file
     * @return Collections of cut background images
     */
        @PostMapping(value = "/upload-palm-print-video/{studentCode}", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> uploadPalmPrintVideo(
            @PathVariable("studentCode") String studentCode,
            @Parameter(description = "Video file", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("video")
            MultipartFile videoFile
    ) {
        return staffService.uploadPalmPrintVideo(studentCode, videoFile);
    }


    @PostMapping("/recognize-palm-print")
    public ResponseEntity<?> recognizePalmPrint(
            @Parameter(description = "Video file", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("video") MultipartFile videoFile
    ) {
        return staffService.recognizePalmPrint(videoFile);
    }

    @PostMapping("/test-recognition-palm-print")
    public ResponseEntity<?> testRecognitionPalmPrint(
            @Parameter(description = "Image files", content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE))
            @RequestParam("images") MultipartFile[] files
    ) {
            return staffService.recognizePalmPrint(files);
    }

    @GetMapping("/test-ai")
    public ResponseEntity<?> testAI() {
        ResponseEntity<?> response = staffService.testAI();
        log.info("Response: {}", response);
        return response;
    }



}
