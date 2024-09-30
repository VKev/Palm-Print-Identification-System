package app.controller;

import app.service.ImageDataService;
import app.service.VideoService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import java.io.IOException;

@RestController
@RequestMapping("/api/video")
@RequiredArgsConstructor
public class VideoController {

    private final VideoService videoService;

    private final ImageDataService imageDataService;

    @PostMapping("/process")
    public String extractVideo(
            @RequestParam("filename") String filename,
            @RequestParam("roleNumber") String roleNumber
    ) throws IOException {
        videoService.extractVideo(filename.trim());
        System.out.println("Role number: "+roleNumber);
        return imageDataService.saveImagesToDB(roleNumber);
    }

}
