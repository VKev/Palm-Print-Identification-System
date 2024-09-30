package app.controller;

import app.model.ImageData;
import app.service.ImageDataService;
import lombok.RequiredArgsConstructor;
import org.bson.types.Binary;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/student/images")
public class ImageDataController {

    public final ImageDataService imageDataService;

    /**
     * Tet get images from DB
     * @param roleNumber
     * @return
     */
    @GetMapping("/get/{code}")
    public ResponseEntity<?> getImagesByRoleNumber(@PathVariable("code") String roleNumber) {
        ImageData imageData = imageDataService.getAllImagesByStudentId(roleNumber);

        List<Binary> images = (List<Binary>) imageData.getHandImages();
        Binary firstImage = images.get(0);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.IMAGE_JPEG);  // or MediaType.IMAGE_PNG based on content type
        headers.setContentLength(firstImage.length());

        return new ResponseEntity<>(firstImage.getData(), headers, HttpStatus.OK);
    }
}
