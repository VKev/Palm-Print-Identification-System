package app.service;

import app.configuration.ConstantConfiguration;
import app.model.ImageData;
import app.repository.ImageDataRepository;
import lombok.RequiredArgsConstructor;
import org.bson.BsonBinarySubType;
import org.bson.types.Binary;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.gridfs.GridFsTemplate;
import org.springframework.stereotype.Service;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Service
@RequiredArgsConstructor
public class ImageDataService {

    private final ImageDataRepository imageDataRepository;

    private final ConstantConfiguration constant;

    public ImageData getAllImagesByStudentId(String studentRoleNumber) {
        return imageDataRepository.findById(studentRoleNumber).get();
    }

//String roleNumber
    public String saveImagesToDB(String roleNumber) throws IOException {
        roleNumber = (roleNumber == null || roleNumber.isEmpty()) ? UUID.randomUUID().toString().substring(1,6) : roleNumber;
        File folder = new File(constant.LOCAL_IMAGES_PATH);
        File[] files = folder.listFiles((dir, name) -> name.endsWith(".jpg") || name.endsWith(".png"));
        if (files != null) {
            List<Binary> handImages = new ArrayList<>();
            for (File file : files) {
                try (FileInputStream fileInputStream = new FileInputStream(file)) {
                    byte[] fileData = new byte[(int) file.length()];
                    fileInputStream.read(fileData);
                    Binary binaryImage = new Binary(BsonBinarySubType.BINARY, fileData);
                    handImages.add(binaryImage);
                }
            }
            //String roleNumberTest = UUID.randomUUID().toString().substring(1,6);
            ImageData imageData = new ImageData(roleNumber, handImages);
            imageDataRepository.save(imageData);
            deleteAllImagesAtLocal(files);
            return ("Images for role number " + roleNumber + " saved successfully.");
        }
        else {
            return ("No image files found in the folder.");
        }
    }

    public void deleteAllImagesAtLocal(File[] files) {
        for (File file : files) {
            file.delete();
        }
    }

}
