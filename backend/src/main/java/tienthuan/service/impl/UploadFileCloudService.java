package tienthuan.service.impl;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.service.def.IUploadFileCloudService;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class UploadFileCloudService implements IUploadFileCloudService {

    private final Cloudinary cloudinary;

    @Override
    public String uploadFile(MultipartFile file) {
        assert file.getOriginalFilename() != null;
        String publicValue = this.generatePublicValue(file.getOriginalFilename());
        String fileExtension = this.getFileName(file.getOriginalFilename())[1];
        File fileUpload;
        try {
            fileUpload = this.convertFile(file);
            cloudinary.uploader().upload(fileUpload, ObjectUtils.asMap("public_id", publicValue));
            String fileUrl = cloudinary.url().generate(StringUtils.join(publicValue, ".", fileExtension));
            cleanDisk(fileUpload);
            return fileUrl;
        }
        catch (IOException ioException) {
            log.error("Exception at upload file to cloud: " + ioException.getMessage());
        }
        return "";
    }

    public String uploadFile(File file) {
        try {
            String publicValue = this.generatePublicValue(file.getName());
            String fileExtension = this.getFileName(file.getName())[1];
            cloudinary.uploader().upload(file, ObjectUtils.asMap("public_id", publicValue));
            String fileUrl = cloudinary.url().generate(StringUtils.join(publicValue, ".", fileExtension));
            cleanDisk(file);
            return fileUrl;
        }
        catch (IOException ioException) {
            log.error("Exception at upload file to cloud: " + ioException.getMessage());
        }
        return "";
    }

    private File convertFile(MultipartFile file)  throws IOException {
        assert file.getOriginalFilename() != null;
        File convertedFile = new File(
                StringUtils.join(
                        this.generatePublicValue(file.getOriginalFilename()),
                        this.getFileName(file.getOriginalFilename())[1]
                )
        );
        try(InputStream inputStream = file.getInputStream()) {
            Files.copy(inputStream, convertedFile.toPath());
        }
        return convertedFile;
    }

    private void cleanDisk(File file) {
        try {
            Path filePath = file.toPath();
            Files.delete(filePath);
        }
        catch (IOException ioException) {
            log.info("Exception at clean disk: " + ioException.getMessage());
        }
    }

    private String generatePublicValue(String originalFilename) {
        String filename = this.getFileName(originalFilename)[0];
        return StringUtils.join(UUID.randomUUID().toString(), "_", filename);
    }

    private String[] getFileName(String originalFilename) {
        return originalFilename.split("\\.");
    }

}
