package tienthuan.service.impl;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.service.def.ICloudinaryService;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class CloudinaryService implements ICloudinaryService {

    private final Cloudinary cloudinary;
    private final String BASE_URL = "http://res.cloudinary.com/";

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

    @Override
    public void deleteFile(String fileUrl) {
        try {
            this.cloudinary.uploader().destroy(this.extractPublicId(fileUrl), ObjectUtils.emptyMap());
        }
        catch (Exception exception) {
            log.error("Exception at delete file from cloud: " + exception.getMessage());
        }
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

    private String extractPublicId(String fileUrl) {
        if (fileUrl == null || fileUrl.isEmpty())
            throw new IllegalArgumentException("File URL cannot be null or empty");

        String[] resourceParts = this.getStrings(fileUrl);
        String publicIdWithFormat = resourceParts[resourceParts.length - 1];
        int dotIndex = publicIdWithFormat.lastIndexOf(".");
        if (dotIndex != -1) {
            return publicIdWithFormat.substring(0, dotIndex);
        }
        return publicIdWithFormat;
    }

    private String[] getStrings(String fileUrl) {
        String[] urlParts = fileUrl.split(BASE_URL);
        if (urlParts.length < 2) {
            throw new IllegalArgumentException("Invalid Cloudinary URL");
        }
        String resourcePath = urlParts[1];
        int versionIndex = resourcePath.indexOf("/v");
        if (versionIndex != -1) {
            resourcePath = resourcePath.substring(versionIndex + 1);
        }
        return resourcePath.split("/");
    }

}
