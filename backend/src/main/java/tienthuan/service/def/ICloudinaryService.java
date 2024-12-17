package tienthuan.service.def;

import org.springframework.web.multipart.MultipartFile;

public interface ICloudinaryService {

    String uploadFile(MultipartFile file);

    void deleteFile(final String publicId);

}
