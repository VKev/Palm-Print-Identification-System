package tienthuan.service.def;

import org.springframework.web.multipart.MultipartFile;

public interface IUploadFileCloudService {

    String uploadFile(MultipartFile file);

}
