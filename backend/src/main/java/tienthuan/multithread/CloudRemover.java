package tienthuan.multithread;

import lombok.extern.slf4j.Slf4j;
import tienthuan.service.def.ICloudinaryService;
import java.util.List;

@Slf4j
public class CloudRemover extends Thread {

    private final ICloudinaryService cloudinaryService;

    private final List<String> fileUrls;

    public CloudRemover(ICloudinaryService cloudinaryService, List<String> fileUrls) {
        this.cloudinaryService = cloudinaryService;
        this.fileUrls = fileUrls;
    }

    @Override
    public void run() {
        for (String fileUrl : fileUrls) {
            cloudinaryService.deleteFile(fileUrl);
        }
    }

}
