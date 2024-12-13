package tienthuan.multithread;

import lombok.extern.slf4j.Slf4j;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;
import tienthuan.repository.PalmPrintImageRepository;
import tienthuan.service.impl.UploadFileCloudService;
import tienthuan.util.ImageUtil;
import java.io.File;
import java.nio.file.Files;
import java.util.List;


@Slf4j
public class CloudUploader extends Thread {

    private final UploadFileCloudService uploadFileCloudService;
    private final PalmPrintImageRepository palmPrintImageRepository;
    private final MultipartFile[] multipartFiles;
    private final List<File> files;
    private final Student student;

    public CloudUploader(
            UploadFileCloudService uploadFileCloudService,
            PalmPrintImageRepository palmPrintImageRepository,
            MultipartFile[] multipartFiles,
            List<File> files,
            Student student
    ) {
        this.uploadFileCloudService = uploadFileCloudService;
        this.palmPrintImageRepository = palmPrintImageRepository;
        this.multipartFiles = multipartFiles;
        this.files = files;
        this.student = student;
    }

    @Override
    public void run() {
        if (multipartFiles != null) {
            for (MultipartFile file : multipartFiles) {
                savePalmPrintImages(student, file);
            }
        }
        else if (files != null) {
            for (File file : files) {
                savePalmPrintImages(student, file);
            }
        }

    }

    private void savePalmPrintImages(Student student, MultipartFile file) {
        try {
            byte[] compressedImage = ImageUtil.compressImage(file.getBytes());
            String fileUrlCloud = uploadFileCloudService.uploadFile(file);
            PalmPrintImage palmPrintImage = PalmPrintImage.builder()
                    .student(student)
                    .imagePath(fileUrlCloud)
                    .image(compressedImage)
                    .build();
            palmPrintImageRepository.save(palmPrintImage);
        }
        catch (Exception exception) {
            log.info("Exception at save palm print images: " + exception.getMessage());
        }
    }

    private void savePalmPrintImages(Student student, File file) {
        try {
            byte[] compressedImage = ImageUtil.compressImage(Files.readAllBytes(file.toPath()));
            String fileUrlCloud = uploadFileCloudService.uploadFile(file);
            log.info(fileUrlCloud);
            PalmPrintImage palmPrintImage = PalmPrintImage.builder()
                    .student(student)
                    .image(compressedImage)
                    .imagePath(fileUrlCloud)
                    .build();
            palmPrintImageRepository.save(palmPrintImage);
        }
        catch (Exception exception) {
            log.info("Exception at save palm print images: " + exception.getMessage());
        }
    }

}
