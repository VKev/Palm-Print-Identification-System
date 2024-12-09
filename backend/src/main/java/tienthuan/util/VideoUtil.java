package tienthuan.util;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;
import tienthuan.configuration.ConstantConfiguration;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;

@Slf4j
@Component
@RequiredArgsConstructor
public class VideoUtil {

    private final ConstantConfiguration constant;

    public Collection<File> extractImages(MultipartFile multipartFile) throws IOException {
        Collection<File> imageFiles = new ArrayList<>();
        File videoFile = this.convertToFile(multipartFile);
        FFmpegFrameGrabber frameGrabber = new FFmpegFrameGrabber(videoFile);
        try {
            frameGrabber.start();  Frame frame;  int frameNumber = 0;
            Java2DFrameConverter converter = new Java2DFrameConverter();
            while ((frame = frameGrabber.grab()) != null) {
                if (frame.image != null) {
                    if (frameNumber % constant.IMAGES_FRAME_JUMP == 0) {
                        BufferedImage bufferedImage = converter.convert(frame);
                        if (bufferedImage != null) {
                            File output = new File(System.getProperty("java.io.tmpdir") + "/frame_" + frameNumber + constant.IMAGES_EXTENSION_TYPE);
                            imageFiles.add(output);
                            ImageIO.write(bufferedImage, constant.IMAGES_EXTENSION_TYPE.substring(1), output);
                            //log.info("Saved frame: " + output.getAbsolutePath());
                        }
                    }
                }
                frameNumber++;
            }
            frameGrabber.stop();
            frameGrabber.release();
        }
        catch (FrameGrabber.Exception frameGrabberException) {
            log.error("Error extracting images: " + frameGrabberException.getMessage());
        }
        return imageFiles;
    }

    private File convertToFile(MultipartFile multipartFile) throws IOException {
        File convertedFile = new File(System.getProperty("java.io.tmpdir") + "/" + multipartFile.getOriginalFilename());
        try {
            InputStream inputStream = multipartFile.getInputStream();
            Files.copy(inputStream, convertedFile.toPath());
        }
        catch (IOException ignored) {
        }
        return convertedFile;
    }

}
