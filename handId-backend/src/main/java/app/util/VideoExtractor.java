package app.util;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

@Slf4j
public class VideoExtractor {

    public void extractImages(String videoFilePath, String saveImgPath, String imgFormat, int frameInterval)
            throws FrameGrabber.Exception, IOException {
        FFmpegFrameGrabber frameGrabber = new FFmpegFrameGrabber(videoFilePath);
        try {
            frameGrabber.start();  Frame frame;  int frameNumber = 0;
            Java2DFrameConverter converter = new Java2DFrameConverter();
            while ((frame = frameGrabber.grab()) != null) {
                if (frame.image != null) {
                    if (frameNumber % frameInterval == 0) {
                        BufferedImage bufferedImage = converter.convert(frame);
                        if (bufferedImage != null) {
                            File output = new File(saveImgPath + "/frame_" + frameNumber + imgFormat);
                            ImageIO.write(bufferedImage, imgFormat.substring(1), output);
                            System.out.println("Saved frame: " + output.getAbsolutePath());
                        }
                    }
                }
                else {
                    System.out.println("Skipping non-image frame: " + frameNumber);
                }
                frameNumber++;
            }

            frameGrabber.stop();
        }
        catch (IOException e) {
            throw new IOException("Error extracting images: " + e.getMessage());
        }
        finally {
            frameGrabber.release();
        }
    }

}
