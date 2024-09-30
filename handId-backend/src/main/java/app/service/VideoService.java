package app.service;

import app.configuration.ConstantConfiguration;
import app.dto.response.Response;
import app.model.ImageData;
import app.repository.ImageDataRepository;
import app.util.VideoExtractor;
import lombok.RequiredArgsConstructor;
import org.bson.BsonBinarySubType;
import org.bson.types.Binary;
import org.bytedeco.javacv.FrameGrabber;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static java.rmi.server.LogStream.log;

@Service
@RequiredArgsConstructor
public class VideoService {

    private final ConstantConfiguration constant;


    public String extractVideo(String filepath) {
        String videoFullPath = constant.LOCAL_VIDEO_PATH + "\\" + filepath;
        File videoFile = new File(videoFullPath);
        System.out.println(videoFullPath);
        try {
            if (videoFile.exists()) {
                VideoExtractor extractor = new VideoExtractor();
                extractor.extractImages(
                        videoFullPath,
                        constant.LOCAL_IMAGES_PATH,
                        constant.IMAGES_EXTENSION_TYPE,
                        constant.EXTRACTOR_IMAGE_FRAME_JUMP
                );
                //System.out.println("Processing video at: " + videoFullPath);
                videoFile.delete();
                return "Video processed successfully!";
            }
            else {
                return "Video file not found.";
            }
        }
        catch (FrameGrabber.Exception | IOException e) {
            return "exception: "+ e.getMessage();
        }
    }



 }
