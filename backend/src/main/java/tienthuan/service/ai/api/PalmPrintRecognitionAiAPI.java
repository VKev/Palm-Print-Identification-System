package tienthuan.service.ai.api;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;

import java.util.*;

@Slf4j
@Service
public class PalmPrintRecognitionAiAPI {

    private final HttpHeaders headers = new HttpHeaders();
    private final String BASE_URL = "https://0412-118-69-69-187.ngrok-free.app"; // "http://localhost:5000"

    public PalmPrintRecognitionAiAPI () {
        this.headers.set("Content-Type", "application/json");
    }

    @AllArgsConstructor
    private static class InferenceRequest {
        @JsonProperty("id")
        public String studentCode;

        @JsonProperty("images")
        public Collection<byte[]> frames;
    }

    @AllArgsConstructor
    private static class FramesRequest {
        @JsonProperty("images")
        public Collection<byte[]> frames;
    }

    @AllArgsConstructor
    private static class FramesRequestMultipart {
        @JsonProperty("images")
        public Collection<String> frames;
    }



    public ResponseEntity<?> testRequestAiServer() {
        RestTemplate restTemplate = new RestTemplate();
        long startTime = System.currentTimeMillis();
        ResponseEntity<?> response = restTemplate.getForEntity(BASE_URL + "/hello-world", Object.class);
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        log.info("Execution time: {} seconds", executionTime / 1000.0);
        return response;
    }


    public ResponseEntity<?> registerBackgroundCut(Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/register/backgroundcut";
        ResponseEntity<?> response = this.exchangeFramesToAiServer(new FramesRequest(frames), url, HttpMethod.POST);
        return response;
    }

//    public ResponseEntity<?> registerBackgroundCutV2(Collection<String> base64Images) {
//        String url = BASE_URL + "/ai/register/backgroundcut";
//        RestTemplate restTemplate = new RestTemplate();
//        HttpEntity<Object> requestEntity = new HttpEntity<>(new FramesRequestMultipart(base64Images), headers);
//        return restTemplate.exchange(url, HttpMethod.POST, requestEntity, Object.class);
//    }


    public ResponseEntity<?> registerRoiCut(Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/register/roicut";
        return this.exchangeFramesToAiServer(new FramesRequest(frames), url, HttpMethod.POST);
    }


    public ResponseEntity<Object> registerInference(String studentCode, Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/register/inference";
        return this.exchangeFramesToAiServer(new InferenceRequest(studentCode, frames), url, HttpMethod.POST);
    }

    /*
    {
        "accept": true/false,
        "average_occurrence_score": 1.0,
        "average_similarity_score": 1.0,
        "most_common_id": "SE184160",
        "occurrence_count": 2
    }
     */
    public ResponseEntity<Object> recognizePalmPrintEuclidean(Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/recognize/euclidean";
        return this.exchangeFramesToAiServer(new FramesRequest(frames), url, HttpMethod.POST);
    }

    public ResponseEntity<Object> recognizePalmPrintCosine(Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/recognize/cosine";
        return this.exchangeFramesToAiServer(new FramesRequest(frames), url, HttpMethod.POST);
    }


    private <T> ResponseEntity<Object> exchangeFramesToAiServer(T data, String url, HttpMethod httpMethod) {
        RestTemplate restTemplate = new RestTemplate();
        HttpEntity<Object> requestEntity = new HttpEntity<>(data, headers);
        return restTemplate.exchange(url, httpMethod, requestEntity, Object.class);
    }



}
