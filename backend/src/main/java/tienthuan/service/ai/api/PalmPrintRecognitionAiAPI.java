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
    private final String BASE_URL = "http://localhost:5000"; // "http://localhost:5000"

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
    private static class FeatureVectorRequest {
        @JsonProperty("images")
        public Collection<String> frames;
    }


    public ResponseEntity<?> testRequestAiServer() {
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<?> response = restTemplate.getForEntity(BASE_URL + "/hello-world", Object.class);
        return response;
    }


    public ResponseEntity<?> registerBackgroundCut(Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/register/backgroundcut";
        ResponseEntity<?> response = this.exchangeFramesToAiServer(new FramesRequest(frames), url, HttpMethod.POST);
        return response;
    }


    public ResponseEntity<?> registerRoiCut(Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/register/roicut";
        return this.exchangeFramesToAiServer(new FramesRequest(frames), url, HttpMethod.POST);
    }


    public ResponseEntity<Object> registerInference(String studentCode, Collection<byte[]> frames) {
        String url = BASE_URL + "/ai/register/inference";
        return this.exchangeFramesToAiServer(new InferenceRequest(studentCode, frames), url, HttpMethod.POST);
    }
    
    /* ---------------------------------- */

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

    /*
    {
        "feature_vector": {
            [
                -0.1908714473247528,
                -0.5968778729438782,
                ....
            ],
            [
                -0.1908714473247528,
                -0.5968778729438782,
                ....
            ],...
        }
    }
    */
    public ResponseEntity<Object> getFeatureVector(Collection<String> frames) {
        RestTemplate restTemplate = new RestTemplate();
        HttpEntity<Object> requestEntity = new HttpEntity<>(new FeatureVectorRequest(frames), headers);
        String url = BASE_URL + "/ai/recognize/cosine-only";
        return restTemplate.exchange(url, HttpMethod.POST, requestEntity, Object.class);
    }


    private <T> ResponseEntity<Object> exchangeFramesToAiServer(T data, String url, HttpMethod httpMethod) {
        RestTemplate restTemplate = new RestTemplate();
        HttpEntity<Object> requestEntity = new HttpEntity<>(data, headers);
        return restTemplate.exchange(url, httpMethod, requestEntity, Object.class);
    }

}
