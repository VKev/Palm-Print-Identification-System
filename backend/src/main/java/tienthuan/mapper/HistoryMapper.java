package tienthuan.mapper;

import org.springframework.stereotype.Component;
import tienthuan.dto.response.AiRecognitionResponse;
import tienthuan.dto.response.HistoryResponse;
import tienthuan.model.History;
import tienthuan.model.User;

import java.util.Date;

@Component
public class HistoryMapper {

    public History toEntity(User user, AiRecognitionResponse aiRecognitionResponse) {
        return History.builder()
                .accept(aiRecognitionResponse.getAccept())
                .averageSimilarityScore(aiRecognitionResponse.getAverageSimilarityScore())
                .averageOccurrenceScore(aiRecognitionResponse.getAverageOccurrenceScore())
                .mostCommonId(aiRecognitionResponse.getMostCommonId())
                .occurrenceCount(aiRecognitionResponse.getOccurrenceCount())
                .score(aiRecognitionResponse.getScore())
                .user(user)
                .historyDate(new Date())
                .build();
    }

    public HistoryResponse toResponse(History history) {
        return HistoryResponse.builder()
                .id(history.getId())
                .accept(history.getAccept())
                .averageOccurrenceScore(history.getAverageOccurrenceScore())
                .averageSimilarityScore(history.getAverageSimilarityScore())
                .mostCommonId(history.getMostCommonId())
                .occurrenceCount(history.getOccurrenceCount())
                .score(history.getScore())
                .historyDate(history.getHistoryDate())
                .build();
    }

}
