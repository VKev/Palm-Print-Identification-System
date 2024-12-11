package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AiRecognitionResponse {

    @JsonProperty("accept")
    private Boolean accept;

    @JsonProperty("average_occurrence_score")
    private Double averageOccurrenceScore;

    @JsonProperty("average_similarity_score")
    private Double averageSimilarityScore;

    @JsonProperty("most_common_id")
    private String mostCommonId;

    @JsonProperty("occurrence_count")
    private Integer occurrenceCount;

    @JsonProperty("score")
    private Double score;

    @JsonProperty("student_info")
    private StudentResponse studentResponse;
}
