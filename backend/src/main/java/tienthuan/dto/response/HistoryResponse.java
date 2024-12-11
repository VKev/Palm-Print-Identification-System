package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.persistence.Column;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.*;

import java.util.Date;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class HistoryResponse {

    @JsonProperty("id")
    private Long id;

    @JsonProperty("accept")
    private Boolean accept;

    @JsonProperty("averageOccurrenceScore")
    private Double averageOccurrenceScore;

    @JsonProperty("averageSimilarityScore")
    private Double averageSimilarityScore;

    @JsonProperty("mostCommonId")
    private String mostCommonId;

    @JsonProperty("occurrenceCount")
    private Integer occurrenceCount;

    @JsonProperty("score")
    private Double score;

    @JsonProperty("historyDate")
    private Date historyDate;

}
