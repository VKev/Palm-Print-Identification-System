package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FeatureVectorResponse {

    @JsonProperty("processStatus")
    public String processStatus;

}
