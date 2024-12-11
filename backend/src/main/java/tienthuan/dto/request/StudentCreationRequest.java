package tienthuan.dto.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StudentCreationRequest {

    @JsonProperty("studentCode")
    private String studentCode;

    @JsonProperty("studentName")
    private String studentName;

}
