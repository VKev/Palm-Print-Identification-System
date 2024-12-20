package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import tienthuan.model.PalmPrintImage;

import java.util.Collection;
import java.util.List;

@Setter
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class StudentResponse {

    @JsonProperty("id")
    private Long id;

    @JsonProperty("studentCode")
    private String studentCode;

    @JsonProperty("studentName")
    private String studentName;

    @JsonProperty("isRegistered")
    private Boolean isRegistered;

    @JsonProperty("imagePaths")
    private List<String> imagePaths;

}
