package tienthuan.dto.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class CapturedFrameRequest {

    @JsonProperty("uuid")
    private String uuid;

    @JsonProperty("base64Image")
    private String base64Image;
}
