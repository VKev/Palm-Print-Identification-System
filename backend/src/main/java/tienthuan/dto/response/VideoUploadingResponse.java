package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import java.util.Collection;

@Getter
@Setter
@AllArgsConstructor
public class VideoUploadingResponse {

    @JsonProperty("message")
    private String message;

    @JsonProperty("status")
    private Boolean status;

    @JsonProperty("base64Images")
    private Collection<byte[]> frames;

}
