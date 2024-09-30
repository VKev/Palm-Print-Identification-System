package app.dto.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UpdateStaffRequest {

    @JsonProperty("fullname")
    private String fullname;

    @JsonProperty("phone")
    private String phone;

}
