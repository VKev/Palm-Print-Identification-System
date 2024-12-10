package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class AccountCreationResponse {

    @JsonProperty("createdUser")
    private UserResponse createdUser;

    @JsonProperty("message")
    private String message;

}
