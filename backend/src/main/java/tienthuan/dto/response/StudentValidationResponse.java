package tienthuan.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Getter
@Setter
public class StudentValidationResponse {

    @JsonProperty("statusResult")
    private Boolean statusResult;

    @JsonProperty("isRegistered")
    private Boolean isRegistered;

    @JsonProperty("message")
    private String message;

    public StudentValidationResponse(Boolean statusResult, Boolean isRegistered) {
        this.statusResult = statusResult;
        this.isRegistered = isRegistered;
        if (statusResult && !isRegistered) {
            this.message = "Student is valid!";
        }
        else {
            this.message = "Student is invalid";
        }
        if (statusResult && isRegistered) {
            this.message = "Student is valid and registered!";
        }
    }

}
