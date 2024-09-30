package app.configuration;

import org.springframework.stereotype.Component;

@Component
public class MessageConfiguration {

    public final String ERROR_USERNAME_PASSWORD_INVALID = "Username or password is incorrect!";
    public final String ERROR_LOGIN_SESSION_EXPIRED = "Login session expired! Please login again!";
    public final String ERROR_STUDENT_ROLENUMBER_NOT_EXIST = "Student role number does not exist in university system";
    public final String SUCCESS_STUDENT_ROLENUMBER_VALID = "Student role number is valid";
    public final String SUCCESS_UPDATE_STAFF_SUCCESS = "Update staff successfully!";

}
