package app.exception.handler;

import app.exception.def.ErrorResponse;
import app.exception.def.InvalidUsernamePasswordException;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class InvalidTokenExceptionHandler {

    @ExceptionHandler(InvalidUsernamePasswordException.class)
    @ResponseStatus(HttpStatus.FORBIDDEN)
    @ResponseBody
    public ErrorResponse handle(InvalidUsernamePasswordException exception) {
        return new ErrorResponse(HttpStatus.FORBIDDEN, exception.getMessage());
    }

}
