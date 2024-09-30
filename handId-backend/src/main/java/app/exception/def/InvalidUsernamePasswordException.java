package app.exception.def;

public class InvalidUsernamePasswordException extends RuntimeException {

    public InvalidUsernamePasswordException(String message) {
        super(message);
    }

}
