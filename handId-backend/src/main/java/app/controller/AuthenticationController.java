package app.controller;

import app.dto.request.AuthenticationRequest;
import app.dto.response.AuthenticationResponse;
import app.exception.def.InvalidTokenException;
import app.exception.def.InvalidUsernamePasswordException;
import app.service.AuthenticationService;
import app.service.UserService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.io.IOException;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthenticationController {

    private final AuthenticationService authenticationService;

    private final UserService userService;

    @PostMapping("/authenticate")
    public ResponseEntity<AuthenticationResponse> authenticate(@RequestBody AuthenticationRequest request)
            throws InvalidUsernamePasswordException {
        return ResponseEntity.ok(authenticationService.authenticate(request));
    }

    @PostMapping("/refresh-token")
    public ResponseEntity<AuthenticationResponse> refreshToken(HttpServletRequest request, HttpServletResponse response)
            throws InvalidUsernamePasswordException, InvalidTokenException, IOException {
        return ResponseEntity.ok(
                authenticationService.refreshToken(request, response)
        );
    }

}
