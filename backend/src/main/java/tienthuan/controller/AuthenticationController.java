package tienthuan.controller;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import tienthuan.dto.request.AuthenticationRequest;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.service.def.IAuthenticationService;
import java.io.IOException;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/auth")
public class AuthenticationController {

    private final IAuthenticationService authenticationService;

//    @PostMapping("/register")
//    @PreAuthorize("hasRole('ADMIN')")
//    public ResponseEntity<?> register(@RequestBody RegisterRequest registerRequest) {
//        return authenticationService.register(registerRequest);
//    }

    @PostMapping("/authenticate")
    public ResponseEntity<?> authenticate(@RequestBody AuthenticationRequest authRequest) {
        return authenticationService.authenticate(authRequest);
    }

    @PostMapping("/refresh-token")
    public ResponseEntity<?> refreshToken(HttpServletRequest request, HttpServletResponse response) throws IOException {
        return authenticationService.refreshToken(request, response);
    }

    @GetMapping("/user/info")
    public ResponseEntity<?> getUserInfo(HttpServletRequest request, HttpServletResponse response) {
        return authenticationService.getUser(request, response);
    }
}
