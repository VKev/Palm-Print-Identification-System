package tienthuan.service.def;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.ResponseEntity;
import tienthuan.dto.request.AuthenticationRequest;
import tienthuan.dto.request.RegisterRequest;

import java.io.IOException;

public interface IAuthenticationService {

    ResponseEntity<?> register(RegisterRequest registerRequest);

    ResponseEntity<?> authenticate(AuthenticationRequest authRequest);

    ResponseEntity<?> refreshToken(HttpServletRequest request, HttpServletResponse response) throws IOException;

    ResponseEntity<?> getUser(HttpServletRequest request, HttpServletResponse response);

}
