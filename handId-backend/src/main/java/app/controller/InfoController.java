package app.controller;

import app.dto.response.UserDTO;
import app.service.AuthenticationService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/info")
public class InfoController {

    @Autowired
    private AuthenticationService authenticationService;

    @GetMapping("/get")
    public ResponseEntity<UserDTO> getUserInfoFromToken(HttpServletRequest request, HttpServletResponse response) {
        UserDTO userDTO = authenticationService.getUserInfo(request, response);
        return ResponseEntity.ok(userDTO);
    }

}

