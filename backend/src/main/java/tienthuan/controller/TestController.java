package tienthuan.controller;

import lombok.*;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import tienthuan.model.Student;
import tienthuan.model.User;
import tienthuan.model.fixed.Role;
import tienthuan.repository.StudentRepository;
import tienthuan.repository.UserRepository;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/test")
public class TestController {
    private final StudentRepository studentRepository;
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    record TestRegisterRequest(String username, String fullname, String password, String role) { }
    record TestStudentRequest(String studentCode, String fullname) { }

    @PostMapping("/register")
    public ResponseEntity<?> createAccount(@RequestBody TestRegisterRequest registerRequest) {
        try {
            userRepository.save(
                    User.builder()
                            .username(registerRequest.username())
                            .fullname(registerRequest.fullname())
                            .password(passwordEncoder.encode(registerRequest.password()))
                            .isEnable(Boolean.TRUE)
                            .role(Role.valueOf(registerRequest.role()))
                            .build()
            );
            return new ResponseEntity<>("Account created successfully", HttpStatus.OK);
        }
        catch (Exception e) {
            return new ResponseEntity<>("Error: " + e.getMessage(), HttpStatus.BAD_REQUEST);
        }
    }

    @PostMapping("/create-student")
    public ResponseEntity<?> createStudent(@RequestBody TestStudentRequest studentRequest) {
        try {
            studentRepository.save(
                    Student.builder()
                            .studentCode(studentRequest.studentCode())
                            .studentName(studentRequest.fullname())
                            .isRegistered(Boolean.FALSE)
                            .build()
            );
            return new ResponseEntity<>("Student created successfully", HttpStatus.OK);
        }
        catch (Exception e) {
            return new ResponseEntity<>("Error: " + e.getMessage(), HttpStatus.BAD_REQUEST);
        }
    }
}
