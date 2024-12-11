package tienthuan.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.dto.response.*;
import tienthuan.mapper.StudentMapper;
import tienthuan.mapper.UserMapper;
import tienthuan.model.User;
import tienthuan.model.fixed.Role;
import tienthuan.repository.StudentRepository;
import tienthuan.repository.UserRepository;
import tienthuan.service.def.IAdminService;

import java.util.Collection;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AdminService implements IAdminService {

    private final UserMapper userMapper;

    private final UserRepository userRepository;

    private final StudentMapper studentMapper;

    private final StudentRepository studentRepository;

    @Override
    public ResponseEntity<Collection<UserResponse>> getAllStaffAccounts() {
        List<UserResponse> staffAccounts = userRepository.findAllByRole(Role.STAFF).stream().map(
                userMapper::toResponse
        ).toList();
        return new ResponseEntity<>(staffAccounts, HttpStatus.OK);
    }

    @Override
    public ResponseEntity<Collection<StudentResponse>> getAllStudentData() {
        Collection<StudentResponse> studentData = studentRepository.findAll().stream().map(
                studentMapper::toResponse
        ).toList();
        return new ResponseEntity<>(studentData, HttpStatus.OK);
    }

    @Override
    public ResponseEntity<?> registerStaffAccount(RegisterRequest registerRequest) {
        try {
            User user = userMapper.toEntity(registerRequest);
            User createdUser = userRepository.save(user);
            return new ResponseEntity<>(new AccountCreationResponse(
                    userMapper.toResponse(createdUser),
                    "Create account successfully!"),
                    HttpStatus.OK
            );
        }
        catch (Exception exception) {
            return new ResponseEntity<>(new ErrorResponse("Some error occur when creating a account!"), HttpStatus.BAD_REQUEST);
        }
    }

    @Override
    public ResponseEntity<?> createStudent() {
        return null;
    }

}
