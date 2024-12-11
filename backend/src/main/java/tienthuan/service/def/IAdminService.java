package tienthuan.service.def;

import org.springframework.http.ResponseEntity;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.dto.response.StudentResponse;
import tienthuan.dto.response.UserResponse;

import java.util.Collection;

public interface IAdminService {

    ResponseEntity<Collection<UserResponse>> getAllStaffAccounts();

    ResponseEntity<Collection<StudentResponse>> getAllStudentData();

    ResponseEntity<?> registerStaffAccount(RegisterRequest registerRequest);

    ResponseEntity<?> createStudent();

}
