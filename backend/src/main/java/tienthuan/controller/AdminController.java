package tienthuan.controller;

import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.dto.request.StudentCreationRequest;
import tienthuan.dto.response.StudentResponse;
import tienthuan.dto.response.UserResponse;
import tienthuan.service.def.IAdminService;
import java.util.Collection;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/admin")
public class AdminController {

    private final IAdminService adminService;

    @GetMapping("/staff-accounts/get-all")
    public ResponseEntity<Collection<UserResponse>> getAllStaffAccounts() {
        return adminService.getAllStaffAccounts();
    }

    @GetMapping("/student-data/get-all")
    public ResponseEntity<Collection<StudentResponse>> getAllStudentData() {
        return adminService.getAllStudentData();
    }

    @PostMapping("/staff-accounts/register")
    public ResponseEntity<?> registerStaffAccount(@RequestBody RegisterRequest registerRequest) {
        return adminService.registerStaffAccount(registerRequest);
    }

    @PostMapping("/student-data/create")
    public ResponseEntity<?> createStudent(@RequestBody StudentCreationRequest studentCreationRequest) {
        return adminService.createStudent(studentCreationRequest);
    }

    @DeleteMapping("/student-data/delete-student/{studentCode}")
    public ResponseEntity<?> deleteStudent(@PathVariable("studentCode") String studentCode) {
        return adminService.deleteStudent(studentCode);
    }
}
