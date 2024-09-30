package app.controller;

import app.dto.request.RegisterRequest;
import app.dto.request.UpdateStaffRequest;
import app.dto.response.Response;
import app.service.AdminService;
import app.service.AuthenticationService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
@PreAuthorize("hasRole('ADMIN')")
public class AdminController {

    private final AuthenticationService authenticationService;

    private final AdminService adminService;

    @PostMapping("/register/staff-accounts")
    @PreAuthorize("hasAuthority('admin:create')")
    public ResponseEntity<Response> register(@RequestBody RegisterRequest registerRequest) {
        return ResponseEntity.ok(
                Response.builder()
                        .httpStatus(HttpStatus.OK)
                        .message("Add staff successfully!")
                        .object(authenticationService.register(registerRequest))
                        .build()
        );
    }

    @GetMapping("/get/staff-accounts")
    @PreAuthorize("hasAuthority('admin:read')")
    public ResponseEntity<Response> getAllUser() {
        return ResponseEntity.ok(
                Response.builder()
                        .httpStatus(HttpStatus.OK)
                        .message("")
                        .object(adminService.getAllStaffAccounts())
                        .build()
        );
    }

    @PutMapping("/update/staff-acccount/{username}")
    @PreAuthorize("hasAuthority('admin:update')")
    public ResponseEntity<Response> updateUser(
            @PathVariable("username") String username,
            @RequestBody UpdateStaffRequest updateStaffRequest
    ) {
        return ResponseEntity.ok(
                Response.builder()
                        .httpStatus(HttpStatus.OK)
                        .message("")
                        .object(adminService.updateStaffAccount(username, updateStaffRequest))
                        .build()
        );
    }

    @PutMapping("/disable/staff-acccount/{username}")
    @PreAuthorize("hasAuthority('admin:update')")
    public ResponseEntity<Response> disableEnableUser(@PathVariable("username") String username) {
        adminService.disableEnableStaffAccount(username);
        return ResponseEntity.ok(Response.builder()
                .httpStatus(HttpStatus.OK)
                .message("")
                .build()
        );
    }
}
