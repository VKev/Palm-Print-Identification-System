package tienthuan.mapper;

import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.dto.response.UserResponse;
import tienthuan.model.User;
import tienthuan.model.fixed.Role;

@Component
@RequiredArgsConstructor
public class UserMapper {

    private final PasswordEncoder passwordEncoder;

    public User toEntity(RegisterRequest registerRequest) {
        return User.builder()
                .username(registerRequest.username())
                .fullname(registerRequest.fullname())
                .password(passwordEncoder.encode(registerRequest.password()))
                .isEnable(Boolean.TRUE)
                .role(Role.STAFF)
                .build();
    }

    public UserResponse toResponse(User user) {
        return UserResponse.builder()
                .id(user.getId())
                .username(user.getUsername())
                .fullname(user.getFullname())
                .role(user.getRole().name())
                .build();
    }

}
