package app.service;

import app.configuration.MessageConfiguration;
import app.dto.request.UpdateStaffRequest;
import app.dto.response.UserDTO;
import app.model.User;
import app.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class AdminService {

    private final UserRepository userRepository;

    private final MessageConfiguration messageConfig;

    public List<UserDTO> getAllStaffAccounts() {
        return userRepository.findAllStaffAccounts().stream().map(
                user -> UserDTO.builder()
                        .username(user.getUsername())
                        .fullname(user.getFullname())
                        .phone(user.getPhone())
                        .role(user.getRole().name())
                        .isEnable(user.isEnable())
                        .build()
        ).collect(Collectors.toList());
    }

    public String updateStaffAccount(String username, UpdateStaffRequest updateStaffRequest) {
        User user = userRepository.findByUsername(username).get();
        user.setFullname(updateStaffRequest.getFullname());
        user.setPhone(updateStaffRequest.getPhone());
        userRepository.save(user);
        return messageConfig.SUCCESS_UPDATE_STAFF_SUCCESS;
    }

    public void disableEnableStaffAccount(String username) {
        User user = userRepository.findByUsername(username).get();
        user.setEnable(!user.isEnabled());
        userRepository.save(user);
    }
}
