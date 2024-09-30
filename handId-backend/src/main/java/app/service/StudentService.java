package app.service;

import app.configuration.MessageConfiguration;
import app.dto.response.StudentChecking;
import app.model.StudentUni;
import app.repository.StudentUniRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.sql.SQLOutput;

@Service
@RequiredArgsConstructor
public class StudentService {

    public final StudentUniRepository uniRepository;

    private final MessageConfiguration messageConfig;

    public StudentChecking checkStudentRoleNumber(String roleNumber) {
        var studentUni = uniRepository.findByRoleNumber(roleNumber);
        boolean exist = true; String message;
        if (!studentUni.isPresent()) {
            exist = false;  message = messageConfig.ERROR_STUDENT_ROLENUMBER_NOT_EXIST;
        }
        else message = messageConfig.SUCCESS_STUDENT_ROLENUMBER_VALID;
        return StudentChecking.builder()
                .isExist(exist)
                .message(message)
                .build();
    }

}
