package tienthuan.mapper;

import org.springframework.stereotype.Component;
import tienthuan.dto.response.StudentResponse;
import tienthuan.model.Student;

@Component
public class StudentMapper {

    public StudentResponse toResponse(Student student) {
        return StudentResponse.builder()
                .id(student.getId())
                .studentCode(student.getStudentCode())
                .studentName(student.getStudentName())
                .isRegistered(student.getIsRegistered())
                //.palmPrintImages(student.getPalmPrintImages())
                .build();
    }

}
