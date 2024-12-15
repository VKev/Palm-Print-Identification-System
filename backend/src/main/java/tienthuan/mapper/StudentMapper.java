package tienthuan.mapper;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import tienthuan.dto.request.StudentCreationRequest;
import tienthuan.dto.response.StudentResponse;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;
import tienthuan.repository.PalmPrintImageRepository;

import java.util.List;

@Component
public class StudentMapper {

    @Autowired
    private PalmPrintImageRepository palmPrintImageRepository;

    public Student toEntity(StudentCreationRequest request) {
        return Student.builder()
                .studentCode(request.getStudentCode())
                .studentName(request.getStudentName())
                .isRegistered(Boolean.FALSE)
                .build();
    }

    public StudentResponse toResponse(Student student) {
        List<PalmPrintImage> palmPrintImages = palmPrintImageRepository.findAllByStudent(student);
        return StudentResponse.builder()
                .id(student.getId())
                .studentCode(student.getStudentCode())
                .studentName(student.getStudentName())
                .isRegistered(student.getIsRegistered())
                .imagePaths(palmPrintImages.stream().map(PalmPrintImage::getImagePath).toList())
                .build();
    }

}
