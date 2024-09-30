package app.service;

import app.model.StudentUni;

import app.repository.StudentDataRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.List;

@Service
public class StudentDataService {

    @Autowired
    private StudentDataRepository studentDataRepository;

    // Phương thức để lấy tất cả dữ liệu từ collection
    public List<StudentUni> getAllStudents() {
        return studentDataRepository.findAll();
    }
}
