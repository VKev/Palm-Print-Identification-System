package tienthuan.service.impl;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.dto.request.StudentCreationRequest;
import tienthuan.dto.response.*;
import tienthuan.mapper.StudentMapper;
import tienthuan.mapper.UserMapper;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;
import tienthuan.model.User;
import tienthuan.model.fixed.Role;
import tienthuan.multithread.CloudRemover;
import tienthuan.repository.PalmPrintImageRepository;
import tienthuan.repository.StudentRepository;
import tienthuan.repository.UserRepository;
import tienthuan.service.def.IAdminService;
import tienthuan.service.def.ICloudinaryService;

import java.util.Collection;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AdminService implements IAdminService {

    private final UserMapper userMapper;

    private final UserRepository userRepository;

    private final StudentMapper studentMapper;

    private final StudentRepository studentRepository;

    private final PalmPrintImageRepository palmPrintImageRepository;

    private final ICloudinaryService cloudinaryService;

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
    public ResponseEntity<?> createStudent(StudentCreationRequest studentCreationRequest) {
        try {
            var savedStudent = studentRepository.save(studentMapper.toEntity(studentCreationRequest));
            return new ResponseEntity<>(studentMapper.toResponse(savedStudent), HttpStatus.OK);
        }
        catch (Exception exception) {
            return new ResponseEntity<>(new ErrorResponse("Some error occur when creating a student!"), HttpStatus.BAD_REQUEST);
        }
    }

    @Override
    public ResponseEntity<?> deleteStudent(String studentCode) {
        try {
            Student student = studentRepository.findByStudentCode(studentCode).orElseThrow();
            studentRepository.deleteById(student.getId());
            List<PalmPrintImage> palmPrintImages = palmPrintImageRepository.findAllByStudent(student);
            palmPrintImageRepository.deleteAllByStudent(student);
            // Multi thread delete file on cloud
            if (palmPrintImages != null && !palmPrintImages.isEmpty()) {
                CloudRemover cloudRemover = new CloudRemover(
                        cloudinaryService,
                        palmPrintImages.stream().map(PalmPrintImage::getImagePath).toList()
                );
                cloudRemover.start();
            }
            // ---------------
            return new ResponseEntity<>(studentMapper.toResponse(student), HttpStatus.OK);
        }
        catch (Exception exception) {
            return new ResponseEntity<>(new ErrorResponse("Some error occur when deleting a student!"), HttpStatus.BAD_REQUEST);
        }
    }

}
