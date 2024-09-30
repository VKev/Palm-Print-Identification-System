package app.controller;

import app.dto.response.Response;
import app.model.StudentUni;
import app.service.StudentDataService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.http.HttpStatus;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@ComponentScan
@RequestMapping("/admin/students")
public class StudentDataController {
    @Autowired
    private StudentDataService studentDataService;

    @GetMapping("/list")
    public ResponseEntity<Response<List<StudentUni>>> getAllStudents() {
        List<StudentUni> listStudent = studentDataService.getAllStudents();

        Response<List<StudentUni>> response = Response.<List<StudentUni>>builder()
                .message("okeoke")
                .object(listStudent)
                .httpStatus(listStudent == null ? HttpStatus.NOT_FOUND : HttpStatus.OK)
                .build();

        return new ResponseEntity<>(response, HttpStatusCode.valueOf(200));
    }
}
