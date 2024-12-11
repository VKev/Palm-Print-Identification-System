package tienthuan.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;

import java.util.List;

@Repository
public interface PalmPrintImageRepository extends JpaRepository<PalmPrintImage, Long> {

    List<PalmPrintImage> findAllByStudent(Student student);

}
