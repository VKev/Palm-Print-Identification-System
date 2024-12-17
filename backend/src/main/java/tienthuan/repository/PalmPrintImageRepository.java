package tienthuan.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;
import tienthuan.model.PalmPrintImage;
import tienthuan.model.Student;

import java.util.List;

@Repository
public interface PalmPrintImageRepository extends JpaRepository<PalmPrintImage, Long> {

    List<PalmPrintImage> findAllByStudent(Student student);

    @Modifying
    @Transactional(isolation = Isolation.REPEATABLE_READ)
    void deleteAllByStudent(Student student);

}
