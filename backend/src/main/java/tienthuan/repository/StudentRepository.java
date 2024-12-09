package tienthuan.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import tienthuan.model.Student;

@Repository
public interface StudentRepository extends JpaRepository<Student, Long> {
}
