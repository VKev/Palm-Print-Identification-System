package tienthuan.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import tienthuan.model.PalmPrintImage;

@Repository
public interface PalmPrintImageRepository extends JpaRepository<PalmPrintImage, Long> {
}
