package tienthuan.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import tienthuan.model.History;

@Repository
public interface HistoryRepository extends JpaRepository<History, Long> {
}
