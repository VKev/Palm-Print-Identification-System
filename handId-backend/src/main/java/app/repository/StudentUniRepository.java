package app.repository;

import app.model.StudentUni;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import java.util.Optional;

@Repository
public interface StudentUniRepository extends MongoRepository<StudentUni, String> {

    @Query("{'roleNumber': ?0}")
    Optional<StudentUni> findByRoleNumber(String roleNumber);

}
