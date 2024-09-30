package app.repository;

import app.model.StudentUni;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface StudentDataRepository extends MongoRepository<StudentUni,String>{
}
