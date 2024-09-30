package app.repository;

import app.model.ImageData;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface ImageDataRepository extends MongoRepository<ImageData, String> {
}
