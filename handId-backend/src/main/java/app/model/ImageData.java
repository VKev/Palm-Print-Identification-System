package app.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bson.types.Binary;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

import java.util.Collection;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "student_hand_images")
public class ImageData {

    @Id
    @Field(name = "role_number")
    private String roleNumber;

    @Field(name = "hand_images")
    private Collection<Binary> handImages;

}
