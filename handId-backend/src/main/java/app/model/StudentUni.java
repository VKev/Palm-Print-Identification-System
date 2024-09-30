package app.model;


import lombok.*;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.bson.types.ObjectId;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@ToString
@Document(collection = "students_uni")
public class StudentUni {

    @Id
    private ObjectId id;

    @Field(name = "roleNumber")
    private String roleNumber;

    @Field(name = "fullname")
    private String fullname;
}
