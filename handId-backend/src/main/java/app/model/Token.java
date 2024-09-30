package app.model;

import app.model.constant.TokenType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.data.mongodb.core.mapping.Field;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "tokens")
public class Token {

    @Id
    private String id;

    @Indexed(unique = true)
    @Field(name = "token")
    private String token;

    @Field(name = "token_type")
    private TokenType tokenType;

    @Field(name = "revoked")
    private boolean revoked;

    @Field(name = "expired")
    private boolean expired;

    @Field(name = "username")
    private String username;
}
