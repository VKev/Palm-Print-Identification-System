package tienthuan.model;

import jakarta.persistence.*;
import lombok.*;
import tienthuan.model.fixed.TokenType;

@Getter
@Setter
@Entity
@Table(name = "tokens")
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Token {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "token", unique = true, nullable = false, length = 1024)
    private String token;

    @Column(name = "token_type", nullable = false)
    private TokenType tokenType;

    @Column(name = "revoked")
    private boolean revoked;

    @Column(name = "expired")
    private boolean expired;

    @Column(name = "username", nullable = false)
    private String username;
}
