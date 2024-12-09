package tienthuan.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import tienthuan.model.Token;
import java.util.Collection;
import java.util.Optional;

@Repository
public interface TokenRepository extends JpaRepository<Token, Long> {

    Optional<Token> findByToken(String token);

    @Query(
            "SELECT t FROM Token t WHERE t.username = :username AND t.revoked = false AND t.expired = false"
    )
    Collection<Token> findAllValidTokenByUsername(String username);

}
