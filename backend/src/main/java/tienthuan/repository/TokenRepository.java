package tienthuan.repository;

import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;
import tienthuan.model.Token;
import tienthuan.model.User;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

@Repository
public interface TokenRepository extends JpaRepository<Token, Long> {

    Optional<Token> findByToken(String token);


    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query(
            "SELECT t from Token t " +
                    "INNER JOIN User u " +
                    "ON t.user.id = u.id " +
                    "WHERE u.id =:userId AND (t.expired = false OR t.revoked = false)"
    )
    List<Token> findAllValidTokenByUsername(Long userId);

    @Transactional(isolation = Isolation.REPEATABLE_READ)
    @Modifying
    void deleteAllByUserAndExpiredAndRevoked(User user, boolean expired, boolean revoked);
}
