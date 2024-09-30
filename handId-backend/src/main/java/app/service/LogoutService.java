package app.service;

import app.configuration.ConstantConfiguration;
import app.model.Token;
import app.repository.TokenRepository;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.logout.LogoutHandler;
import org.springframework.stereotype.Service;

@Service
public class LogoutService implements LogoutHandler {

    @Autowired
    private TokenRepository tokenRepository;

    @Autowired
    private ConstantConfiguration constant;

    @Override
    public void logout(HttpServletRequest request, HttpServletResponse response, Authentication authentication) {
        final String authHeader = request.getHeader(constant.AUTHENTICATION_HEADER);
        if (authHeader == null || !authHeader.startsWith(constant.AUTHENTICATION_HEADER_BEARER))
            return;
        String jwt = authHeader.substring(constant.AUTHENTICATION_HEADER_BEARER.length()); // Access token
        Token token = tokenRepository.findByToken(jwt).orElse(null);
        if(token != null) {
            token.setExpired(constant.JWT_EXPIRED_ENABLE);
            token.setRevoked(constant.JWT_REVOKED_ENABLE);
            tokenRepository.save(token);
            SecurityContextHolder.clearContext();
        }
    }

}
