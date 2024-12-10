package tienthuan.service.impl;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.CannotAcquireLockException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import tienthuan.configuration.ConstantConfiguration;
import tienthuan.dto.request.AuthenticationRequest;
import tienthuan.dto.request.RegisterRequest;
import tienthuan.dto.response.AuthenticationResponse;
import tienthuan.dto.response.ErrorResponse;
import tienthuan.mapper.UserMapper;
import tienthuan.model.Token;
import tienthuan.model.User;
import tienthuan.model.fixed.TokenType;
import tienthuan.repository.TokenRepository;
import tienthuan.repository.UserRepository;
import tienthuan.service.def.IAuthenticationService;
import java.io.IOException;
import java.util.Collection;

@Service
@RequiredArgsConstructor
public class AuthenticationService implements IAuthenticationService {

    private final UserRepository userRepository;

    private final TokenRepository tokenRepository;

    private final UserMapper userMapper;

    private final JwtService jwtService;

    private final AuthenticationManager authenticationManager;

    private final UserDetailsService userDetailsService;

    private final ConstantConfiguration constant;

    @Override
    public ResponseEntity<?> register(RegisterRequest registerRequest) {
        try {
            User user = userMapper.toEntity(registerRequest);
            userRepository.save(user);
            AuthenticationResponse authenticationResponse = AuthenticationResponse.builder()
                    .accessToken(jwtService.generateToken(user))
                    .refreshToken(jwtService.generateRefreshToken(user))
                    .build();
            return new ResponseEntity<>(authenticationResponse, HttpStatus.OK);
        } catch (Exception exception) {
            return new ResponseEntity<>(
                    new ErrorResponse("Some error occur when creating a account!"), HttpStatus.BAD_REQUEST);
        }
    }

    @Override
    public ResponseEntity<?> authenticate(AuthenticationRequest authRequest) {
        try {
            authenticationManager.authenticate(new UsernamePasswordAuthenticationToken(
                    authRequest.username(), authRequest.password()));
        } catch (AuthenticationException exception) {
            return new ResponseEntity<>(new ErrorResponse("Invalid username or password!"), HttpStatus.UNAUTHORIZED);
        }

        try {
            User user = getUser(authRequest.username());
            String jwtToken = jwtService.generateToken(user);
            String refreshToken = jwtService.generateRefreshToken(user);
            revokeAllOldUserToken(user);
            saveToken(user, jwtToken);
            AuthenticationResponse authenticationResponse = AuthenticationResponse.builder()
                    .accessToken(jwtToken)
                    .refreshToken(refreshToken)
                    .build();
            return new ResponseEntity<>(authenticationResponse, HttpStatus.OK);
        }
        catch (Exception exception) {
            return new ResponseEntity<>(new ErrorResponse("An error occur while login!"), HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @Override
    public ResponseEntity<?> refreshToken(HttpServletRequest request, HttpServletResponse response) throws IOException {
        final String authHeader = request.getHeader(constant.AUTHENTICATION_HEADER);
        if (authHeader == null || !authHeader.startsWith(constant.AUTHENTICATION_HEADER_BEARER)) {
            return new ResponseEntity<>(
                    new AuthenticationResponse(null, null), HttpStatus.UNAUTHORIZED);
        }

        final String refreshToken = authHeader.substring(constant.AUTHENTICATION_HEADER_BEARER.length());
        final String username = jwtService.extractUsername(refreshToken);

        if (username != null) {
            User user = getUser(username);
            if (jwtService.isValidToken(refreshToken, user)) {
                String newAccessToken = jwtService.generateToken(user);
                revokeAllOldUserToken(user);
                saveToken(user, newAccessToken);
                AuthenticationResponse authResponse = AuthenticationResponse.builder()
                        .accessToken(newAccessToken)
                        .refreshToken(refreshToken)
                        .build();
                // new ObjectMapper().writeValue(response.getOutputStream(), authResponse);
                return new ResponseEntity<>(authResponse, HttpStatus.OK);
            } else {
                return new ResponseEntity<>(
                        new ErrorResponse("Login session time expired!"), HttpStatus.UNAUTHORIZED);
            }
        }
        return new ResponseEntity<>(
                new AuthenticationResponse(null, null), HttpStatus.UNAUTHORIZED);
    }

    @Override
    public ResponseEntity<?> getUser(HttpServletRequest request, HttpServletResponse response) {
        final String authHeader = request.getHeader(constant.AUTHENTICATION_HEADER);
        if (authHeader == null || !authHeader.startsWith(constant.AUTHENTICATION_HEADER_BEARER))
            return this.getLoginSessionExpiredResponse();

        String jwt = authHeader.substring(constant.AUTHENTICATION_HEADER_BEARER.length());
        String username = jwtService.extractUsername(jwt);
        User user = (User) userDetailsService.loadUserByUsername(username);
        return new ResponseEntity<>(userMapper.toResponse(user), HttpStatus.OK);
    }

    private void saveToken(User user, String jwtToken) {
        Token token = Token.builder()
                .token(jwtToken)
                .tokenType(TokenType.BEARER)
                .expired(constant.JWT_EXPIRED_DISABLE)
                .revoked(constant.JWT_REVOKED_DISABLE)
                .user(user)
                .build();
        tokenRepository.save(token);
    }

    @Transactional
    public void revokeAllOldUserToken(User user) {
        int maxRetries = 3; int retries = 0; boolean success = false;
        while (!success && retries < maxRetries) {
            try {
                tokenRepository.deleteAllByUserAndExpiredAndRevoked(user, false, false);
                success = true;
            }
            catch (CannotAcquireLockException e) {
                retries++;
                if (retries >= maxRetries) {
                    throw e;
                }
            }
        }
    }

    private User getUser(String username) {
        return userRepository.findByUsername(username).orElseThrow(null);
    }

    private ResponseEntity<?> getLoginSessionExpiredResponse() {
        return new ResponseEntity<>(
                new ErrorResponse("Login session time expired!"), HttpStatus.UNAUTHORIZED);
    }


//    private void revokeAllOldUserToken(User user) {
//        Collection<Token> validTokenList = tokenRepository.findAllValidTokenByUsername(user.getId());
//        if (validTokenList != null && !validTokenList.isEmpty()) {
//            for (Token token : validTokenList) {
//                tokenRepository.deleteById(token.getId());
//            }
//        }
//        tokenRepository.deleteAllByUserAndExpiredAndRevoked(user, false, false);
//    }
    
}
