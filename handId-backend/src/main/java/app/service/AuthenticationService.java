package app.service;

import app.configuration.ConstantConfiguration;
import app.configuration.MessageConfiguration;
import app.dto.request.AuthenticationRequest;
import app.dto.request.RegisterRequest;
import app.dto.response.AuthenticationResponse;
import app.dto.response.UserDTO;
import app.exception.def.InvalidTokenException;
import app.exception.def.InvalidUsernamePasswordException;
import app.exception.def.NotFoundException;
import app.model.constant.Role;
import app.model.Token;
import app.model.constant.TokenType;
import app.model.User;
import app.repository.TokenRepository;
import app.repository.UserRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import java.io.IOException;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AuthenticationService {

    private final UserRepository userRepository;

    private final TokenRepository tokenRepository;

    private final PasswordEncoder passwordEncoder;

    private final JwtService jwtService;

    private final AuthenticationManager authenticationManager;

    private final UserDetailsService userDetailsService;

    private final ConstantConfiguration constant;

    private final MessageConfiguration messageConfig;


    private void saveToken(User user, String jwtToken) {
        Token token = Token.builder()
                .token(jwtToken)
                .tokenType(TokenType.BEARER)
                .expired(constant.JWT_EXPIRED_DISABLE)
                .revoked(constant.JWT_REVOKED_DISABLE)
                .username(user.getUsername())
                .build();
        tokenRepository.save(token);
    }


    private void revokeAllOldUserToken(User user) {
        List<Token> validTokenList = tokenRepository.findAllValidTokenByUser(user.getUsername());
        if (!validTokenList.isEmpty()) {
            tokenRepository.deleteAll(validTokenList);
        }
    }


    public AuthenticationResponse register(RegisterRequest registerRequest) {
        User user = User.builder()
                .username(registerRequest.getUsername())
                .fullname(registerRequest.getFullname())
                .enable(Boolean.TRUE)
                .password(passwordEncoder.encode(registerRequest.getPassword()))
                .phone(registerRequest.getPhone())
                .role(Role.USER)
                .build();

        userRepository.save(user);
        String jwtToken = jwtService.generateToken(user);
        String refreshToken = jwtService.generateRefreshToken(user);

        return AuthenticationResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .build();
    }


    public AuthenticationResponse authenticate(AuthenticationRequest authRequest)
            throws InvalidUsernamePasswordException {

        try {
            authenticationManager.authenticate(new UsernamePasswordAuthenticationToken(
                    authRequest.getUsername(), authRequest.getPassword()
            ));
        }
        catch (AuthenticationException exception) {
            throw new InvalidUsernamePasswordException(messageConfig.ERROR_USERNAME_PASSWORD_INVALID);
        }

        User user = getUser(authRequest.getUsername());

        String jwtToken = jwtService.generateToken(user); // Access token
        String refreshToken = jwtService.generateRefreshToken(user);  // Refresh token

        revokeAllOldUserToken(user);
        saveToken(user, jwtToken);

        return AuthenticationResponse.builder()
                .accessToken(jwtToken)
                .refreshToken(refreshToken)
                .build();
    }


    public AuthenticationResponse refreshToken(HttpServletRequest request, HttpServletResponse response)
            throws InvalidTokenException, IOException {
        final String authHeader = request.getHeader(constant.AUTHENTICATION_HEADER);
        if (authHeader == null ||!authHeader.startsWith(constant.AUTHENTICATION_HEADER_BEARER))
            return new AuthenticationResponse(null, null);

        final String refreshToken =authHeader.substring(constant.AUTHENTICATION_HEADER_BEARER.length());
        final String username = jwtService.extractUsername(refreshToken);

        if (username != null) {
            User user = getUser(username); // Get entity
            if(jwtService.isValidToken(refreshToken, user)) {
                String newAccessToken = jwtService.generateToken(user);
                revokeAllOldUserToken(user);
                saveToken(user, newAccessToken);
                AuthenticationResponse authResponse = AuthenticationResponse.builder()
                        .accessToken(newAccessToken)
                        .refreshToken(refreshToken)
                        .build();
                new ObjectMapper().writeValue(response.getOutputStream(), authResponse);
                return authResponse;
            }
            else throw new InvalidTokenException(messageConfig.ERROR_LOGIN_SESSION_EXPIRED);
        }
        return new AuthenticationResponse(null, null);
    }


    public UserDTO getUserInfo(HttpServletRequest request, HttpServletResponse response) {
        final String authHeader = request.getHeader(constant.AUTHENTICATION_HEADER);
        if (authHeader == null || !authHeader.startsWith(constant.AUTHENTICATION_HEADER_BEARER))
            throw new InvalidTokenException(messageConfig.ERROR_LOGIN_SESSION_EXPIRED);

        String jwt = authHeader.substring(constant.AUTHENTICATION_HEADER_BEARER.length());
        String username = jwtService.extractUsername(jwt);
        User user = (User) userDetailsService.loadUserByUsername(username);
        return UserDTO.builder()
                .username(user.getUsername())
                .fullname(user.getFullname())
                .phone(user.getPhone())
                .role(user.getRole().name())
                .build();
    }


    private User getUser(String username) {
        return userRepository.findByUsername(username).orElseThrow(
                () -> new InvalidUsernamePasswordException(messageConfig.ERROR_USERNAME_PASSWORD_INVALID)
        );
    }
}
