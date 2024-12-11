package tienthuan.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@PropertySource("classpath:security.properties")
public class ConstantConfiguration {

    public final Long JWT_TOKEN_EXPIRATION;
    public final Long REFRESH_TOKEN_EXPIRATION;
    public final String SECRET_KEY;
    public final String AUTHENTICATION_HEADER = "Authorization";
    public final String AUTHENTICATION_HEADER_BEARER = "Bearer ";
    public final Boolean JWT_EXPIRED_ENABLE = true;
    public final Boolean JWT_EXPIRED_DISABLE = false;
    public final Boolean JWT_REVOKED_ENABLE = true;
    public final Boolean JWT_REVOKED_DISABLE = false;
    public final String LOGOUT_HANDLER_URL = "/api/logout";

    // Video extractor properties
    public final Integer IMAGES_FRAME_JUMP = 6;
    public final Integer FRAMES_LIMITATION = 30;
    public final String IMAGES_EXTENSION_TYPE = ".png";


    public ConstantConfiguration(
            @Value("${jwt.secret-key}") String jwtSecretKey,
            @Value("${jwt.expiration}") Long jwtExpiration,
            @Value("${jwt.refresh-token.expiration}") Long refreshTokenExpiration
    ){
        this.JWT_TOKEN_EXPIRATION = jwtExpiration;
        this.REFRESH_TOKEN_EXPIRATION = refreshTokenExpiration;
        this.SECRET_KEY = jwtSecretKey;
    }

}
